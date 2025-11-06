#include <cstdint>

#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Arith/FPSqrt.h"
#include "Arith/FPSub.h"
#include "Func/LayerNorm.h"
#include "Utils/Config.h"
#include "Utils/ConstDiv.h"
#include "Utils/FormatUtils.h"
#include "Utils/HlsVector.h"

void layer_norm(uint16_t *x_bf16, uint16_t *y_bf16, const int len, float eps)
{
    // ============================================================
    // Input conversion: bf16 → 25-bit (convert once)
    // ============================================================
    uint32_t x_25bit[len];
    bool has_special = false;

    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        x_25bit[i] = float_to_extended_25bit(bf16_to_float(x_bf16[i]));

        // Check special values
        ValueType x_type = classify_25bit(x_25bit[i]);
        if (x_type == ValueType::NaN || x_type == ValueType::PosInf ||
            x_type == ValueType::NegInf) {
            has_special = true;
            break;
        }
    }

    // If there are special values, output all NaN
    if (has_special) {
        uint16_t nan_bf16 = 0x7FC0;

        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
            y_bf16[i] = nan_bf16;
        }
        return;
    }

    // ============================================================
    // Stage 1: Compute mean - tree-based summation optimization
    // For D=768 = 3 × 2^8, use tree structure to reduce accumulation error
    // Formula: μ = (s1 + s2 + s3) / 3
    // where s1, s2, s3 are three sub-sums obtained through 8-level binary summation
    // ============================================================

    FPAdd fadd_module;
    FPMult mult_module;
    uint32_t mean_25bit;

#ifndef __SYNTHESIS__
    assert(len <= MAX_CAPACITY && "Runtime len exceeds hardware capacity!");
#endif

    HlsVector<uint32_t, MAX_CAPACITY> buffers[2];
    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        buffers[0].push_back(x_25bit[i]);
    }

    // Perform 8-level binary summation (768 → 384 → 192 → 96 → 48 → 24 → 12 → 6 → 3)
    int current_size = len;

    for (int level = 0; level < 8; level++) {
#pragma HLS PIPELINE II = 1
        int read_buf_idx = level % 2;
        int write_buf_idx = (level + 1) % 2;

        // Ready to write data to buffer
        buffers[write_buf_idx].clear();

        int pairs = current_size / 2;
        int odd_node_exists = current_size % 2;

        // Add each pair
        for (int i = 0; i < pairs; i++) {
#pragma HLS PIPELINE II = 1
            fadd_module.reset();
            fadd_module.set_input(buffers[read_buf_idx][2 * i], buffers[read_buf_idx][2 * i + 1]);
            fadd_module.clock_step();
            fadd_module.clock_step();
            uint32_t pair_sum = fadd_module.get_result_extended();

            // Right shift 1 bit (divide by 2, implemented by decreasing exponent by 1)
            // 25-bit format: [exn:2][sign:1][exp:8][mantissa:14]
            uint32_t exn = (pair_sum >> 23) & 0x3;
            uint32_t sign = (pair_sum >> 22) & 0x1;
            uint32_t exp = (pair_sum >> 14) & 0xFF;
            uint32_t mantissa = pair_sum & 0x3FFF;

            // Divide by 2: exp -= 1 (only for normal numbers)
            if (exn == 1 && exp > 0) {  // Normal number and exponent is not zero
                exp -= 1;
                uint32_t halved = (exn << 23) | (sign << 22) | (exp << 14) | mantissa;
                buffers[write_buf_idx].push_back(halved);
            } else {
                // Special values or zero, keep unchanged
                buffers[write_buf_idx].push_back(pair_sum);
            }
        }

        // If there are odd number of elements, pass the last one directly (no need to divide by 2)
        if (odd_node_exists == 1) {
            buffers[write_buf_idx].push_back(buffers[read_buf_idx][current_size - 1]);
        }

        current_size = pairs + odd_node_exists;
    }

    // Now buffers[0](write_buf_idx = 0 when level = 7) should have 3 elements (because 768 = 3 × 256), add the last 3 elements
    fadd_module.reset();
    fadd_module.set_input(buffers[0][0], buffers[0][1]);
    fadd_module.clock_step();
    fadd_module.clock_step();
    uint32_t sum_01 = fadd_module.get_result_extended();

    fadd_module.reset();
    fadd_module.set_input(sum_01, buffers[0][2]);
    fadd_module.clock_step();
    fadd_module.clock_step();
    uint32_t sum_25bit = fadd_module.get_result_extended();

    // Multiply by 1/3 (using ConstDiv lookup table)
    uint32_t inv_3_25bit = ConstDiv::get_inv_k_25bit(3);

    mult_module.reset();
    mult_module.set_input(sum_25bit, inv_3_25bit);
    mult_module.clock_step();
    mult_module.clock_step();
    mean_25bit = mult_module.get_result();

    // ============================================================
    // Stage 2: Compute variance - traditional two-pass method
    // ============================================================

    FPSub fpsub_module;
    uint32_t var_sum_25bit = 0U;  // Initialize to 0

    for (int j = 0; j < len; j++) {
#pragma HLS PIPELINE II = 1
        // Compute x[j] - mean
        fpsub_module.reset();
        fpsub_module.set_input(x_25bit[j], mean_25bit);
        for (int cycle = 0; cycle < 2; cycle++) {
            fpsub_module.clock_step();
        }
        uint32_t diff_25bit = fpsub_module.get_result_extended();

        // Compute (x[j] - mean)^2
        mult_module.reset();
        mult_module.set_input(diff_25bit, diff_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t diff_sq_25bit = mult_module.get_result();

        // Accumulate to var_sum
        fadd_module.reset();
        fadd_module.set_input(var_sum_25bit, diff_sq_25bit);
        for (int cycle = 0; cycle < 2; cycle++) {
            fadd_module.clock_step();
        }
        var_sum_25bit = fadd_module.get_result_extended();
    }

    // variance = var_sum / D (using ConstDiv FP32-based lookup table)
    uint32_t inv_len_25bit_var = ConstDiv::get_inv_k_25bit(len);

    mult_module.reset();
    mult_module.set_input(var_sum_25bit, inv_len_25bit_var);
    mult_module.clock_step();
    mult_module.clock_step();
    uint32_t variance_25bit = mult_module.get_result();

    // ============================================================
    // Stage 3: Compute reciprocal of standard deviation std_dev_inv = 1/sqrt(variance + eps)
    // ============================================================

    // variance + eps
    uint32_t eps_25bit = float_to_extended_25bit(eps);
    fadd_module.reset();
    fadd_module.set_input(variance_25bit, eps_25bit);
    for (int cycle = 0; cycle < 2; cycle++) {
        fadd_module.clock_step();
    }
    uint32_t variance_plus_eps = fadd_module.get_result_extended();

    // Compute 1/sqrt(variance + eps) using FPSqrt module
    FPSqrt sqrt_module;
    sqrt_module.set_input(variance_plus_eps);

    // FPSqrt requires multiple cycles (state machine driven)
    while (!sqrt_module.get_valid_out()) {
        sqrt_module.clock_step();
    }

    uint32_t std_dev_inv_25bit = sqrt_module.get_result_extended();

    // ============================================================
    // Stage 4: Normalize y[i] = (x[i] - mean) * std_dev_inv
    // ============================================================

    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        // Compute x[i] - mean (using tree-summed mean)
        fpsub_module.reset();
        fpsub_module.set_input(x_25bit[i], mean_25bit);

        for (int cycle = 0; cycle < 2; cycle++) {
            fpsub_module.clock_step();
        }

        uint32_t diff_25bit = fpsub_module.get_result_extended();

        // Compute (x[i] - mean) * std_dev_inv
        mult_module.reset();
        mult_module.set_input(diff_25bit, std_dev_inv_25bit);
        mult_module.clock_step();
        mult_module.clock_step();

        uint32_t result_25bit = mult_module.get_result();

        // Output conversion: 25-bit → bf16
        y_bf16[i] = extended_25bit_to_bf16(result_25bit);
    }
}
