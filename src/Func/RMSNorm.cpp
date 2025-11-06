#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Arith/FPSqrt.h"
#include "Func/RMSNorm.h"
#include "Utils/ConstDiv.h"
#include "Utils/FormatUtils.h"
#include "Utils/HlsVector.h"

const int MAX_CAPACITY = 768;

void rms_norm(uint16_t *x_bf16, uint16_t *y_bf16, const int len, float eps)
{
    // ============================================================
    // Input conversion: bf16 → float → 25-bit (convert once)
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
        }
    }

    // If there are special values, output all NaN
    if (has_special) {
        uint16_t nan_bf16 = 0x7FC0;

        for (int i = 0; i < len; i++) {
            y_bf16[i] = nan_bf16;
        }
        return;
    }

    // ============================================================
    // Stage 1: Compute mean of squared values - tree-based summation optimization
    // For D=768 = 3 × 2^8, use tree structure to reduce accumulation error
    // Formula: mean(x^2) = (s1 + s2 + s3) / 3
    // where s1, s2, s3 are three sub-sums obtained through 8-level binary summation
    // ============================================================

    FPMult mult_module;
    FPAdd fadd_module;
    uint32_t mean_sq_25bit;

    // First compute all x[i]^2
    uint32_t x_sq_25bit[len];
    for (int i = 0; i < len; i++) {
        mult_module.set_input(x_25bit[i], x_25bit[i]);
        mult_module.clock_step();
        mult_module.clock_step();
        x_sq_25bit[i] = mult_module.get_result();
    }

#ifndef __SYNTHESIS__
    assert(len <= MAX_CAPACITY && "Runtime len exceeds hardware capacity!");
#endif

    HlsVector<uint32_t, MAX_CAPACITY> buffers[2];
    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        buffers[0].push_back(x_sq_25bit[i]);
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
    uint32_t sum_sq_25bit = fadd_module.get_result_extended();

    // Multiply by 1/3 (using ConstDiv lookup table)
    uint32_t inv_3_25bit = ConstDiv::get_inv_k_25bit(3);

    mult_module.reset();
    mult_module.set_input(sum_sq_25bit, inv_3_25bit);
    mult_module.clock_step();
    mult_module.clock_step();
    mean_sq_25bit = mult_module.get_result();

    // ============================================================
    // Stage 2: Compute RMS_inv = 1/sqrt(mean(x^2) + eps) (using FPSqrt)
    // ============================================================

    // Compute mean_sq + eps
    float mean_sq = extended_25bit_to_float(mean_sq_25bit);
    float var_eps = mean_sq + eps;
    uint32_t var_eps_25bit = float_to_extended_25bit(var_eps);

    // Compute 1/sqrt(variance) (FPSqrt is state machine driven)
    FPSqrt sqrt_module;
    sqrt_module.set_input(var_eps_25bit);
    while (!sqrt_module.get_valid_out()) {
        sqrt_module.clock_step();
    }
    uint32_t rms_inv_25bit = sqrt_module.get_result_extended();

    float rms_inv = extended_25bit_to_float(rms_inv_25bit);

    // ============================================================
    // Stage 4: Normalize y[i] = x[i] * rms_inv (using FPMult)
    // ============================================================

    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        // Compute x[i] * rms_inv (using FPMult)
        mult_module.reset();
        mult_module.set_input(x_25bit[i], rms_inv_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t result_25bit = mult_module.get_result();

        // Output conversion: 25-bit → bf16
        y_bf16[i] = extended_25bit_to_bf16(result_25bit);
    }
}
