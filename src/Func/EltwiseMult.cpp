#include "Arith/FPMult.h"
#include "Func/EltwiseMult.h"
#include "Utils/FormatUtils.h"

void eltwise_mult(uint16_t *x0_bf16, uint16_t *x1_bf16, uint16_t *y_bf16, const int len)
{
    // ============================================================
    // Input conversion: bf16 → float → 25-bit (convert once)
    // ============================================================
    uint32_t x0_25bit[len];
    uint32_t x1_25bit[len];

    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        x0_25bit[i] = float_to_extended_25bit(bf16_to_float(x0_bf16[i]));
        x1_25bit[i] = float_to_extended_25bit(bf16_to_float(x1_bf16[i]));
    }

    // ============================================================
    // Process each element (full 25-bit precision)
    // ============================================================
    FPMult mult_module;

    for (int i = 0; i < len; i++) {
        // Use classify_25bit to check special values
        ValueType x0_type = classify_25bit(x0_25bit[i]);
        ValueType x1_type = classify_25bit(x1_25bit[i]);

        // Special value handling (IEEE 754 standard)
        // NaN * anything = NaN
        if (x0_type == ValueType::NaN || x1_type == ValueType::NaN) {
            // NaN in 25-bit: exn=3, sign=0, exp=0xFF, mantissa!=0
            y_bf16[i] = 0x7FC0;
            continue;
        }

        // 0 * Inf = NaN
        if ((x0_type == ValueType::Zero &&
             (x1_type == ValueType::PosInf || x1_type == ValueType::NegInf)) ||
            (x1_type == ValueType::Zero &&
             (x0_type == ValueType::PosInf || x0_type == ValueType::NegInf))) {
            y_bf16[i] = 0x7FC0;
            continue;
        }

        // Inf * Inf = Inf (sign determined by both operands)
        // Inf * normal = Inf (sign determined by both operands)
        if ((x0_type == ValueType::PosInf || x0_type == ValueType::NegInf) ||
            (x1_type == ValueType::PosInf || x1_type == ValueType::NegInf)) {
            // Extract sign bits
            bool sign0 = (x0_25bit[i] >> 22) & 0x1;
            bool sign1 = (x1_25bit[i] >> 22) & 0x1;
            bool result_sign = sign0 ^ sign1;  // XOR for multiplication

            // Construct ±Inf
            uint32_t inf_25bit = (2U << 23) | (result_sign << 22) | (0xFF << 14);
            y_bf16[i] = extended_25bit_to_bf16(inf_25bit);
            continue;
        }

        // 0 * anything (except Inf/NaN) = 0
        if (x0_type == ValueType::Zero || x1_type == ValueType::Zero) {
            y_bf16[i] = 0x0000;  // bf16 zero
            continue;
        }

        // Normal case: use FPMult hardware module (25-bit precision)
        mult_module.set_input(x0_25bit[i], x1_25bit[i]);
        mult_module.clock_step();  // Cycle 0
        mult_module.clock_step();  // Cycle 1

        uint32_t result_25bit = mult_module.get_result();

        // Output conversion: 25-bit → bf16
        y_bf16[i] = extended_25bit_to_bf16(result_25bit);
    }
}
