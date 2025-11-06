#include "Arith/FPAdd.h"
#include "Func/EltwiseAdd.h"
#include "Utils/FormatUtils.h"

void eltwise_add(uint16_t *x0_bf16, uint16_t *x1_bf16, uint16_t *y_bf16, const int len)
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
    FPAdd fpadd_module;

    for (int i = 0; i < len; i++) {
        // Use classify_25bit to check special values
        ValueType x0_type = classify_25bit(x0_25bit[i]);
        ValueType x1_type = classify_25bit(x1_25bit[i]);

        // Special value handling (IEEE 754 standard)
        // NaN + anything = NaN
        if (x0_type == ValueType::NaN || x1_type == ValueType::NaN) {
            // NaN in 25-bit: exn=3, sign=0, exp=0xFF, mantissa!=0
            y_bf16[i] = 0x7FC0;
            continue;
        }

        // +Inf + (-Inf) = NaN
        if ((x0_type == ValueType::PosInf && x1_type == ValueType::NegInf) ||
            (x0_type == ValueType::NegInf && x1_type == ValueType::PosInf)) {
            y_bf16[i] = 0x7FC0;
            continue;
        }

        // +Inf + anything (except -Inf) = +Inf
        if (x0_type == ValueType::PosInf || x1_type == ValueType::PosInf) {
            // +Inf in 25-bit: exn=2, sign=0, exp=0xFF, mantissa=0
            uint32_t pos_inf_25bit = (2U << 23) | (0xFF << 14);
            float pos_inf_float = extended_25bit_to_float(pos_inf_25bit);
            y_bf16[i] = float_to_bf16(pos_inf_float);
            continue;
        }

        // -Inf + anything (except +Inf) = -Inf
        if (x0_type == ValueType::NegInf || x1_type == ValueType::NegInf) {
            // -Inf in 25-bit: exn=2, sign=1, exp=0xFF, mantissa=0
            uint32_t neg_inf_25bit = (2U << 23) | (1U << 22) | (0xFF << 14);
            float neg_inf_float = extended_25bit_to_float(neg_inf_25bit);
            y_bf16[i] = float_to_bf16(neg_inf_float);
            continue;
        }

        // Normal case: use FPAdd hardware module (25-bit precision)
        fpadd_module.reset();
        fpadd_module.set_input(x0_25bit[i], x1_25bit[i]);

        for (int cycle = 0; cycle < 2; cycle++) {
            fpadd_module.clock_step();
        }

        uint32_t result_25bit = fpadd_module.get_result_extended();

        // Output conversion: 25-bit → bf16
        y_bf16[i] = extended_25bit_to_bf16(result_25bit);
    }
}
