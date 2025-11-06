#include "Arith/Exp.h"
#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Func/GELU.h"
#include "Func/Sigmoid.h"
#include "Utils/FormatUtils.h"

void gelu(uint16_t *x_bf16, uint16_t *y_bf16, const int len)
{
    // ============================================================
    // Input conversion: bf16 → float → 25-bit (convert once)
    // ============================================================
    uint32_t x_25bit[len];
    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        x_25bit[i] = float_to_extended_25bit(bf16_to_float(x_bf16[i]));
    }

    // ============================================================
    // Precompute constants
    // ============================================================
    const float SIGMOID_SCALE = 1.702f;
    uint32_t const_1702_25bit = float_to_extended_25bit(SIGMOID_SCALE);

    // ============================================================
    // Process each element (full 25-bit precision)
    // ============================================================
    FPMult mult_module;
    Exp exp_module;
    FPAdd fpadd_module;

    for (int i = 0; i < len; i++) {
        // Check special values
        ValueType x_type = classify_25bit(x_25bit[i]);

        // GELU(NaN) = NaN
        if (x_type == ValueType::NaN) {
            y_bf16[i] = 0x7FC0;
            continue;
        }

        // GELU(+∞) = +∞ * σ(+∞) = +∞ * 1 = +∞
        if (x_type == ValueType::PosInf) {
            y_bf16[i] = 0x7F80;
            continue;
        }

        // GELU(-∞) = -∞ * σ(-∞) = -∞ * 0 = NaN (undefined)
        // But according to GELU definition, GELU(-∞) = 0
        if (x_type == ValueType::NegInf) {
            y_bf16[i] = 0x0000;
            continue;
        }

        // ============================================================
        // Step 1: Compute 1.702 * x
        // ============================================================
        mult_module.reset();
        mult_module.set_input(const_1702_25bit, x_25bit[i]);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t scaled_x_25bit = mult_module.get_result();

        // ============================================================
        // Step 2: Compute σ(1.702x) = 1 / (1 + exp(-1.702x))
        // ============================================================

        uint32_t sigmoid_25bit;
        sigmoid(scaled_x_25bit, sigmoid_25bit);

        // ============================================================
        // Step 3: Compute x * σ(1.702x)
        // ============================================================

        mult_module.reset();
        mult_module.set_input(x_25bit[i], sigmoid_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t result_25bit = mult_module.get_result();

        // Output conversion: 25-bit → bf16
        y_bf16[i] = extended_25bit_to_bf16(result_25bit);
    }
}
