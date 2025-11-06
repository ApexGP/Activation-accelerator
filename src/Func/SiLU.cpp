#include "Arith/FPMult.h"
#include "Func/SiLU.h"
#include "Func/Sigmoid.h"
#include "Utils/FormatUtils.h"

void silu(uint16_t *x_bf16, uint16_t *y_bf16, const int len)
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
    // Process each element (full 25-bit precision)
    // ============================================================
    FPMult mult_module;

    for (int i = 0; i < len; i++) {
        // Check special values
        ValueType x_type = classify_25bit(x_25bit[i]);

        switch (x_type) {
            // SiLU(NaN) = NaN
            case ValueType::NaN:
                y_bf16[i] = 0x7FC0;
                continue;

            // SiLU(+∞) = +∞ * σ(+∞) = +∞ * 1 = +∞
            case ValueType::PosInf:
                y_bf16[i] = 0x7F80;
                continue;

            // SiLU(-∞) = -∞ * σ(-∞) = -∞ * 0 → 0 (by L'Hôpital's rule)
            case ValueType::NegInf:
                y_bf16[i] = 0x0000;
                continue;

            // Normal case: Zero and Normal values continue to computation
            case ValueType::Zero:
            case ValueType::Normal:
                break;

            default:
                break;
        }

        // Numerical stability check (avoid exp overflow/underflow)
        float x_float = extended_25bit_to_float(x_25bit[i]);

        // For x > 88, SiLU(x) ≈ x (σ(x) ≈ 1)
        if (x_float > 88.0f) {
            y_bf16[i] = x_bf16[i];
            continue;
        }

        // For x < -88, SiLU(x) ≈ 0 (σ(x) ≈ 0)
        if (x_float < -88.0f) {
            y_bf16[i] = 0x0000;
            continue;
        }

        // Use sigmoid activation function
        uint32_t sigmoid_25bit;
        sigmoid(x_25bit[i], sigmoid_25bit);

        // Compute x * σ(x) (using FPMult hardware module, full 25-bit precision)
        mult_module.set_input(x_25bit[i], sigmoid_25bit);
        mult_module.clock_step();  // Cycle 0
        mult_module.clock_step();  // Cycle 1

        uint32_t result_25bit = mult_module.get_result();

        // Output conversion: 25-bit → bf16
        y_bf16[i] = extended_25bit_to_bf16(result_25bit);
    }
}
