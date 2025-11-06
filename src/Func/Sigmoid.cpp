#include "Arith/Exp.h"
#include "Arith/FPAdd.h"
#include "Func/Sigmoid.h"
#include "Utils/FormatUtils.h"

void sigmoid(uint32_t x_25bit, uint32_t &y_25bit)
{
    // ============================================================
    // Process each element (full 25-bit precision)
    // ============================================================
    Exp exp_module;
    FPAdd fpadd_module;

    // Precompute 1.0 in 25-bit format
    uint32_t one_25bit = float_to_extended_25bit(1.0f);

    // Check special values
    ValueType x_type = classify_25bit(x_25bit);

    switch (x_type) {
        // σ(NaN) = NaN
        case ValueType::NaN: {
            uint16_t nan_bf16 = 0x7FC0;
            y_25bit = float_to_extended_25bit(bf16_to_float(nan_bf16));
            return;
        }

        // σ(+∞) = 1 / (1 + exp(-∞)) = 1 / (1 + 0) = 1
        case ValueType::PosInf:
            y_25bit = float_to_extended_25bit(1.0f);
            return;

        // σ(-∞) = 1 / (1 + exp(+∞)) = 1 / (+∞) = 0
        case ValueType::NegInf:
            y_25bit = float_to_extended_25bit(0.0f);
            return;

        // Normal case: Zero and Normal values continue to computation
        case ValueType::Zero:
        case ValueType::Normal:
            break;

        default:
            break;
    }

    // Numerical stability check (avoid exp overflow/underflow)
    float x_float = extended_25bit_to_float(x_25bit);

    // For x > 88, σ(x) ≈ 1 (exp(-x) underflows to 0)
    if (x_float > 88.0f) {
        y_25bit = float_to_extended_25bit(1.0f);
        return;
    }

    // For x < -88, σ(x) ≈ 0 (exp(-x) overflows)
    if (x_float < -88.0f) {
        y_25bit = float_to_extended_25bit(0.0f);
        return;
    }

    // Normal case: compute -x (flip sign bit)
    uint32_t neg_x_25bit = x_25bit ^ (1U << 22);  // Flip sign bit at position 22

    // Compute exp(-x)
    exp_module.reset();
    exp_module.set_input(neg_x_25bit, true);

    for (int j = 0; j < 8; j++) {
        exp_module.clock_step();
    }

    uint32_t exp_neg_x_25bit = exp_module.get_Exp_Ans_extended();

    // Check if exp result overflows
    ValueType exp_type = classify_25bit(exp_neg_x_25bit);
    if (exp_type == ValueType::PosInf) {
        // exp(-x) = +∞, meaning x < 0 and |x| is large, σ(x) ≈ 0
        y_25bit = float_to_extended_25bit(0.0f);
        return;
    }

    // Compute 1 + exp(-x) (full 25-bit precision)
    fpadd_module.reset();
    fpadd_module.set_input(one_25bit, exp_neg_x_25bit);

    for (int cycle = 0; cycle < 2; cycle++) {
        fpadd_module.clock_step();
    }

    uint32_t sum_25bit = fpadd_module.get_result_extended();

    // Output conversion: compute 1 / (1 + exp(-x)), then 25-bit → float → bf16
    float sum_float = extended_25bit_to_float(sum_25bit);
    float result_float = 1.0f / sum_float;
    y_25bit = float_to_extended_25bit(result_float);
}
