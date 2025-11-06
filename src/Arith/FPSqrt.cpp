#include "Arith/FPSqrt.h"
#include "Utils/FormatUtils.h"

// Constants for inverse square root computation
// -0.5 in 25-bit format: exn=01, sign=1, exp=126, mantissa=0
// -0.5 = -(1.0 * 2^(-1)), so exp = 127 - 1 = 126, sign = 1
const uint32_t FPSqrt::CONST_NEG_HALF = (1U << 23) | (1U << 22) | (126U << 14) | 0U;

// 3.0 in 25-bit format: exn=01, sign=0, exp=128, mantissa=0
// 3.0 = 1.5 * 2^1, normalized: 1.5 has mantissa 0.5, so mantissa_14bit = 0x2000
// Actually 3.0 = 1.1b * 2^1 in binary, so exp = 127 + 1 = 128
const uint32_t FPSqrt::CONST_THREE = (1U << 23) | (0U << 22) | (128U << 14) | (1U << 13);

FPSqrt::FPSqrt()
{
    reset();
}

void FPSqrt::reset()
{
    X_reg = 0;
    valid_in_reg = false;

    analysis.reset();
    mult_module.reset();
    exp_module.reset();

    ln_X = 0;
    half_ln_X = 0;
    exp_result = 0;

    state = IDLE;
    cycle_count = 0;

    result = 0;
    valid_out = false;

    // Newton-Raphson variables
    A_float = 0.0f;
    x_current = 0.0f;
}

void FPSqrt::set_input(uint32_t X)
{
    X_reg = X & 0x1FFFFFF;  // 25-bit
    valid_in_reg = true;

    // Save input value as float for Newton-Raphson iterations
    A_float = extended_25bit_to_float(X_reg);

    // Check for special cases (exn field)
    uint8_t exn = (X_reg >> 23) & 0x3;
    bool sign = (X_reg >> 22) & 0x1;

    // exn encoding: 00=zero, 01=normal, 10=infinity, 11=NaN
    // Handle special cases for 1/sqrt(X):
    // - 1/sqrt(NaN) = NaN
    // - 1/sqrt(+inf) = +0
    // - 1/sqrt(-inf) = NaN
    // - 1/sqrt(-x) = NaN (for any negative non-zero x)
    // - 1/sqrt(+0) = +inf
    // - 1/sqrt(-0) = -inf

    if (exn == 3) {
        // NaN
        result = X_reg;  // Propagate NaN
        valid_out = true;
        state = DONE;
        return;
    } else if (exn == 2) {
        // Infinity
        if (sign) {
            // 1/sqrt(-inf) = NaN
            result = (3U << 23) | (0xFF << 14) | 0x1;  // NaN
        } else {
            // 1/sqrt(+inf) = +0
            result = (0U << 23) | (0U << 22);  // +0
        }
        valid_out = true;
        state = DONE;
        return;
    } else if (exn == 0) {
        // Zero: 1/sqrt(±0) = ±inf (preserve sign)
        if (sign) {
            // 1/sqrt(-0) = -inf
            result = (2U << 23) | (1U << 22) | (0xFF << 14);  // -inf
        } else {
            // 1/sqrt(+0) = +inf
            result = (2U << 23) | (0U << 22) | (0xFF << 14);  // +inf
        }
        valid_out = true;
        state = DONE;
        return;
    } else if (sign) {
        // Normal negative number: 1/sqrt(-x) = NaN
        result = (3U << 23) | (0xFF << 14) | 0x1;  // NaN
        valid_out = true;
        state = DONE;
        return;
    }

    // Normal positive number: proceed with computation
    // Reset all modules for new computation
    analysis.reset();
    mult_module.reset();
    exp_module.reset();

    state = ANALYSIS;
    cycle_count = 0;
    valid_out = false;
}

void FPSqrt::stage_analysis()
{
    // Compute ln(X) using Analysis module
    // Extract 23-bit input (remove exn bits) for Analysis module
    uint32_t X_23bit = X_reg & 0x7FFFFF;  // Remove exn bits, keep sign + exp + mantissa

    // 学习 Softmax 的模式：只在第一个周期调用 set_input，后续只调用 clock_step
    if (cycle_count == 0) {
        // Set input for Analysis module
        analysis.set_input(X_23bit, true);
    }

    // Clock Analysis module
    analysis.clock_step();

    cycle_count++;

    // Analysis outputs valid_out on cycle 17 (PIPELINE_DEPTH = 17)
    if (analysis.get_valid_out()) {
        // Get ln(X) result (25-bit extended format)
        ln_X = analysis.get_Ln_Ans_extended();

        state = MULT_HALF;
        cycle_count = 0;
    }
}

void FPSqrt::stage_mult()
{
    // Compute -0.5 * ln(X) using FPMult for inverse square root
    if (cycle_count == 0) {
        // Set inputs for FPMult module (both are 25-bit)
        mult_module.set_input(CONST_NEG_HALF, ln_X);
    }

    // Clock FPMult module
    mult_module.clock_step();

    cycle_count++;

    // FPMult takes 2 cycles
    if (cycle_count >= 2) {
        // Get result (25-bit format): -0.5 * ln(X)
        half_ln_X = mult_module.get_result();

        state = EXP_COMPUTE;
        cycle_count = 0;
    }
}

void FPSqrt::stage_exp()
{
    // Compute exp(-0.5 * ln(X)) = exp(ln(X^(-0.5))) = X^(-0.5) = 1/sqrt(X)
    // 学习 Softmax 的模式：只在第一个周期调用 set_input，后续只调用 clock_step
    if (cycle_count == 0) {
        // Set input for Exp module (25-bit)
        exp_module.set_input(half_ln_X, true);
    }

    // Clock Exp module
    exp_module.clock_step();

    cycle_count++;

    // Exp takes 8 cycles (stage0_input + stage1_K_calculation + stage2_Y_calculation + 5 tree stages)
    if (cycle_count >= 8 && exp_module.get_valid_out()) {
        // Get exp(-0.5 * ln(X)) = 1/sqrt(X) result (25-bit extended format)
        // This is our initial estimate x0 for Newton-Raphson
        exp_result = exp_module.get_Exp_Ans_extended();

        // Convert to float for Newton-Raphson iterations
        x_current = extended_25bit_to_float(exp_result);

        // Proceed to Newton-Raphson refinement
        state = NEWTON_ITER1;
        cycle_count = 0;
    }
}

void FPSqrt::stage_newton_iter1()
{
    // Newton-Raphson iteration 1 for 1/sqrt(X):
    // x_{n+1} = 0.5 * x_n * (3 - X * x_n^2)
    // This formula converges to 1/sqrt(X) without division
    if (x_current != 0.0f) {
        float x_squared = x_current * x_current;        // x_n^2
        float X_times_x_squared = A_float * x_squared;  // X * x_n^2
        float three_minus = 3.0f - X_times_x_squared;   // 3 - X * x_n^2
        x_current = 0.5f * x_current * three_minus;     // 0.5 * x_n * (3 - X * x_n^2)
    }

    state = NEWTON_ITER2;
}

void FPSqrt::stage_newton_iter2()
{
    // Newton-Raphson iteration 2 for 1/sqrt(X):
    // x_{n+1} = 0.5 * x_n * (3 - X * x_n^2)
    // Final iteration before converting back to 25-bit format
    if (x_current != 0.0f) {
        float x_squared = x_current * x_current;        // x_n^2
        float X_times_x_squared = A_float * x_squared;  // X * x_n^2
        float three_minus = 3.0f - X_times_x_squared;   // 3 - X * x_n^2
        x_current = 0.5f * x_current * three_minus;     // 0.5 * x_n * (3 - X * x_n^2)
    }

    // Convert final result back to 25-bit format
    // Note: This is the only 25-bit conversion in the entire iteration process
    result = float_to_extended_25bit(x_current);
    valid_out = true;
    state = DONE;
}

void FPSqrt::clock_step()
{
    switch (state) {
        case IDLE:
            valid_out = false;
            break;

        case ANALYSIS:
            stage_analysis();
            break;

        case MULT_HALF:
            stage_mult();
            break;

        case EXP_COMPUTE:
            stage_exp();
            break;

        case NEWTON_ITER1:
            stage_newton_iter1();
            break;

        case NEWTON_ITER2:
            stage_newton_iter2();
            break;

        case DONE:
            // Stay in DONE state, valid_out remains true
            break;
    }
}
