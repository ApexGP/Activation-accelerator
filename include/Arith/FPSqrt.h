#ifndef FPSQRT_H
#define FPSQRT_H

#include <cstdint>

#include "Arith/Analysis.h"
#include "Arith/Exp.h"
#include "Arith/FPMult.h"

/**
 * FPSqrt: Inverse Square Root (1/sqrt(X)) using Hybrid Method
 * 
 * Purpose: Compute reciprocal square root for normalization operations
 *          (RMS normalization, Layer normalization, etc.)
 * 
 * Algorithm (Hybrid Method):
 *   1. Initial estimate: x0 = exp(-0.5 * ln(X))  [log-exp method, 27 cycles]
 *   2. Newton-Raphson iteration 1: x1 = 0.5 * x0 * (3 - X * x0^2)  [software float]
 *   3. Newton-Raphson iteration 2: x2 = 0.5 * x1 * (3 - X * x1^2)  [software float]
 * 
 * Newton-Raphson formula for 1/sqrt(X):
 *   x_{n+1} = 0.5 * x_n * (3 - X * x_n^2)
 *   Converges to 1/sqrt(X) when x_0 is close to 1/sqrt(X)
 * 
 * Pipeline stages:
 *   Stage 0-16: Analysis: ln(X) (17 cycles)
 *   Stage 17-18: FPMult: -0.5 * ln(X) (2 cycles)
 *   Stage 19-26: Exp: exp(-0.5 * ln(X)) = 1/sqrt(X) (8 cycles)
 *   Stage 27: Newton-Raphson iteration 1 (1 cycle, software)
 *   Stage 28: Newton-Raphson iteration 2 + 25-bit conversion (1 cycle, software)
 * 
 * Total latency: ~29 cycles
 * 
 * Precision: Reaches 25-bit format's theoretical limit
 * - Relative error: < 0.01% (all values)
 * - Suitable for normalization operations
 * 
 * Advantages over sqrt(X) then divide:
 * - Direct computation of 1/sqrt(X) avoids FPDiv (saves ~28 cycles)
 * - Single conversion to 25-bit at the end
 * - Better numerical stability
 * 
 * Data format:
 *   Input:  25-bit = exn(2) + sign(1) + exp(8) + mantissa(14)
 *   Output: 25-bit = exn(2) + sign(1) + exp(8) + mantissa(14)
 * 
 * Special cases:
 *   1/sqrt(+0) = +inf
 *   1/sqrt(-0) = -inf
 *   1/sqrt(+inf) = +0
 *   1/sqrt(negative) = NaN
 *   1/sqrt(NaN) = NaN
 */
class FPSqrt
{
public:
    FPSqrt();
    void reset();
    void clock_step();
    void set_input(uint32_t X);

    // Output getters
    bool get_valid_out() const
    {
        return valid_out;
    }

    // Output: 25-bit format (exn(2) + sign(1) + exp(8) + mantissa(14))
    uint32_t get_result_extended() const
    {
#pragma HLS INLINE
        return result;
    }

    // Debug getters
    uint32_t get_ln_X() const
    {
#pragma HLS INLINE
        return ln_X;
    }
    uint32_t get_half_ln_X() const
    {
#pragma HLS INLINE
        return half_ln_X;
    }

    enum State { IDLE, ANALYSIS, MULT_HALF, EXP_COMPUTE, NEWTON_ITER1, NEWTON_ITER2, DONE };
    State get_state() const
    {
#pragma HLS INLINE
        return state;
    }

private:
    // Input register
    uint32_t X_reg;  // 25-bit
    bool valid_in_reg;

    // Analysis module (compute ln)
    Analysis analysis;

    // FPMult module (compute ln(X) * 0.5)
    FPMult mult_module;

    // Exp module (compute exp(0.5 * ln(X)))
    Exp exp_module;

    // Pipeline results
    uint32_t ln_X;        // 25-bit from Analysis
    uint32_t half_ln_X;   // 25-bit from FPMult
    uint32_t exp_result;  // 25-bit from Exp

    // Constants in 25-bit format
    // -0.5 = -(1.0 * 2^(-1)), so exp = 126, sign = 1
    static const uint32_t CONST_NEG_HALF;  // -0.5 for ln(X) * (-0.5)
    static const uint32_t CONST_THREE;     // 3.0 for Newton-Raphson (3 - X*x^2)

    // State machine
    State state;
    int cycle_count;

    // Output
    uint32_t result;  // 25-bit
    bool valid_out;

    // Newton-Raphson refinement variables
    float A_float;    // Input value in float format
    float x_current;  // Current iteration value

    // Helper functions
    void stage_analysis();
    void stage_mult();
    void stage_exp();
    void stage_newton_iter1();
    void stage_newton_iter2();
};

#endif  // FPSQRT_H
