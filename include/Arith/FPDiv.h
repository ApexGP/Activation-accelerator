#ifndef FPDIV_H
#define FPDIV_H

#include <cstdint>

#include "Arith/Analysis.h"
#include "Arith/Exp.h"
#include "Arith/FPSub.h"

/**
 * FPDiv: Hardware Floating-Point Divider using Log-Exp Algorithm
 * 
 * Algorithm:
 *   X1/X2 = exp(ln(X1) - ln(X2))
 * 
 * Pipeline stages:
 *   Stage 0-16: Analysis (ln) for X1 and X2 in parallel (17 cycles each)
 *   Stage 17-18: FPSub for ln(X1) - ln(X2) (2 cycles)
 *   Stage 19-26: Exp for exp(ln(X1) - ln(X2)) (8 cycles)
 * 
 * Total latency: ~27 cycles
 * 
 * Data format (uniform throughout):
 *   Input:  25-bit = exn(2) + sign(1) + exp(8) + mantissa(14)
 *   Output: 25-bit = exn(2) + sign(1) + exp(8) + mantissa(14)
 */
class FPDiv
{
public:
    FPDiv();
    void reset();
    void clock_step();
    void set_input(uint32_t X1, uint32_t X2);

    // Output getters
    bool get_valid_out() const
    {
        return valid_out;
    }

    // Output: 25-bit format (exn(2) + sign(1) + exp(8) + mantissa(14))
    uint32_t get_result_extended() const
    {
        return result;
    }

    // Debug/testing getters for intermediate results
    uint32_t get_ln_X1() const
    {
        return ln_X1;
    }
    uint32_t get_ln_X2() const
    {
        return ln_X2;
    }
    uint32_t get_ln_diff() const
    {
        return ln_diff;
    }

    enum State { IDLE, ANALYSIS, FPSUB_COMPUTE, EXP_COMPUTE, DONE };
    State get_state() const
    {
        return state;
    }

private:
    // Input registers
    uint32_t X1_reg;  // 25-bit
    uint32_t X2_reg;  // 25-bit
    bool valid_in_reg;

    // Analysis modules (compute ln)
    Analysis analysis1;
    Analysis analysis2;

    // FPSub module (compute ln(X1) - ln(X2))
    FPSub fpsub_module;

    // Exp module (compute exp(diff))
    Exp exp_module;

    // Pipeline results
    uint32_t ln_X1;       // 25-bit from Analysis
    uint32_t ln_X2;       // 25-bit from Analysis
    uint32_t ln_diff;     // 25-bit from FPSub
    uint32_t exp_result;  // 25-bit from Exp

    // State machine
    State state;
    int cycle_count;

    // Division sign (X1/X2 sign = sign1 XOR sign2)
    bool result_sign;

    // Output
    uint32_t result;  // 25-bit
    bool valid_out;

    // Helper functions
    void stage_analysis();
    void stage_fpsub();
    void stage_exp();
};

#endif  // FPDIV_H
