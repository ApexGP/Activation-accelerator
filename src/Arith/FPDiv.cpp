#include "Arith/FPDiv.h"

FPDiv::FPDiv()
{
    reset();
}

void FPDiv::reset()
{
    X1_reg = 0;
    X2_reg = 0;
    valid_in_reg = false;

    analysis1.reset();
    analysis2.reset();
    fpsub_module.reset();
    exp_module.reset();

    ln_X1 = 0;
    ln_X2 = 0;
    ln_diff = 0;
    exp_result = 0;

    state = IDLE;
    cycle_count = 0;

    result_sign = false;
    result = 0;
    valid_out = false;
}

void FPDiv::set_input(uint32_t X1, uint32_t X2)
{
    X1_reg = X1 & 0x1FFFFFF;  // 25-bit
    X2_reg = X2 & 0x1FFFFFF;  // 25-bit
    valid_in_reg = true;

    // Check for special cases (exn field)
    uint8_t exn1 = (X1_reg >> 23) & 0x3;
    uint8_t exn2 = (X2_reg >> 23) & 0x3;

    // exn encoding: 00=zero, 01=normal, 10=infinity, 11=NaN
    // Handle special cases according to IEEE 754:
    // - NaN / anything = NaN
    // - anything / NaN = NaN
    // - Inf / Inf = NaN
    // - Inf / normal = Inf (with sign)
    // - normal / Inf = 0 (with sign)
    // - 0 / 0 = NaN
    // - 0 / normal = 0 (with sign)
    // - normal / 0 = Inf (with sign)

    bool sign1 = (X1_reg >> 22) & 0x1;
    bool sign2 = (X2_reg >> 22) & 0x1;
    result_sign = sign1 ^ sign2;  // Division sign: XOR (saved as member variable)

    // All conditions that result in NaN
    bool is_NaN = (exn1 == 3 || exn2 == 3) ||  // NaN / anything or anything / NaN
                  (exn1 == 2 && exn2 == 2) ||  // Inf / Inf
                  (exn1 == 0 && exn2 == 0);    // 0 / 0

    // All conditions that result in Inf (that are not already NaN)
    bool is_Inf = (exn1 == 2) ||  // Inf / normal or Inf / 0
                  (exn2 == 0);    // normal / 0

    // All conditions that result in Zero (that are not already NaN or Inf)
    bool is_Zero = (exn2 == 2) ||  // normal / Inf or 0 / Inf
                   (exn1 == 0);    // 0 / normal

    // Handle special cases
    if (is_NaN) {
        result = (0x03 << 23) | (0xFF << 14) | 0x1;  // NaN
        valid_out = true;
        state = DONE;
        return;

    } else if (is_Inf) {
        result = (0x02 << 23) | (result_sign << 22) | (0xFF << 14) | 0x0;  // Inf
        valid_out = true;
        state = DONE;
        return;

    } else if (is_Zero) {
        result = (0x00 << 23) | (result_sign << 22);  // Zero
        valid_out = true;
        state = DONE;
        return;
    }

    // Special optimization: X1 == X2 (ignoring sign)
    // Result = ±1.0 depending on signs
    uint32_t X1_abs = X1_reg & 0x3FFFFF;  // Remove exn and sign
    uint32_t X2_abs = X2_reg & 0x3FFFFF;

    if (X1_abs == X2_abs) {
        // X1 and X2 have same absolute value
        // Result = +1.0 if signs are same, -1.0 if different
        // 1.0 in 25-bit: exn=01, sign=result_sign, exp=127, mantissa=0
        result = (0x01 << 23) | (result_sign << 22) | (127 << 14) | 0x0000;
        valid_out = true;
        state = DONE;
        return;
    }

    // Normal case: both are normal numbers
    // Reset all modules for new computation
    analysis1.reset();
    analysis2.reset();
    fpsub_module.reset();
    exp_module.reset();

    state = ANALYSIS;
    cycle_count = 0;
    valid_out = false;
}

void FPDiv::stage_analysis()
{
    // Parallel computation of ln(|X1|) and ln(|X2|)
    // Extract 23-bit input (remove exn bits) for Analysis module
    // Clear sign bit (bit 22) since ln() only works with positive numbers
    uint32_t X1_23bit = X1_reg & 0x3FFFFF;  // Clear sign bit, keep exp(8) + mantissa(14)
    uint32_t X2_23bit = X2_reg & 0x3FFFFF;  // Clear sign bit

    // Call set_input only in the first cycle, then only call clock_step
    if (cycle_count == 0) {
        // Set inputs for Analysis modules
        analysis1.set_input(X1_23bit, true);
        analysis2.set_input(X2_23bit, true);
    }

    // Clock both Analysis modules
    analysis1.clock_step();
    analysis2.clock_step();

    cycle_count++;

    // Analysis outputs valid_out on cycle 17 (PIPELINE_DEPTH = 17)
    if (analysis1.get_valid_out() && analysis2.get_valid_out()) {
        // Get ln(X1) and ln(X2) results (25-bit extended format)
        ln_X1 = analysis1.get_Ln_Ans_extended();
        ln_X2 = analysis2.get_Ln_Ans_extended();

        state = FPSUB_COMPUTE;
        cycle_count = 0;
    }
}

void FPDiv::stage_fpsub()
{
    // Compute ln(X1) - ln(X2) using FPSub
    if (cycle_count == 0) {
        // Set inputs for FPSub module (both are 25-bit)
        fpsub_module.set_input(ln_X1, ln_X2);
    }

    // Clock FPSub module
    fpsub_module.clock_step();

    cycle_count++;

    // FPSub takes 2 cycles (FPSub::clock_step() executes 3 stages in one call)
    // Cycle 0: set_input + clock_step
    // Cycle 1: clock_step (result ready)
    if (cycle_count >= 2) {
        // Get result (25-bit format)
        ln_diff = fpsub_module.get_result_extended();

        state = EXP_COMPUTE;
        cycle_count = 0;
    }
}

void FPDiv::stage_exp()
{
    // Compute exp(ln(X1) - ln(X2))
    if (cycle_count == 0) {
        // Set input for Exp module (25-bit)
        exp_module.set_input(ln_diff, true);
    }

    // Clock Exp module
    exp_module.clock_step();

    cycle_count++;

    // Exp takes 8 cycles (stage0_input + stage1_K_calculation + stage2_Y_calculation + 5 tree stages)
    if (cycle_count >= 8 && exp_module.get_valid_out()) {
        // Get exp(ln_diff) result (25-bit extended format)
        // This gives us |X1/X2|, we need to apply the sign
        exp_result = exp_module.get_Exp_Ans_extended();

        // Apply division sign to the result
        // Extract components from exp_result
        uint32_t exn = (exp_result >> 23) & 0x3;
        uint32_t exp_val = (exp_result >> 14) & 0xFF;
        uint32_t mantissa = exp_result & 0x3FFF;

        // Apply the result_sign (from X1/X2 division)
        // Reconstruct result with correct sign
        result = (exn << 23) | (result_sign << 22) | (exp_val << 14) | mantissa;

        valid_out = true;
        state = DONE;
    }
}

void FPDiv::clock_step()
{
    switch (state) {
        case IDLE:
            valid_out = false;
            break;

        case ANALYSIS:
            stage_analysis();
            break;

        case FPSUB_COMPUTE:
            stage_fpsub();
            break;

        case EXP_COMPUTE:
            stage_exp();
            break;

        case DONE:
            // Stay in DONE state, valid_out remains true
            break;
    }
}
