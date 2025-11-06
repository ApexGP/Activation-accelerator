#include "Arith/FPMult.h"

FPMult::FPMult()
{
    reset();
}

void FPMult::reset()
{
    X_input = 0;
    Y_input = 0;

    // Stage 0 signals
    sign = 0;
    expX = 0;
    expY = 0;
    expSumPreSub = 0;
    bias = 127;  // bf16 bias
    expSum = 0;
    sigX = 0;
    sigY = 0;
    sigProd = 0;
    excSel = 0;
    exc = 0;
    norm = false;
    expPostNorm = 0;
    sigProdExt = 0;
    expSig = 0;
    sticky = false;

    // Stage 1 delayed signals
    sign_d1 = 0;
    exc_d1 = 0;
    sigProdExt_d1 = 0;
    sticky_d1 = false;

    // Output
    R = 0;

    // Pipeline cycle
    cycle = 0;
}

void FPMult::set_input(uint32_t X, uint32_t Y)
{
    X_input = X & 0x1FFFFFF;  // 25-bit
    Y_input = Y & 0x1FFFFFF;  // 25-bit
    cycle = 0;
}

void FPMult::clock_step()
{
    if (cycle == 0) {
        // ========== Cycle 0: Compute stage 0 ==========

        // Special case: if either input is exactly 0 (all bits zero), result is 0
        if (X_input == 0 || Y_input == 0) {
            // Set result to zero (both cycles completed immediately)
            R = 0;      // All bits zero represents zero (exn=0, zero)
            cycle = 1;  // Mark as ready for next cycle

            // Initialize stage 1 registers to safe values for zero result
            sigProdExt_d1 = 0;
            sticky_d1 = false;
            exc_d1 = 0b01;  // Internal zero code
            sign_d1 = 0;
            expSig = 0;

            return;
        }

        // Extract fields from inputs
        // Format: [24:23]=exn, [22]=sign, [21:14]=exp, [13:0]=mantissa
        sign = (X_input >> 22) & 0x1;
        sign = sign ^ ((Y_input >> 22) & 0x1);  // XOR for multiplication

        expX = (X_input >> 14) & 0xFF;
        expY = (Y_input >> 14) & 0xFF;

        // Exponent addition
        expSumPreSub = ((uint16_t) expX) + ((uint16_t) expY);

        // Check for exponent overflow/underflow BEFORE subtraction
        // Valid range after bias subtraction: [1, 254] (expSum = expX + expY - 127)
        // Overflow: expX + expY >= 127 + 255 = 382
        // Underflow: expX + expY <= 127 (after subtracting 127, result <= 0)
        if (expSumPreSub >= 382) {
            // Exponent overflow → result will be infinity
            exc = 0b10;    // Set to infinity early
            expSum = 255;  // Clamp to max
        } else if (expSumPreSub <= 127) {
            // Exponent underflow → result will be zero or subnormal
            exc = 0b01;  // Set to zero early
            expSum = 0;
        } else {
            expSum = expSumPreSub - bias;  // Subtract bias (127)
        }

        // Significand multiplication (add implicit leading 1)
        sigX = (1 << 14) | (X_input & 0x3FFF);  // {1'b1, mantissa[13:0]} = 15 bits
        sigY = (1 << 14) | (Y_input & 0x3FFF);  // {1'b1, mantissa[13:0]} = 15 bits

        // 15-bit × 15-bit = 30-bit multiplication
        sigProd = multiply_15x15(sigX, sigY);

        // Exception handling using switch-case
        // exn encoding: 00=zero, 01=normal, 10=infinity, 11=NaN
        uint8_t exn_X = (X_input >> 23) & 0x3;
        uint8_t exn_Y = (Y_input >> 23) & 0x3;
        excSel = (exn_X << 2) | exn_Y;

        // Multiplication exception table (4-bit selector = 16 cases)
        // Internal exc: 00=normal, 01=zero, 10=inf, 11=NaN
        switch (excSel) {
            // Zero cases (0x0_): 0 * {0, normal, inf, NaN}
            case 0b0000:  // 0 * 0 = 0
            case 0b0001:  // 0 * normal = 0
                exc = 0b01;
                break;
            case 0b0010:  // 0 * inf = NaN
            case 0b0011:  // 0 * NaN = NaN
            case 0b1000:  // inf * 0 = NaN
            case 0b1100:  // NaN * 0 = NaN
                exc = 0b11;
                break;

            // Normal cases (0x1_): normal * {0, normal, inf, NaN}
            case 0b0100:  // normal * 0 = 0
                exc = 0b01;
                break;
            case 0b0101:  // normal * normal = normal
                exc = 0b00;
                break;
            case 0b0110:  // normal * inf = inf
            case 0b1001:  // inf * normal = inf
            case 0b1010:  // inf * inf = inf
                exc = 0b10;
                break;
            case 0b0111:  // normal * NaN = NaN
            case 0b1011:  // inf * NaN = NaN
            case 0b1101:  // NaN * normal = NaN
            case 0b1110:  // NaN * inf = NaN
            case 0b1111:  // NaN * NaN = NaN
                exc = 0b11;
                break;

            default:
                exc = 0b11;  // Safety: treat unknown as NaN
                break;
        }

        // Normalization check and shift (use ternary operator)
        norm = (sigProd >> 29) & 0x1;                           // Check bit 29
        expPostNorm = expSum + norm;                            // Add 1 if normalized, 0 otherwise
        sigProdExt = (sigProd << (norm ? 1 : 2)) & 0x3FFFFFFF;  // Shift by 1 or 2 bits (30-bit)

        // Combine exponent and significand (upper 14 bits of mantissa)
        expSig = (expPostNorm << 14) | ((sigProdExt >> 16) & 0x3FFF);  // 24-bit

        // Sticky bit (bit 15 of sigProdExt)
        sticky = (sigProdExt >> 15) & 0x1;

        // Register update (pipeline stage 0 → 1)
        sign_d1 = sign;
        exc_d1 = exc;
        sigProdExt_d1 = sigProdExt;
        sticky_d1 = sticky;

        cycle = 1;

    } else if (cycle == 1) {
        // ========== Cycle 1: Rounding and final output ==========

        // Guard bit: OR of bits [14:0] using bit mask
        bool guard = (sigProdExt_d1 & 0x7FFF) != 0;  // bits [14:0]

        // Round bit
        // round = sticky & ((guard & ~sigProdExt[16]) | sigProdExt[16])
        bool round = sticky_d1 &&
                     ((guard && !((sigProdExt_d1 >> 16) & 0x1)) || ((sigProdExt_d1 >> 16) & 0x1));

        // Rounding adder (24-bit)
        uint32_t expSigPostRound = round_add_24bit(expSig, round);

        // Exception update after normalization
        uint8_t excPostNorm;
        uint8_t exp_high_bits = (expSigPostRound >> 22) & 0x3;  // bits [23:22]

        switch (exp_high_bits) {
            case 0b00:
                excPostNorm = 0b01;  // Underflow
                break;
            case 0b01:
                excPostNorm = 0b10;  // Overflow
                break;
            case 0b10:
            case 0b11:
                excPostNorm = 0b00;  // Normal
                break;
            default:
                excPostNorm = 0b11;
                break;
        }

        // Final exception selection (use ternary operator)
        // exc=01 (zero) can be overridden by excPostNorm, others remain unchanged
        uint8_t finalExc = (exc_d1 != 0b01) ? exc_d1 : excPostNorm;

        // Assemble output using switch-case
        // Convert internal exc to external exn
        // Internal: exc=00(normal), exc=01(zero), exc=10(inf), exc=11(NaN)
        // External: exn=0(zero), exn=1(normal), exn=2(inf), exn=3(NaN)
        // 25-bit format: [24:23]=exn, [22]=sign, [21:0]={exp[7:0], mantissa[13:0]}
        switch (finalExc) {
            case 0b01:  // Zero
                R = 0;  // exn=0, sign=0, exp=0, mantissa=0
                break;
            case 0b00:  // Normal
                R = (1 << 23) | (sign_d1 << 22) | (expSigPostRound & 0x3FFFFF);
                break;
            case 0b10:  // Infinity
                R = (2 << 23) | (sign_d1 << 22) | (0xFF << 14);
                break;
            case 0b11:  // NaN
                R = (3 << 23) | (sign_d1 << 22) | (0xFF << 14) | 1;
                break;
            default:
                R = 0;  // Safety
                break;
        }

        cycle = 0;  // Reset for next computation
    }
}

uint32_t FPMult::multiply_15x15(uint16_t a, uint16_t b)
{
    // 15-bit × 15-bit = 30-bit unsigned multiplication
    uint32_t result = (uint32_t) a * (uint32_t) b;
    return result & 0x3FFFFFFF;  // Mask to 30 bits
}

uint32_t FPMult::round_add_24bit(uint32_t X, bool round)
{
    // 24-bit adder: X + round
    uint32_t result = (X & 0xFFFFFF) + (round ? 1 : 0);
    return result & 0xFFFFFF;  // Mask to 24 bits
}
