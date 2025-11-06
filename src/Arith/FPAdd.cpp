#include <cstdint>

#include "Arith/FPAdd.h"

FPAdd::FPAdd()
{
    reset();
}

void FPAdd::reset()
{
    X_reg = 0;
    Y_reg = 0;
    R = 0;

    newX_d1 = 0;
    EffSub_d1 = false;
    selectClosePath_d1 = false;
    exponentResultfar0_d1 = 0;
    syncEffSub_d1 = false;
    syncX_d1 = 0;
    syncSignY_d1 = false;
    syncResSign_d1 = false;
    syncExnXY_d1 = 0;

    fracRClosexMy_reg = 0;
    fracRCloseyMx_reg = 0;

    count3_d1 = 0;
    level3_d1 = 0;
    count2_d1 = 0;
    nZerosNew_reg = 0;
    shiftedFrac_reg = 0;

    ps_d1 = 0;
    stk2_d1 = false;
    level2_d1 = 0;
    level1_d1 = 0;
    shiftedFracY_reg = 0;
    sticky_reg = false;

    X_adder18_d1 = 0;
    Y_adder18_d1 = 0;
    fracResultfar0_reg = 0;

    Y_adder24_d1 = 0;
    resultRounded_reg = 0;
}

void FPAdd::set_input(uint32_t X, uint32_t Y)
{
    X_reg = X & 0x1FFFFFF;  // 25-bit
    Y_reg = Y & 0x1FFFFFF;  // 25-bit
}

// Dual subtractor: compute X-Y and Y-X simultaneously (17-bit)
void FPAdd::IntDualSub_17(uint32_t X, uint32_t Y, uint32_t &XmY, uint32_t &YmX)
{
    XmY = (X + (~Y) + 1) & 0x1FFFF;  // 17-bit
    YmX = (Y + (~X) + 1) & 0x1FFFF;  // 17-bit
}

// Normalizer: leading zero count and left shift (16-bit)
void FPAdd::Normalizer_Z_16(uint16_t X, uint8_t &Count, uint16_t &R)
{
    uint16_t level5 = X;

    // count4: check if all 16 bits are zero
    uint8_t count4 = (level5 == 0) ? 1 : 0;
    uint16_t level4 = count4 ? 0 : level5;

    // count3: check if high 8 bits are zero (shift left 8 bits)
    uint8_t count3 = ((level4 >> 8) == 0) ? 1 : 0;
    uint16_t level3 = count3 ? ((level4 << 8) & 0xFFFF) : level4;

    // count2: check if high 4 bits are zero (shift left 4 bits)
    uint8_t count2 = ((level3 >> 12) == 0) ? 1 : 0;
    uint16_t level2 = count2 ? ((level3 << 4) & 0xFFFF) : level3;

    // count1: check if high 2 bits are zero (shift left 2 bits)
    uint8_t count1 = ((level2 >> 14) == 0) ? 1 : 0;
    uint16_t level1 = count1 ? ((level2 << 2) & 0xFFFF) : level2;

    // count0: check if MSB is zero (shift left 1 bit)
    uint8_t count0 = ((level1 >> 15) == 0) ? 1 : 0;
    uint16_t level0 = count0 ? ((level1 << 1) & 0xFFFF) : level1;

    Count = (count4 << 4) | (count3 << 3) | (count2 << 2) | (count1 << 1) | count0;
    R = level0;
}

// Right shifter with sticky bit (15-bit input, 17-bit output)
void FPAdd::RightShifterSticky15(uint16_t X, uint8_t S, uint32_t &R, bool &Sticky)
{
    uint8_t ps = S & 0x1F;
    uint32_t Xpadded = (X << 2) & 0x1FFFF;  // shift left 2 bits, pad zeros, 17-bit
    uint32_t level5 = Xpadded;

    bool stk4 = ((level5 & 0xFFFF) != 0 && ((ps >> 4) & 1)) ? true : false;
    uint32_t level4 = ((ps >> 4) & 1) ? (level5 >> 16) : level5;

    bool stk3 = (((level4 & 0xFF) != 0 && ((ps >> 3) & 1)) || stk4) ? true : false;
    uint32_t level3 = ((ps >> 3) & 1) ? (level4 >> 8) : level4;

    bool stk2 = (((level3 & 0xF) != 0 && ((ps >> 2) & 1)) || stk3) ? true : false;
    uint32_t level2 = ((ps >> 2) & 1) ? (level3 >> 4) : level3;

    bool stk1 = (((level2 & 0x3) != 0 && ((ps >> 1) & 1)) || stk2) ? true : false;
    uint32_t level1 = ((ps >> 1) & 1) ? (level2 >> 2) : level2;

    bool stk0 = (((level1 & 0x1) != 0 && (ps & 1)) || stk1) ? true : false;
    uint32_t level0 = (ps & 1) ? (level1 >> 1) : level1;

    R = level0 & 0x1FFFF;  // 17-bit
    Sticky = stk0;
}

// 18-bit integer adder
void FPAdd::IntAdder_18(uint32_t X, uint32_t Y, bool Cin, uint32_t &R)
{
    R = (X + Y + (Cin ? 1 : 0)) & 0x3FFFF;  // 18-bit
}

// 24-bit integer adder
void FPAdd::IntAdder_24(uint32_t X, uint32_t Y, bool Cin, uint32_t &R)
{
    R = (X + Y + (Cin ? 1 : 0)) & 0xFFFFFF;  // 24-bit
}

void FPAdd::stage_input_and_swap()
{
    uint32_t inX = X_reg;
    uint32_t inY = Y_reg;

    // Extract exception bits exn[24:23]
    uint8_t exnX = (inX >> 23) & 0x3;
    uint8_t exnY = (inY >> 23) & 0x3;
    bool exceptionXSuperiorY = (exnX >= exnY);
    bool exceptionXEqualY = (exnX == exnY);

    // Extract exponent exp[21:14]
    uint16_t expX = (inX >> 14) & 0xFF;
    uint16_t expY = (inY >> 14) & 0xFF;
    int16_t signedExpX = expX;
    int16_t signedExpY = expY;

    int16_t expDiffXY = signedExpX - signedExpY;
    uint8_t expDiffYX = (signedExpY - signedExpX) & 0xFF;

    bool swap = (exceptionXEqualY && (expDiffXY < 0)) || (!exceptionXSuperiorY);

    // Swap X and Y directly, no need for negation
    uint32_t newX = swap ? inY : inX;
    uint32_t newY = swap ? inX : inY;
    uint8_t exponentDifference = swap ? expDiffYX : (expDiffXY & 0xFF);

    bool shiftedOut = (exponentDifference >> 5) != 0;
    uint8_t shiftVal = shiftedOut ? 17 : (exponentDifference & 0x1F);

    bool EffSub = ((newX >> 22) & 1) ^ ((newY >> 22) & 1);

    bool selectClosePath = (EffSub && ((exponentDifference >> 1) == 0));

    newX_d1 = newX;
    EffSub_d1 = EffSub;
    selectClosePath_d1 = selectClosePath;

    // Close path: fracXClose1 = {2'b01, newX[13:0], 1'b0} = 17-bit
    uint32_t fracXClose1 = ((1 << 15) | ((newX & 0x3FFF) << 1)) & 0x1FFFF;

    // fracYClose1 selected based on exponentDifference[0]
    uint32_t fracYClose1 =
        (exponentDifference & 1)
            ? ((1 << 14) | (newY & 0x3FFF))          // {3'b001, newY[13:0]} = 17-bit
            : ((1 << 15) | ((newY & 0x3FFF) << 1));  // {2'b01, newY[13:0], 1'b0} = 17-bit
    fracYClose1 &= 0x1FFFF;

    IntDualSub_17(fracXClose1, fracYClose1, fracRClosexMy_reg, fracRCloseyMx_reg);

    // Far path: fracNewY = {1'b1, newY[13:0]} = 15-bit
    uint16_t fracNewY = ((1 << 14) | (newY & 0x3FFF)) & 0x7FFF;
    RightShifterSticky15(fracNewY, shiftVal, shiftedFracY_reg, sticky_reg);

    // Prepare far path adder inputs
    uint32_t fracYfar = shiftedFracY_reg & 0x1FFFF;  // 17-bit -> pad to 18-bit
    uint32_t EffSubVector = EffSub ? 0x3FFFF : 0;    // 18-bit all 1s or all 0s
    uint32_t fracYfarXorOp = fracYfar ^ EffSubVector;
    uint32_t fracXfar =
        ((1 << 16) | ((newX & 0x3FFF) << 2)) & 0x3FFFF;  // {2'b01, newX[13:0], 2'b00} = 18-bit

    X_adder18_d1 = fracXfar;
    Y_adder18_d1 = fracYfarXorOp;
}

void FPAdd::stage_close_far_path()
{
    // Close path processing
    bool fracSignClose = (fracRClosexMy_reg >> 16) & 1;
    uint16_t fracRClose1 =
        fracSignClose ? (fracRCloseyMx_reg & 0xFFFF) : (fracRClosexMy_reg & 0xFFFF);  // 16-bit

    bool resSign = (selectClosePath_d1 && (fracRClose1 == 0))
                       ? false
                       : (((newX_d1 >> 22) & 1) ^ (selectClosePath_d1 && fracSignClose));

    Normalizer_Z_16(fracRClose1, nZerosNew_reg, shiftedFrac_reg);

    bool roundClose0 = (shiftedFrac_reg & 1) && ((shiftedFrac_reg >> 1) & 1);
    bool resultCloseIsZero0 = (nZerosNew_reg == 16);

    uint16_t expXnew = (newX_d1 >> 14) & 0xFF;
    uint16_t exponentResultClose = (expXnew - nZerosNew_reg) & 0x3FF;
    uint32_t resultBeforeRoundClose =
        ((exponentResultClose << 14) | ((shiftedFrac_reg >> 1) & 0x3FFF)) & 0xFFFFFF;  // 24-bit

    // Far path processing
    bool cInAddFar = EffSub_d1 && (!sticky_reg);
    IntAdder_18(X_adder18_d1, Y_adder18_d1, cInAddFar, fracResultfar0_reg);

    uint8_t fracLeadingBits = (fracResultfar0_reg >> 16) & 0x3;
    uint16_t fracResultFar1 = 0;
    bool fracResultRoundBit = false;
    bool fracResultStickyBit = false;

    switch (fracLeadingBits) {
        case 0:
            fracResultFar1 = (fracResultfar0_reg >> 1) & 0x3FFF;
            fracResultRoundBit = fracResultfar0_reg & 1;
            fracResultStickyBit = sticky_reg;
            break;

        case 1:
            fracResultFar1 = (fracResultfar0_reg >> 2) & 0x3FFF;
            fracResultRoundBit = (fracResultfar0_reg >> 1) & 1;
            fracResultStickyBit = (fracResultfar0_reg & 1) || sticky_reg;
            break;

        default:
            fracResultFar1 = (fracResultfar0_reg >> 3) & 0x3FFF;
            fracResultRoundBit = (fracResultfar0_reg >> 2) & 1;
            fracResultStickyBit = ((fracResultfar0_reg & 0x3) != 0) || sticky_reg;
            break;
    }

    bool roundFar1 = fracResultRoundBit && (fracResultStickyBit || (fracResultFar1 & 1));

    uint16_t expOperationSel = (fracLeadingBits == 0) ? 3 : (fracLeadingBits == 1) ? 0 : 1;
    int16_t exponentUpdate = (expOperationSel == 3) ? -1 : (expOperationSel == 0) ? 0 : 1;

    uint16_t expXbase = (newX_d1 >> 14) & 0xFF;
    exponentResultfar0_d1 = (expXbase + exponentUpdate) & 0x3FF;

    uint32_t resultBeforeRoundFar =
        ((exponentResultfar0_d1 << 14) | fracResultFar1) & 0xFFFFFF;  // 24-bit

    // Select Close or Far path result
    uint32_t resultBeforeRound = selectClosePath_d1 ? resultBeforeRoundClose : resultBeforeRoundFar;
    bool round = selectClosePath_d1 ? roundClose0 : roundFar1;
    bool zeroFromClose = selectClosePath_d1 && resultCloseIsZero0;

    // Final rounding addition
    Y_adder24_d1 = 0;
    IntAdder_24(resultBeforeRound, Y_adder24_d1, round, resultRounded_reg);

    // Synchronize signals
    syncEffSub_d1 = EffSub_d1;
    syncX_d1 = newX_d1;
    syncSignY_d1 = (newX_d1 >> 22) & 1;
    syncResSign_d1 = resSign;
    syncExnXY_d1 = ((newX_d1 >> 23) & 0x3) << 2 | ((Y_reg >> 23) & 0x3);
}

void FPAdd::stage_final_round()
{
    uint8_t UnderflowOverflow = (resultRounded_reg >> 22) & 0x3;

    uint8_t exnR = 0;
    switch (syncExnXY_d1) {
        case 0x5:
            if (UnderflowOverflow == 1) {
                exnR = 2;  // overflow
            } else if (UnderflowOverflow >= 2) {
                exnR = 0;  // underflow
            } else {
                exnR = 1;  // normal
            }
            break;

        case 0xA:
            exnR = 2 | (syncEffSub_d1 ? 1 : 0);
            break;

        case 0xE:
            exnR = 3;
            break;

        default:
            exnR = (syncExnXY_d1 >> 2) & 0x3;
            break;
    }

    bool sgnR = false;
    uint32_t expsigR = 0;

    switch (syncExnXY_d1) {
        case 0x5:
            sgnR = syncResSign_d1;
            expsigR = resultRounded_reg & 0x3FFFFF;  // 22-bit
            break;

        case 0x0:
            sgnR = ((syncX_d1 >> 22) & 1) && syncSignY_d1;
            expsigR = syncX_d1 & 0x3FFFFF;
            break;

        default:
            sgnR = (syncX_d1 >> 22) & 1;
            expsigR = syncX_d1 & 0x3FFFFF;
            break;
    }

    R = ((exnR << 23) | (sgnR ? (1 << 22) : 0) | expsigR) & 0x1FFFFFF;  // 25-bit
}

void FPAdd::clock_step()
{
    /*
     * FPAdd computes X + Y (with proper sign handling via exn bits)
     */
    stage_input_and_swap();
    stage_close_far_path();
    stage_final_round();
}
