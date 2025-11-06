#ifndef FPADD_H
#define FPADD_H

#include <cstdint>

/**
 * @brief FPAdd - Floating-point adder (wE=8, wF=14)
 * 
 * Data format:
 * - Input: 25-bit extended bf16 format (exn(2) + sign(1) + exp(8) + mantissa(14))
 * - Output: 25-bit extended bf16 format (exn(2) + sign(1) + exp(8) + mantissa(14))
 * 
 * Bit field definition:
 * - exn[24:23]: Exception bits (00=zero, 01=normal, 10=infinity, 11=NaN)
 * - sign[22]: Sign bit
 * - exp[21:14]: 8-bit exponent
 * - frac[13:0]: 14-bit mantissa (without implicit 1)
 * 
 * Pipeline: 2-stage pipeline
 */
class FPAdd
{
private:
    // ========== Stage 0 Registers (Input) ==========
    uint32_t X_reg;  // 25-bit input X
    uint32_t Y_reg;  // 25-bit input Y
    uint32_t R;      // 25-bit output result

    // ========== Stage 0->1 Registers ==========
    uint32_t newX_d1;
    bool EffSub_d1;
    bool selectClosePath_d1;
    uint16_t exponentResultfar0_d1;

    // Close path
    uint32_t fracRClosexMy_reg;  // 18-bit
    uint32_t fracRCloseyMx_reg;  // 18-bit

    // Normalizer
    uint8_t count3_d1;
    uint16_t level3_d1;
    uint8_t count2_d1;
    uint8_t nZerosNew_reg;
    uint16_t shiftedFrac_reg;

    // Right shifter
    uint8_t ps_d1;
    bool stk2_d1;
    uint32_t level2_d1;
    uint32_t level1_d1;
    uint32_t shiftedFracY_reg;  // 18-bit
    bool sticky_reg;

    // Far path adder
    uint32_t X_adder18_d1;
    uint32_t Y_adder18_d1;
    uint32_t fracResultfar0_reg;

    // ========== Stage 1->2 Registers ==========
    bool syncEffSub_d1;
    uint32_t syncX_d1;
    bool syncSignY_d1;
    bool syncResSign_d1;
    uint8_t syncExnXY_d1;
    uint32_t Y_adder24_d1;
    uint32_t resultRounded_reg;

    // ========== Internal Functions ==========
    void IntDualSub_17(uint32_t X, uint32_t Y, uint32_t &XmY, uint32_t &YmX);
    void Normalizer_Z_16(uint16_t X, uint8_t &Count, uint16_t &R);
    void RightShifterSticky15(uint16_t X, uint8_t S, uint32_t &R, bool &Sticky);
    void IntAdder_18(uint32_t X, uint32_t Y, bool Cin, uint32_t &R);
    void IntAdder_24(uint32_t X, uint32_t Y, bool Cin, uint32_t &R);

    void stage_input_and_swap();
    void stage_close_far_path();
    void stage_final_round();

public:
    FPAdd();
    void reset();
    void set_input(uint32_t X, uint32_t Y);
    void clock_step();

    /**
     * @brief Get 25-bit extended format output (exn(2) + sign(1) + exp(8) + mantissa(14))
     * @return 25-bit extended bf16 result
     */
    uint32_t get_result_extended() const
    {
        return R & 0x1FFFFFF;
    }
};

#endif  // FPADD_H
