#ifndef EXP_H
#define EXP_H

#include <cstdint>

class Exp
{
public:
    Exp();
    void reset();
    void clock_step();

    // Input: wE=8, wF=14 format (25-bit total: exn(2) + sign(1) + exp(8) + mantissa(14))
    void set_input(uint32_t X, bool valid_in);

    bool get_valid_out() const
    {
        return valid_out;
    }
    uint32_t get_Exp_Ans() const
    {
        return Exp_Ans;
    }  // 26-bit (1Q25 format)

    // Extended bf16 format output: 25-bit = exn(2) + sign(1) + exp(8) + mantissa(14)
    // This method applies K_final to convert exp(Y) to exp(X)
    uint32_t get_Exp_Ans_extended() const;

    uint8_t get_absInt_K() const
    {
        return absInt_K;
    }
    uint32_t get_Exp_Y() const
    {
        return Exp_Y;  // 18-bit value for wF=14
    }

private:
    static const int DATA_WE = 8;
    static const int DATA_WF = 14;      // 14-bit mantissa (upgraded from 13)
    static const int EXP_Y_WIDTH = 18;  // Y value width (upgraded from 14)
    static const int Guard_Bits = 2;
    static const int UFIX_DATA_WIDTH = 25;  // Fixed-point data width (7,-18)
    static const int UFIX_DATA_WF = 18;     // Fixed-point fractional bits
    static const int LUT_Width = 24;
    static const int Data_Width =
        25;  // Input data width (25-bit: exn(2) + sign(1) + exp(8) + mantissa(14))
    static const int TREE_DEPTH = 5;
    static const int MAX_NODES = 18;  // Updated for EXP_Y_WIDTH=18

    // Lookup tables
    uint16_t Exp_inv_ln2_LUT_t0[32];
    uint8_t Exp_inv_ln2_LUT_t1[32];
    uint32_t Exp_ln2_LUT_t0[32];
    uint32_t Exp_ln2_LUT_t1[8];

    // Pipeline registers
    uint8_t absInt_K;
    int16_t K_value;          // Signed K value (stage 0)
    int16_t K_value_pipe[7];  // K value pipeline (TREE_DEPTH+2 stages)
    uint32_t Exp_Y;           // 18-bit Y value (upgraded from 16-bit)
    uint16_t Exp_Result;
    bool valid_pipe[8];

    // Special case handling
    bool is_special_case_pipe[8];
    uint32_t special_result_pipe[8];  // 26-bit 1Q25 format

    // Binary tree data
    uint32_t tree_data[6][18];
    bool tree_valid[6][18];

    // Input (25-bit: exn(2) + sign(1) + exp(8) + mantissa(14))
    uint32_t X_reg;
    bool valid_in_reg;

    // Output
    uint32_t Exp_Ans;
    bool valid_out;

    // Internal signals
    uint8_t wE;
    uint32_t uxFix;  // 25-bit fixed-point value
    int16_t exp_diff;

    void extract_fields();
    void stage0_input();
    void stage1_K_calculation();
    void stage2_Y_calculation();
    void stage3_tree_input();
    void stage4_7_tree_multiply();
    uint32_t fixed_mult_1q23_guard(uint32_t a, uint32_t b);
    int calc_level_nodes(int level);
    int count_leading_zeros(uint32_t value) const;
};

#endif  // EXP_H
