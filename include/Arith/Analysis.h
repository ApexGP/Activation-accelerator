#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <cstdint>
#include <vector>

class FPAdd;  // Forward declaration

class Analysis
{
public:
    Analysis();
    ~Analysis();
    void reset();
    void clock_step();

    // Input: 23-bit format (sign + exp(8) + mantissa(14))
    void set_input(uint32_t X, bool valid_in);

    // Output getters
    bool get_valid_out() const;

    // Output: 25-bit format = exn(2) + sign(1) + exp(8) + mantissa(14)
    // This is the complete ln(X) value (hardware already computed bias*ln(2) + fractional part)
    uint32_t get_Ln_Ans_extended() const;

private:
    static const int DATA_WE = 8;
    static const int DATA_WF = 14;
    static const int FIX_DATA_WIDTH = 25;  // 2u23 format
    static const int LUT_SIZE = 23;
    static const int PIPELINE_DEPTH = 17;  // DATA_WF + 3
    static const int BIAS_VALUE = 127;

    // Lookup table (2u23 format, 25-bit)
    uint32_t Exp_Power_Space_LUT[LUT_SIZE];

    // Pipeline registers
    std::vector<bool> pipeline_valid;
    std::vector<uint32_t> pipeline_mult_result;   // 25-bit 2u23
    std::vector<uint32_t> pipeline_select_power;  // 23-bit
    std::vector<uint32_t> pipeline_ufixX;         // 25-bit
    std::vector<uint8_t> pipeline_start_index;    // 5-bit
    std::vector<uint32_t> pipeline_bias_kcm;      // 24-bit Q7.16

    // Input registers
    uint32_t X_reg;  // 23-bit
    bool valid_in_reg;

    // Extracted fields
    uint8_t wE;      // 8-bit
    uint16_t wF;     // 15-bit (1 + 14)
    uint32_t ufixX;  // 25-bit 2u23
    int16_t bias;    // signed

    // Final output (from FPAdd)
    uint32_t Ln_Ans_reg;  // 25-bit

    // FPAdd module instance (持续运行，模拟硬件模块)
    FPAdd *fadd_module;

    // Helper functions
    uint32_t fixed_mult_2q23(uint32_t a, uint32_t b);
    uint8_t find_start_index(uint32_t target_value);
    uint32_t bias_times_ln2_kcm(int16_t bias_val);
    uint32_t fixed_to_float_q7_16(uint32_t fixed_value);
    uint32_t greedy_to_float(uint32_t select_power_val, uint8_t start_idx);
};

#endif  // ANALYSIS_H
