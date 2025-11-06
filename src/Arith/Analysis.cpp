#include <cstdint>

#include "Arith/Analysis.h"
#include "Arith/FPAdd.h"

Analysis::Analysis()
{
    // Initialize LUT (2u23 format, 25-bit): Exp(2^-1) to Exp(2^-23)
    // Hardware reference: Analysis.sv lines 136-162
    Exp_Power_Space_LUT[0] = 0x0D3094C;   // 2^-1
    Exp_Power_Space_LUT[1] = 0x0A45AF2;   // 2^-2
    Exp_Power_Space_LUT[2] = 0x0910B02;   // 2^-3
    Exp_Power_Space_LUT[3] = 0x088415B;   // 2^-4
    Exp_Power_Space_LUT[4] = 0x084102B;   // 2^-5
    Exp_Power_Space_LUT[5] = 0x0820405;   // 2^-6
    Exp_Power_Space_LUT[6] = 0x0810101;   // 2^-7
    Exp_Power_Space_LUT[7] = 0x0808040;   // 2^-8
    Exp_Power_Space_LUT[8] = 0x0804010;   // 2^-9
    Exp_Power_Space_LUT[9] = 0x0802004;   // 2^-10
    Exp_Power_Space_LUT[10] = 0x0801001;  // 2^-11
    Exp_Power_Space_LUT[11] = 0x0800800;  // 2^-12
    Exp_Power_Space_LUT[12] = 0x0800400;  // 2^-13
    Exp_Power_Space_LUT[13] = 0x0800200;  // 2^-14
    Exp_Power_Space_LUT[14] = 0x0800100;  // 2^-15
    Exp_Power_Space_LUT[15] = 0x0800080;  // 2^-16
    Exp_Power_Space_LUT[16] = 0x0800040;  // 2^-17
    Exp_Power_Space_LUT[17] = 0x0800020;  // 2^-18
    Exp_Power_Space_LUT[18] = 0x0800010;  // 2^-19
    Exp_Power_Space_LUT[19] = 0x0800008;  // 2^-20
    Exp_Power_Space_LUT[20] = 0x0800004;  // 2^-21
    Exp_Power_Space_LUT[21] = 0x0800002;  // 2^-22
    Exp_Power_Space_LUT[22] = 0x0800001;  // 2^-23

    // Initialize pipeline vectors
    pipeline_valid.resize(PIPELINE_DEPTH + 1, false);
    pipeline_mult_result.resize(PIPELINE_DEPTH, 0x0800000);  // 1.0 in 2u23
    pipeline_select_power.resize(PIPELINE_DEPTH, 0);
    pipeline_ufixX.resize(PIPELINE_DEPTH, 0);
    pipeline_start_index.resize(PIPELINE_DEPTH, 0);
    pipeline_bias_kcm.resize(PIPELINE_DEPTH, 0);

    // Create FPAdd module instance
    fadd_module = new FPAdd();

    reset();
}

Analysis::~Analysis()
{
    delete fadd_module;
}

void Analysis::reset()
{
    // Manual reset of all pipeline stages
    for (int i = 0; i < PIPELINE_DEPTH + 1; i++) {
#pragma HLS PIPELINE II = 1
        pipeline_valid[i] = false;
    }
    for (int i = 0; i < PIPELINE_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
        pipeline_mult_result[i] = 0x0800000;  // 1.0 in 2u23
        pipeline_select_power[i] = 0;
        pipeline_ufixX[i] = 0;
        pipeline_start_index[i] = 0;
        pipeline_bias_kcm[i] = 0;
    }

    X_reg = 0;
    valid_in_reg = false;
    wE = 0;
    wF = 0;
    ufixX = 0;
    bias = 0;
    Ln_Ans_reg = 0;

    // Reset FPAdd module
    fadd_module->reset();
}

void Analysis::set_input(uint32_t X, bool valid_in)
{
    X_reg = X & 0x7FFFFF;  // 23-bit
    valid_in_reg = valid_in;
}

// 2u23 fixed-point multiplication (Analysis.sv lines 262-269)
uint32_t Analysis::fixed_mult_2q23(uint32_t a, uint32_t b)
{
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return (product >> 23) & 0x1FFFFFF;  // 25-bit
}

// Find starting index in LUT (Analysis.sv lines 176-252)
uint8_t Analysis::find_start_index(uint32_t target_value)
{
    for (uint8_t i = 0; i < LUT_SIZE; i++) {
#pragma HLS PIPELINE II = 1
        if (target_value >= Exp_Power_Space_LUT[i]) {
            return i;
        }
    }
    return LUT_SIZE;  // Out of range
}

// KCM: bias * ln(2) in Q7.16 format (Analysis.sv lines 278-281)
// ln(2) ≈ 0.693147180559945309417232121458
// Q7.16: ln(2) * 2^16 = 45426 = 0xB172
uint32_t Analysis::bias_times_ln2_kcm(int16_t bias_val)
{
    // Simple constant multiplication: bias * ln(2)
    // Result in Q7.16 format (24-bit: 7 integer + 16 fractional + 1 sign)
    int32_t result = static_cast<int32_t>(bias_val) * 45426;  // Q7.16
    return result & 0xFFFFFF;                                 // 24-bit
}

// Convert Q7.16 fixed-point to 25-bit float (Analysis.sv lines 284-420)
uint32_t Analysis::fixed_to_float_q7_16(uint32_t fixed_value)
{
    // Extract sign
    bool sign = (fixed_value >> 23) & 0x1;
    uint32_t abs_value = sign ? ((~fixed_value & 0x7FFFFF) + 1) : (fixed_value & 0x7FFFFF);

    // Check for zero
    if (abs_value == 0) {
        return 0x0000000;  // exn=00 (zero)
    }

    // Leading zero detection
    uint8_t leading_zeros = 0;
    uint32_t normalized_mantissa = abs_value;

    for (int bit = 22; bit >= 0; bit--) {
        if (abs_value & (1 << bit)) {
            leading_zeros = 22 - bit;
            normalized_mantissa = abs_value << leading_zeros;
            break;
        }
    }

    // Calculate exponent
    int16_t first_one_pos = 22 - leading_zeros;
    int16_t integer_weight = first_one_pos - 16;  // Q7.16 format
    int16_t computed_exp = 127 + integer_weight;

    // Check for overflow/underflow
    if (computed_exp <= 0) {
        // Underflow to zero
        return 0x0800000 | (sign << 22);  // exn=01, zero mantissa
    } else if (computed_exp >= 255) {
        // Overflow to infinity
        return 0x13FC000 | (sign << 22);  // exn=10 (infinity)
    }

    // Extract mantissa (bits [21:8] = 14 bits)
    uint32_t output_mantissa = (normalized_mantissa >> 8) & 0x3FFF;
    uint32_t output_exp = computed_exp & 0xFF;

    // Return: exn(2)=01 + sign(1) + exp(8) + mantissa(14)
    return 0x0800000 | (sign << 22) | (output_exp << 14) | output_mantissa;
}

// Convert greedy algorithm result to float (Analysis.sv lines 89-131)
uint32_t Analysis::greedy_to_float(uint32_t select_power_val, uint8_t start_idx)
{
    if (select_power_val == 0) {
        return 0x0000000;  // exn=00 (zero)
    }

    // Hardware uses start_idx to infer highest bit position
    // Assumes first iteration selected LUT[start_idx] -> bit[(LUT_SIZE-1) - start_idx]
    uint8_t highest_bit_pos = (LUT_SIZE - 1) - start_idx;

    // Calculate shift amount to bit[22]
    uint8_t shift_amount = 22 - highest_bit_pos;

    // Normalize
    uint32_t normalized_mantissa = select_power_val << shift_amount;

    // Calculate exponent for Q0.23 fixed-point format
    // select_power format: bit[22]=2^-1, bit[21]=2^-2, ..., bit[0]=2^-23
    // If bit[k] is the highest bit, the value is approximately 2^-(23-k)
    // To represent 2^-(23-k) in IEEE 754: 1.xxx × 2^(-(23-k))
    // IEEE 754 exponent = 127 + (-(23-k)) = 127 - 23 + k = 104 + k
    int16_t computed_exp = 104 + highest_bit_pos;

    if (computed_exp < 0) {
        return 0x0800000;  // exn=01, exp=0
    } else if (computed_exp >= 255) {
        return 0x13FC000;  // exn=10 (infinity)
    }

    // Extract mantissa [21:8]
    uint32_t output_mantissa = (normalized_mantissa >> 8) & 0x3FFF;
    uint32_t output_exp = computed_exp & 0xFF;

    // Return: exn(2)=01 + sign(1)=0 + exp(8) + mantissa(14)
    return 0x0800000 | (output_exp << 14) | output_mantissa;
}

void Analysis::clock_step()
{
    // Extract input fields (Analysis.sv lines 167-170)
    wE = (X_reg >> DATA_WF) & 0xFF;
    wF = ((X_reg & 0x3FFF) | 0x4000);                      // {1'b1, X[13:0]} = 15-bit
    ufixX = (static_cast<uint32_t>(wF) << 9) & 0x1FFFFFF;  // 25-bit: {0, wF[14:0], 9'b0}
    bias = static_cast<int16_t>(wE) - BIAS_VALUE;

    // Compute KCM result for stage 1 (before pipeline shift)
    uint32_t new_bias_kcm = bias_times_ln2_kcm(bias);

    // Pipeline shift (backward iteration to avoid overwriting)
    for (int j = PIPELINE_DEPTH - 1; j > 0; j--) {
#pragma HLS PIPELINE II = 1
        pipeline_valid[j] = pipeline_valid[j - 1];
        pipeline_ufixX[j] = pipeline_ufixX[j - 1];
        pipeline_start_index[j] = pipeline_start_index[j - 1];

        // Shift bias_kcm pipeline (from stage 1 onwards)
        if (j > 1) {
            pipeline_bias_kcm[j] = pipeline_bias_kcm[j - 1];
        }
    }

    // Stage 0: Accept new input
    if (valid_in_reg) {
        pipeline_valid[0] = true;
        pipeline_mult_result[0] = 0x0800000;  // 1.0 in 2u23
        pipeline_select_power[0] = 0;
        pipeline_ufixX[0] = ufixX;
        pipeline_start_index[0] = find_start_index(ufixX);
    } else {
        pipeline_valid[0] = false;
    }

    // Stage 1: Set KCM result (after backward shift completes)
    if (pipeline_valid[0]) {
        pipeline_bias_kcm[1] = new_bias_kcm;
    }

    // Stage 1-15: Greedy algorithm iterations
    for (int j = 1; j <= DATA_WF + 1; j++) {
#pragma HLS PIPELINE II = 1
        if (pipeline_valid[j - 1]) {
            uint8_t current_index = pipeline_start_index[j - 1] + (j - 1);

            if (current_index < LUT_SIZE) {
                uint32_t lut_value = Exp_Power_Space_LUT[current_index];
                uint32_t mult_temp = fixed_mult_2q23(pipeline_mult_result[j - 1], lut_value);

                if (mult_temp <= pipeline_ufixX[j - 1]) {
                    // Select this term
                    pipeline_mult_result[j] = mult_temp;
                    pipeline_select_power[j] =
                        pipeline_select_power[j - 1] | (1 << ((LUT_SIZE - 1) - current_index));
                } else {
                    // Don't select (hardware clears the bit, but for us it's already 0)
                    pipeline_mult_result[j] = pipeline_mult_result[j - 1];
                    pipeline_select_power[j] =
                        pipeline_select_power[j - 1] & (~(1 << ((LUT_SIZE - 1) - current_index)));
                }
            } else {
                // Out of range, keep current state
                pipeline_mult_result[j] = pipeline_mult_result[j - 1];
                pipeline_select_power[j] = pipeline_select_power[j - 1];
            }
        }
    }

    // ========== Combinational Logic: Float Conversion (based on Stage OUTPUT_WF+1) ==========
    // Hardware: Combinational logic computes immediately when Stage 15 completes
    // These signals connect to FPAdd module input ports
    uint32_t greedy_float_wire = 0;
    uint32_t bias_float_wire = 0;

    if (pipeline_valid[DATA_WF + 1]) {
        greedy_float_wire =
            greedy_to_float(pipeline_select_power[DATA_WF + 1], pipeline_start_index[DATA_WF + 1]);
        bias_float_wire = fixed_to_float_q7_16(pipeline_bias_kcm[DATA_WF + 1]);

        // Set FPAdd inputs (simulating hardware port connections)
        fadd_module->set_input(greedy_float_wire, bias_float_wire);
    }

    // ========== FPAdd Module: Continuously Running (simulating hardware module) ==========
    // In hardware, FPAdd is an independent module that runs every cycle
    fadd_module->clock_step();

    // ========== Stage OUTPUT_WF+2 and OUTPUT_WF+3: Valid Signal Propagation ==========
    // Hardware: pipeline_valid[16] <= pipeline_valid[15]
    //           pipeline_valid[17] <= pipeline_valid[16]
    pipeline_valid[DATA_WF + 3] = pipeline_valid[DATA_WF + 2];
    pipeline_valid[DATA_WF + 2] = pipeline_valid[DATA_WF + 1];

    // ========== Output: Read FPAdd Result ==========
    // Hardware: assign Ln_Ans = fadd_R[OUTPUT_WE + OUTPUT_WF : 0]
    // FPAdd output directly connects to module output
    Ln_Ans_reg = fadd_module->get_result_extended();
}

bool Analysis::get_valid_out() const
{
#pragma HLS INLINE
    return pipeline_valid[PIPELINE_DEPTH];
}

uint32_t Analysis::get_Ln_Ans_extended() const
{
    // Output is already in 25-bit format: exn(2) + sign(1) + exp(8) + mantissa(14)
    // This is the complete ln(X) value computed by hardware
#pragma HLS INLINE
    return Ln_Ans_reg;
}
