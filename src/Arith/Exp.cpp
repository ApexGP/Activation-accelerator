#include <cstdint>

#include "Arith/Exp.h"

Exp::Exp()
{
    // Initialize 1/ln2 KCM Table0 (12-bit, 32 entries)
    Exp_inv_ln2_LUT_t0[0] = 0x008;
    Exp_inv_ln2_LUT_t0[1] = 0x064;
    Exp_inv_ln2_LUT_t0[2] = 0x0C1;
    Exp_inv_ln2_LUT_t0[3] = 0x11D;
    Exp_inv_ln2_LUT_t0[4] = 0x179;
    Exp_inv_ln2_LUT_t0[5] = 0x1D6;
    Exp_inv_ln2_LUT_t0[6] = 0x232;
    Exp_inv_ln2_LUT_t0[7] = 0x28E;
    Exp_inv_ln2_LUT_t0[8] = 0x2EB;
    Exp_inv_ln2_LUT_t0[9] = 0x347;
    Exp_inv_ln2_LUT_t0[10] = 0x3A3;
    Exp_inv_ln2_LUT_t0[11] = 0x400;
    Exp_inv_ln2_LUT_t0[12] = 0x45C;
    Exp_inv_ln2_LUT_t0[13] = 0x4B8;
    Exp_inv_ln2_LUT_t0[14] = 0x515;
    Exp_inv_ln2_LUT_t0[15] = 0x571;
    Exp_inv_ln2_LUT_t0[16] = 0x5CD;
    Exp_inv_ln2_LUT_t0[17] = 0x62A;
    Exp_inv_ln2_LUT_t0[18] = 0x686;
    Exp_inv_ln2_LUT_t0[19] = 0x6E2;
    Exp_inv_ln2_LUT_t0[20] = 0x73F;
    Exp_inv_ln2_LUT_t0[21] = 0x79B;
    Exp_inv_ln2_LUT_t0[22] = 0x7F7;
    Exp_inv_ln2_LUT_t0[23] = 0x854;
    Exp_inv_ln2_LUT_t0[24] = 0x8B0;
    Exp_inv_ln2_LUT_t0[25] = 0x90C;
    Exp_inv_ln2_LUT_t0[26] = 0x969;
    Exp_inv_ln2_LUT_t0[27] = 0x9C5;
    Exp_inv_ln2_LUT_t0[28] = 0xA21;
    Exp_inv_ln2_LUT_t0[29] = 0xA7E;
    Exp_inv_ln2_LUT_t0[30] = 0xADA;
    Exp_inv_ln2_LUT_t0[31] = 0xB36;

    // Initialize 1/ln2 KCM Table1 (7-bit, 32 entries)
    Exp_inv_ln2_LUT_t1[0] = 0x00;
    Exp_inv_ln2_LUT_t1[1] = 0x03;
    Exp_inv_ln2_LUT_t1[2] = 0x06;
    Exp_inv_ln2_LUT_t1[3] = 0x09;
    Exp_inv_ln2_LUT_t1[4] = 0x0C;
    Exp_inv_ln2_LUT_t1[5] = 0x0E;
    Exp_inv_ln2_LUT_t1[6] = 0x11;
    Exp_inv_ln2_LUT_t1[7] = 0x14;
    Exp_inv_ln2_LUT_t1[8] = 0x17;
    Exp_inv_ln2_LUT_t1[9] = 0x1A;
    Exp_inv_ln2_LUT_t1[10] = 0x1D;
    Exp_inv_ln2_LUT_t1[11] = 0x20;
    Exp_inv_ln2_LUT_t1[12] = 0x23;
    Exp_inv_ln2_LUT_t1[13] = 0x26;
    Exp_inv_ln2_LUT_t1[14] = 0x28;
    Exp_inv_ln2_LUT_t1[15] = 0x2B;
    Exp_inv_ln2_LUT_t1[16] = 0x2E;
    Exp_inv_ln2_LUT_t1[17] = 0x31;
    Exp_inv_ln2_LUT_t1[18] = 0x34;
    Exp_inv_ln2_LUT_t1[19] = 0x37;
    Exp_inv_ln2_LUT_t1[20] = 0x3A;
    Exp_inv_ln2_LUT_t1[21] = 0x3D;
    Exp_inv_ln2_LUT_t1[22] = 0x3F;
    Exp_inv_ln2_LUT_t1[23] = 0x42;
    Exp_inv_ln2_LUT_t1[24] = 0x45;
    Exp_inv_ln2_LUT_t1[25] = 0x48;
    Exp_inv_ln2_LUT_t1[26] = 0x4B;
    Exp_inv_ln2_LUT_t1[27] = 0x4E;
    Exp_inv_ln2_LUT_t1[28] = 0x51;
    Exp_inv_ln2_LUT_t1[29] = 0x54;
    Exp_inv_ln2_LUT_t1[30] = 0x57;
    Exp_inv_ln2_LUT_t1[31] = 0x59;

    // Initialize K×ln2 KCM Table0 (26-bit, 32 entries)
    Exp_ln2_LUT_t0[0] = 0x0000000;
    Exp_ln2_LUT_t0[1] = 0x0162E43;
    Exp_ln2_LUT_t0[2] = 0x02C5C86;
    Exp_ln2_LUT_t0[3] = 0x0428AC9;
    Exp_ln2_LUT_t0[4] = 0x058B90C;
    Exp_ln2_LUT_t0[5] = 0x06EE74F;
    Exp_ln2_LUT_t0[6] = 0x0851592;
    Exp_ln2_LUT_t0[7] = 0x09B43D5;
    Exp_ln2_LUT_t0[8] = 0x0B17218;
    Exp_ln2_LUT_t0[9] = 0x0C7A05B;
    Exp_ln2_LUT_t0[10] = 0x0DDCE9E;
    Exp_ln2_LUT_t0[11] = 0x0F3FCE1;
    Exp_ln2_LUT_t0[12] = 0x10A2B24;
    Exp_ln2_LUT_t0[13] = 0x1205967;
    Exp_ln2_LUT_t0[14] = 0x13687AA;
    Exp_ln2_LUT_t0[15] = 0x14CB5ED;
    Exp_ln2_LUT_t0[16] = 0x162E430;
    Exp_ln2_LUT_t0[17] = 0x1791273;
    Exp_ln2_LUT_t0[18] = 0x18F40B6;
    Exp_ln2_LUT_t0[19] = 0x1A56EF9;
    Exp_ln2_LUT_t0[20] = 0x1BB9D3C;
    Exp_ln2_LUT_t0[21] = 0x1D1CB7F;
    Exp_ln2_LUT_t0[22] = 0x1E7F9C2;
    Exp_ln2_LUT_t0[23] = 0x1FE2805;
    Exp_ln2_LUT_t0[24] = 0x2145648;
    Exp_ln2_LUT_t0[25] = 0x22A848B;
    Exp_ln2_LUT_t0[26] = 0x240B2CE;
    Exp_ln2_LUT_t0[27] = 0x256E111;
    Exp_ln2_LUT_t0[28] = 0x26D0F54;
    Exp_ln2_LUT_t0[29] = 0x2833D97;
    Exp_ln2_LUT_t0[30] = 0x2996BDA;
    Exp_ln2_LUT_t0[31] = 0x2AF9A1D;

    // Initialize K×ln2 KCM Table1 (21-bit, 8 entries)
    Exp_ln2_LUT_t1[0] = 0x000000;
    Exp_ln2_LUT_t1[1] = 0x02C5C8;
    Exp_ln2_LUT_t1[2] = 0x058B91;
    Exp_ln2_LUT_t1[3] = 0x085159;
    Exp_ln2_LUT_t1[4] = 0x0B1721;
    Exp_ln2_LUT_t1[5] = 0x0DDCEA;
    Exp_ln2_LUT_t1[6] = 0x10A2B2;
    Exp_ln2_LUT_t1[7] = 0x13687B;

    reset();
}

void Exp::reset()
{
    absInt_K = 0;
    K_value = 0;
    Exp_Y = 0;
    Exp_Result = 0;
    X_reg = 0;
    valid_in_reg = false;
    Exp_Ans = 0;
    valid_out = false;
    wE = 0;
    uxFix = 0;
    exp_diff = 0;

    for (int i = 0; i < 7; i++) {
        K_value_pipe[i] = 0;
    }

    for (int i = 0; i < 8; i++) {
        valid_pipe[i] = false;
        is_special_case_pipe[i] = false;
        special_result_pipe[i] = 0;
    }

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 18; j++) {
            tree_data[i][j] = 0;
            tree_valid[i][j] = false;
        }
    }
}

void Exp::set_input(uint32_t X, bool valid_in)
{
    X_reg = X & 0x1FFFFFF;  // Mask to 25 bits
    valid_in_reg = valid_in;
}

void Exp::extract_fields()
{
    // Input format: 25-bit = 2 exn bits + 1 sign bit + 8 exponent bits + 14 mantissa bits
    // exn bits (bit 24-23): 00=normal, 01=zero, 10=infinity, 11=NaN
    // For Exp module, we expect normal numbers (exn=00), so we can ignore exn bits
    // Extract exponent (bits 21-14)
    wE = (X_reg >> DATA_WF) & 0xFF;

    // Calculate exponent difference: exp_diff = wE - 127
    int16_t exp_diff = static_cast<int16_t>(wE) - 127;

    // Construct uxFix (25-bit fixed-point: {6'b0, 1'b1, X[13:0], 4'b0})
    // Format: unsigned fixed-point (7,-18)
    uint32_t mantissa_14bit = X_reg & 0x3FFF;  // Extract mantissa[13:0]
    uint32_t uxFix_base = ((1 << 18) | (mantissa_14bit << 4)) & 0x1FFFFFF;  // 25-bit

    // Shift according to exp_diff (consistent with hardware reference)
    if (exp_diff >= 0) {
        // Left shift
        uxFix = (uxFix_base << exp_diff) & 0x1FFFFFF;
    } else {
        // Right shift
        uxFix = (uxFix_base >> (-exp_diff)) & 0x1FFFFFF;
    }
}

void Exp::stage0_input()
{
    if (valid_in_reg) {
        extract_fields();

        // Check special cases (sign bit at position 22, 23-bit input)
        // Save special case results as 26-bit 1Q25 format and set special flag
        if (wE > 133 && (X_reg & 0x400000)) {  // Sign bit at bit 22 (changed from bit 21)
            // Negative number, large exponent -> exp(-large) -> approaches 0
            is_special_case_pipe[0] = true;
            special_result_pipe[0] = 0x0;  // 0 in 1Q25
            absInt_K = 0;
            valid_pipe[0] = true;
        } else if (wE > 133 && !(X_reg & 0x400000)) {
            // Positive number, large exponent -> exp(large) -> positive infinity (max value)
            is_special_case_pipe[0] = true;
            special_result_pipe[0] = 0x3FFFFFF;  // Max value in 1Q25
            absInt_K = 0;
            valid_pipe[0] = true;
        } else if (wE < 104 && (X_reg & 0x400000)) {
            // Negative number, small exponent (wE < 104) -> exp(-small) -> approaches 1
            is_special_case_pipe[0] = true;
            special_result_pipe[0] = 0x2000000;  // 1.0 in 1Q25 (2^25)
            absInt_K = 0;
            valid_pipe[0] = true;
        } else if (wE < 104 && !(X_reg & 0x400000)) {
            // Positive number, small exponent (wE < 104) -> exp(small) -> approaches 1
            is_special_case_pipe[0] = true;
            special_result_pipe[0] = 0x2000000;  // 1.0 in 1Q25 (2^25)
            absInt_K = 0;
            valid_pipe[0] = true;
        } else {
            // Normal calculation: multiply by 1/ln2 to get integer part K
            is_special_case_pipe[0] = false;
            // LUT indexing using 25-bit uxFix
            uint32_t t0_val = Exp_inv_ln2_LUT_t0[(uxFix >> 20) & 0x1F];  // uxFix[24:20]
            uint32_t t1_val = Exp_inv_ln2_LUT_t1[(uxFix >> 15) & 0x1F];  // uxFix[19:15]
            absInt_K = ((t0_val + t1_val) >> 4) & 0xFF;

            // Calculate signed K_value (used for 2^K multiplication in output stage)
            bool is_negative = (X_reg & 0x400000) != 0;  // Sign bit at position 22
            if (is_negative) {
                K_value = -(int16_t) (absInt_K & 0x7F);  // Negative K for negative input
            } else {
                K_value = (int16_t) (absInt_K & 0x7F);  // Positive K for positive input
            }

            valid_pipe[0] = true;
        }
    } else {
        valid_pipe[0] = false;
        is_special_case_pipe[0] = false;
    }
}

void Exp::stage1_K_calculation()
{
    if (valid_pipe[0]) {
        // Propagate special case flag
        is_special_case_pipe[1] = is_special_case_pipe[0];
        special_result_pipe[1] = special_result_pipe[0];

        // Pipeline K_value (stage 0 → pipeline stage 0)
        K_value_pipe[0] = K_value;

        // Calculate Exp_Y = uxFix[17:0] - K*ln2 (18-bit)
        uint32_t uxFix_low = uxFix & 0x3FFFF;  // 18-bit (upgraded from 14-bit)
        uint32_t ln2_t0 = Exp_ln2_LUT_t0[(absInt_K >> 3) & 0x1F] & 0x3FFFF;  // 18-bit
        uint32_t ln2_t1 = Exp_ln2_LUT_t1[absInt_K & 0x7] & 0x3FFFF;          // 18-bit
        uint32_t k_ln2 = (ln2_t0 + ln2_t1) & 0x3FFFF;                        // 18-bit

        if ((X_reg & 0x400000) == 0) {  // Sign bit at bit 22 (changed from bit 21)
            // Positive: X - K*ln2 = X + ~(K*ln2) + 1
            Exp_Y = (uxFix_low + (~k_ln2) + 1) & 0x3FFFF;  // 18-bit
        } else {
            // Negative: ~X + K*ln2 + 1
            Exp_Y = ((~uxFix_low) + k_ln2 + 1) & 0x3FFFF;  // 18-bit
        }

        valid_pipe[1] = true;
    } else {
        valid_pipe[1] = false;
        is_special_case_pipe[1] = false;
    }
}

void Exp::stage2_Y_calculation()
{
    if (valid_pipe[1]) {
        // Propagate special case flag
        is_special_case_pipe[2] = is_special_case_pipe[1];
        special_result_pipe[2] = special_result_pipe[1];

        // Pipeline K_value (pipeline stage 0 → 1)
        K_value_pipe[1] = K_value_pipe[0];

        // Special handling: if Exp_Y = 0 (input very close to 0), mark as special case
        // exp(0) = 1.0
        if (Exp_Y == 0 && !is_special_case_pipe[1]) {
            is_special_case_pipe[2] = true;
            special_result_pipe[2] = 0x2000000;  // 1.0 in 1Q25
        }

        // Set tree_data[0] according to Exp_Y bits (18-bit, from bit 17 to bit 0)
        for (int i = 0; i < EXP_Y_WIDTH; i++) {
            if ((Exp_Y >> i) & 1) {
                // Select corresponding exponent value based on bit index
                uint32_t exp_val = 0;
                switch (i) {
                    case 17:
                        exp_val = 0x4DA2CC;
                        break;
                    case 16:
                        exp_val = 0xA45AF2;
                        break;
                    case 15:
                        exp_val = 0x910B02;
                        break;
                    case 14:
                        exp_val = 0x88415B;
                        break;
                    case 13:
                        exp_val = 0x84102B;
                        break;
                    case 12:
                        exp_val = 0x820405;
                        break;
                    case 11:
                        exp_val = 0x810101;
                        break;
                    case 10:
                        exp_val = 0x808040;
                        break;
                    case 9:
                        exp_val = 0x804010;
                        break;
                    case 8:
                        exp_val = 0x802004;
                        break;
                    case 7:
                        exp_val = 0x801001;
                        break;
                    case 6:
                        exp_val = 0x800800;
                        break;
                    case 5:
                        exp_val = 0x800400;
                        break;
                    case 4:
                        exp_val = 0x800200;
                        break;
                    case 3:
                        exp_val = 0x800100;
                        break;
                    case 2:
                        exp_val = 0x800080;
                        break;
                    case 1:
                        exp_val = 0x800040;
                        break;
                    case 0:
                        exp_val = 0x800020;
                        break;
                    default:
                        exp_val = 0;
                        break;
                }
                tree_data[0][i] = (exp_val << Guard_Bits) & 0x3FFFFFF;
                tree_valid[0][i] = true;
            } else {
                tree_data[0][i] = 0;
                tree_valid[0][i] = false;
            }
        }
        valid_pipe[2] = true;
    } else {
        for (int i = 0; i < MAX_NODES; i++) {
            tree_valid[0][i] = false;
        }
        valid_pipe[2] = false;
        is_special_case_pipe[2] = false;
    }
}

// 1Q(23+G) fixed-point multiplication
uint32_t Exp::fixed_mult_1q23_guard(uint32_t a, uint32_t b)
{
    // 26-bit × 26-bit = 52-bit
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    // Extract bits [50:25] (FRAC_BITS=25, TOTAL_WIDTH=26)
    uint32_t result = (product >> 25) & 0x3FFFFFF;
    return result;
}

int Exp::calc_level_nodes(int level)
{
    if (level == 0) return EXP_Y_WIDTH;
    int prev = calc_level_nodes(level - 1);
    return (prev == 1) ? 1 : ((prev + 1) >> 1);
}

void Exp::stage4_7_tree_multiply()
{
    // Binary tree multiplication, 5-stage pipeline (TREE_DEPTH=5)
    for (int level = 0; level < TREE_DEPTH; level++) {
        if (valid_pipe[2 + level]) {
            // Propagate special case flag to next level
            is_special_case_pipe[2 + level + 1] = is_special_case_pipe[2 + level];
            special_result_pipe[2 + level + 1] = special_result_pipe[2 + level];

            // Pipeline K_value through each tree level (pipeline stage 1 → 2 → ... → 6)
            K_value_pipe[level + 2] = K_value_pipe[level + 1];

            int curr_nodes = calc_level_nodes(level);

            for (int i = 0; i < curr_nodes; i += 2) {
                int out_index = i >> 1;

                if (i + 1 < curr_nodes) {
                    if (tree_valid[level][i] && tree_valid[level][i + 1]) {
                        // Both data valid, multiply them
                        tree_data[level + 1][out_index] =
                            fixed_mult_1q23_guard(tree_data[level][i], tree_data[level][i + 1]);
                        tree_valid[level + 1][out_index] = true;
                    } else if (tree_valid[level][i]) {
                        // Only first one valid
                        tree_data[level + 1][out_index] = tree_data[level][i];
                        tree_valid[level + 1][out_index] = true;
                    } else if (tree_valid[level][i + 1]) {
                        // Only second one valid
                        tree_data[level + 1][out_index] = tree_data[level][i + 1];
                        tree_valid[level + 1][out_index] = true;
                    } else {
                        tree_valid[level + 1][out_index] = false;
                    }
                } else {
                    // Odd node, pass through directly
                    if (tree_valid[level][i]) {
                        tree_data[level + 1][out_index] = tree_data[level][i];
                        tree_valid[level + 1][out_index] = true;
                    } else {
                        tree_valid[level + 1][out_index] = false;
                    }
                }
            }
            valid_pipe[2 + level + 1] = true;
        } else {
            for (int i = 0; i < MAX_NODES; i++) {
                tree_valid[level + 1][i] = false;
            }
            valid_pipe[2 + level + 1] = false;
            is_special_case_pipe[2 + level + 1] = false;
        }
    }
}

void Exp::clock_step()
{
    // Stage 0: Input processing
    stage0_input();

    // Stage 1: K calculation
    stage1_K_calculation();

    // Stage 2: Y calculation and tree input
    stage2_Y_calculation();

    // Stage 3-6: Binary tree multiplication
    stage4_7_tree_multiply();

    // Output: Check if it's a special case
    int output_stage = 2 + TREE_DEPTH;
    if (is_special_case_pipe[output_stage]) {
        // Special case: use preset result directly
        valid_out = valid_pipe[output_stage];
        Exp_Ans = special_result_pipe[output_stage];
    } else {
        // Normal case: use tree calculation result
        valid_out = valid_pipe[output_stage] && tree_valid[TREE_DEPTH][0];
        Exp_Ans = tree_data[TREE_DEPTH][0];
    }
}

// Count leading zeros in a 26-bit value (for 1Q25 format)
int Exp::count_leading_zeros(uint32_t value) const
{
    if (value == 0) return 26;

    int count = 0;
    uint32_t mask = 1 << 25;  // Start from bit 25

    while ((value & mask) == 0 && count < 26) {
        count++;
        mask >>= 1;
    }

    return count;
}

// Convert 1Q25 fixed-point result to 25-bit floating-point format
// This applies K_final (2^K multiplication) through exponent adjustment
uint32_t Exp::get_Exp_Ans_extended() const
{
    // Get the fixed-point result (1Q25 format)
    uint32_t fixed_result = Exp_Ans;

    // Special case handling (matching hardware fast path)
    // Check for special values set by stage0_input
    if (fixed_result == 0x0) {
        // Zero: exn=00, sign=0, exp=0, mantissa=0
        return 0x0;
    } else if (fixed_result == 0x3FFFFFF) {
        // Positive infinity: exn=10, sign=0, exp=11111111, mantissa=all 0
        return (0x02 << 23) | (0xFF << 14) | 0x0000;
    } else if (fixed_result == 0x2000000) {
        // exp(Y) = 1.0, but need to apply K_final: exp(X) = 2^K * 1.0 = 2^K
        // Extract K_final from pipeline
        int16_t K_final = K_value_pipe[TREE_DEPTH + 1];
        int exponent_raw = 127 + K_final;

        // Clamp exponent to valid range
        if (exponent_raw <= 0) {
            return 0x0;  // Underflow to zero
        } else if (exponent_raw >= 255) {
            return (0x02 << 23) | (0xFF << 14) | 0x0000;  // Overflow to infinity
        }

        uint8_t exponent = (uint8_t) exponent_raw;
        // 2^K with mantissa = 0: exn=01, sign=0, exp=127+K, mantissa=0
        return (0x01 << 23) | (exponent << 14) | 0x0000;
    }

    // Extract K_final from pipeline (synchronized with Exp_Ans)
    int16_t K_final = K_value_pipe[TREE_DEPTH + 1];

    // Count leading zeros for normalization
    int leading_zeros = count_leading_zeros(fixed_result);

    // Normalize: shift left until MSB is 1
    uint32_t normalized_value = (fixed_result << leading_zeros) & 0x3FFFFFF;

    // Calculate exponent: K_final + BIAS - leading_zeros
    // BIAS = 127 for 8-bit exponent
    // leading_zeros represents how many bits to shift, which reduces the exponent
    int exponent_raw = K_final + 127 - leading_zeros;

    // Clamp exponent to valid range [0, 255]
    if (exponent_raw <= 0) {
        // Underflow: return zero (25-bit: exn=00, sign=0, exp=0, mantissa=0)
        return 0x0;
    } else if (exponent_raw >= 255) {
        // Overflow: return infinity (25-bit: exn=10, sign=0, exp=11111111, mantissa=0)
        return (0x02 << 23) | (0xFF << 14) | 0x0000;
    }

    uint8_t exponent = (uint8_t) exponent_raw;

    // Extract mantissa (top 14 bits after the leading 1)
    // normalized_value format: 1Q25 = 1.FFFFFFFFFFFFF (26 bits total, 25 fractional)
    // After normalization, bit 25 is 1, bits [24:0] are fractional
    // We need 14 bits of mantissa: bits [24:11]
    uint32_t mantissa = (normalized_value >> 11) & 0x3FFF;

    // Assemble 25-bit result: exn(2) + sign(1) + exp(8) + mantissa(14)
    // For normal numbers: exn=01, sign=0 (exp is always positive)
    uint32_t result = (0x01 << 23) | (exponent << 14) | mantissa;

    return result;
}
