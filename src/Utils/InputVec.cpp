#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>

#define __STATIC_INPUT__

#include "Utils/Config.h"
#include "Utils/InputVec.h"

InputVecGenerator::InputVecGenerator(int length, int dimension, unsigned int seed)
    : length_(length),
      dimension_(dimension),
      gen_(seed),
      seed_(seed),
      POS_INF(std::numeric_limits<float>::infinity()),
      NEG_INF(-std::numeric_limits<float>::infinity()),
      NAN_VAL(std::numeric_limits<float>::quiet_NaN())
{
}

float InputVecGenerator::round_1dp(float x) const
{
    if (std::isinf(x) || std::isnan(x)) return x;
    return std::round(x * 10.0f) / 10.0f;
}

void InputVecGenerator::generate(std::vector<float> &x_output, std::vector<float> &x1_output)
{
    // Resize outputs to N * D (row-major flattened)
    int total_size = length_ * dimension_;
    x_output.resize(total_size);
    x1_output.resize(total_size);

    // Generate N rows of test data
    for (int row = 0; row < length_; row++) {
        int row_offset = row * dimension_;

        // Generate each column value for this row
        for (int col = 0; col < dimension_; col++) {
            float val_x = 0.0f, val_x1 = 0.0f;

            // Generate test values for this row and column according to InputVecTutorial.md
            if (row < 12) {
                generate_basic_bf16_tests(row, col, val_x, val_x1);
            } else if (row < 20) {
                generate_random_distribution_tests(row, col, val_x, val_x1);
            } else if (row < 26) {
                generate_softmax_tests(row, col, val_x, val_x1);
            } else if (row < 32) {
                generate_layernorm_tests(row, col, val_x, val_x1);
            } else if (row < 38) {
                generate_rmsnorm_tests(row, col, val_x, val_x1);
            } else if (row < 42) {
                generate_silu_tests(row, col, val_x, val_x1);
            } else if (row < 46) {
                generate_gelu_tests(row, col, val_x, val_x1);
            } else if (row < 54) {
                generate_eltwise_tests(row, col, val_x, val_x1);
            } else {
                generate_random_tests(row, col, val_x, val_x1);
            }

            // Round and store (skip rounding for rows that need precise values)
            bool needs_precision = (row == 27 || row == 29 || row == 33 || row == 35);
            x_output[row_offset + col] = needs_precision ? val_x : round_1dp(val_x);
            x1_output[row_offset + col] = needs_precision ? val_x1 : round_1dp(val_x1);
        }
    }
}

void InputVecGenerator::generate_basic_bf16_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 0-11: Basic BF16 properties and common values
    switch (row) {
        case 0:
            val_x = 0.0f;
            val_x1 = 0.0f;
            break;  // All zeros
        case 1:
            val_x = 1.0f;
            val_x1 = 1.0f;
            break;  // All ones
        case 2:
            val_x = -1.0f;
            val_x1 = -1.0f;
            break;  // All negative ones
        case 3:
            val_x = BF16_MAX;
            val_x1 = BF16_MAX / 2.0f;
            break;  // BF16_MAX
        case 4:
            val_x = -BF16_MAX;
            val_x1 = -BF16_MAX / 2.0f;
            break;  // -BF16_MAX
        case 5:
            val_x = BF16_MIN_NORMAL;
            val_x1 = BF16_MIN_NORMAL;
            break;  // BF16_MIN_NORMAL
        case 6:
            val_x = -BF16_MIN_NORMAL;
            val_x1 = -BF16_MIN_NORMAL;
            break;  // -BF16_MIN_NORMAL
        case 7:
            val_x = 1e-30f;
            val_x1 = 1e-30f;
            break;  // Positive subnormal
        case 8:
            val_x = -1e-30f;
            val_x1 = -1e-30f;
            break;  // Negative subnormal
        case 9: {  // Mixed normal values: [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 10.0, -10.0] repeating
            const float pattern[] = {1.0f,  -1.0f, 0.5f,  -0.5f, 2.0f,
                                     -2.0f, 0.1f,  -0.1f, 10.0f, -10.0f};
            val_x = pattern[col % 10];
            val_x1 = pattern[col % 10];
            break;
        }
        case 10: {  // Mixed special values: [+Inf, -Inf, NaN, +0.0, -0.0, 1.0, -1.0] repeating
            int idx = col % 7;
            switch (idx) {
                case 0:
                    val_x = POS_INF;
                    break;
                case 1:
                    val_x = NEG_INF;
                    break;
                case 2:
                    val_x = NAN_VAL;
                    break;
                case 3:
                    val_x = 0.0f;
                    break;
                case 4:
                    val_x = -0.0f;
                    break;
                case 5:
                    val_x = 1.0f;
                    break;
                case 6:
                    val_x = -1.0f;
                    break;
            }
            val_x1 = val_x;  // Same pattern for x1
            break;
        }
        case 11:
            val_x = (col % 2 == 0) ? 0.0f : -0.0f;
            val_x1 = (col % 2 == 0) ? 0.0f : -0.0f;
            break;  // Alternating +0.0 and -0.0 across columns
    }
}

void InputVecGenerator::generate_random_distribution_tests(int row, int col, float &val_x,
                                                           float &val_x1)
{
    // Rows 12-19: Random distribution tests
    // Calculate injection positions: inject at specific positions only
    // Inject at positions [0,1] and [D/2, D/2+1] to ensure some coverage without overwhelming
    bool should_inject_17_18 = (col <= 1) || (col >= dimension_ / 2 && col <= dimension_ / 2 + 1);

    std::uniform_real_distribution<float> dis;
    switch (row) {
        case 12:
            dis = std::uniform_real_distribution<float>(-10.0f, 10.0f);
            break;
        case 13:
            dis = std::uniform_real_distribution<float>(-1000.0f, 1000.0f);
            break;
        case 14:
            dis = std::uniform_real_distribution<float>(-0.1f, 0.1f);
            break;
        case 15:
            dis = std::uniform_real_distribution<float>(0.0f, 1.0f);
            break;
        case 16: {  // Normal distribution (mean=0, std=5)
            std::normal_distribution<float> ndis(0.0f, 5.0f);
            val_x = ndis(gen_);
            val_x1 = ndis(gen_);
            return;
        }
        case 17: {  // Random with Inf/NaN injection
            dis = std::uniform_real_distribution<float>(-5.0f, 5.0f);
            // Inject +Inf and NaN at specific column positions
            val_x = should_inject_17_18 ? ((col % 2 == 0) ? POS_INF : NAN_VAL) : dis(gen_);
            val_x1 = dis(gen_);
            return;
        }
        case 18: {  // Random with subnormal injection
            dis = std::uniform_real_distribution<float>(-5.0f, 5.0f);
            // Inject subnormal at specific column positions
            val_x = should_inject_17_18 ? 1e-30f : dis(gen_);
            val_x1 = dis(gen_);
            return;
        }
        case 19:
            dis = std::uniform_real_distribution<float>(0.0f, BF16_MAX / 2.0f);
            break;
    }
    val_x = dis(gen_);
    val_x1 = dis(gen_);
}

void InputVecGenerator::generate_softmax_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 20-25: Softmax specific tests (per InputVecTutorial.md)
    // Softmax is a single-input function, so x1 should match x
    switch (row) {
        case 20: {  // Large positive increasing sequence: [100.0, 100.1, 100.2, ...]
            float t = col / float(dimension_ - 1);
            val_x = 100.0f + t * 100.0f;  // From 100.0 to 200.0
            break;
        }
        case 21: {  // Large negative decreasing sequence: [-100.0, -100.1, -100.2, ...]
            float t = col / float(dimension_ - 1);
            val_x = -100.0f - t * 100.0f;  // From -100.0 to -200.0
            break;
        }
        case 22:
            val_x = (col == dimension_ / 2) ? 10.0f : 0.0f;
            break;  // Single element much larger
        case 23:
            val_x = (col == dimension_ / 2) ? POS_INF : 0.0f;
            break;  // Single +Inf element
        case 24:
            val_x = ((col == dimension_ / 3) || (col == 2 * dimension_ / 3)) ? POS_INF : 0.0f;
            break;  // Two +Inf elements
        case 25:
            val_x = (col == dimension_ / 2) ? NAN_VAL : 0.0f;
            break;  // Single NaN element
    }
    val_x1 = val_x;  // Softmax is single-input, x1 mirrors x
}

void InputVecGenerator::generate_layernorm_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 26-31: LayerNorm specific tests (per InputVecTutorial.md)
    // LayerNorm is a single-input function, so x1 should match x
    // Inject at specific positions: [0,1,2] and [D/2, D/2+1, D/2+2]
    bool should_inject_28 = (col <= 2) || (col >= dimension_ / 2 && col <= dimension_ / 2 + 2);

    switch (row) {
        case 26:
            val_x = ((col % 2) == 0) ? 1.0f : -1.0f;
            break;  // Zero mean, non-zero variance: alternating [1.0, -1.0, 1.0, -1.0, ...]
        case 27:
            val_x = 5.0f + col * 0.001f;
            break;  // Near constant (small variance): [5.000, 5.001, 5.002, ...]
        case 28: {  // Contains Inf/NaN
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
            val_x = should_inject_28 ? ((col % 3 == 0)   ? POS_INF
                                        : (col % 3 == 1) ? NEG_INF
                                                         : NAN_VAL)
                                     : dis(gen_);
            break;
        }
        case 29: {  // All values differ by epsilon: [0.0, EPSILON_LN, 2*EPSILON_LN, ...]
            const float EPSILON_LN = 1e-5f;
            val_x = col * EPSILON_LN;
            break;
        }
        case 30: {  // Large values
            std::uniform_real_distribution<float> dis(0.0f, BF16_MAX / 10.0f);
            val_x = dis(gen_);
            break;
        }
        case 31: {  // Small values (non-subnormal)
            std::uniform_real_distribution<float> dis(BF16_MIN_NORMAL, BF16_MIN_NORMAL * 10.0f);
            val_x = dis(gen_);
            break;
        }
    }
    val_x1 = val_x;  // LayerNorm is single-input, x1 mirrors x
}

void InputVecGenerator::generate_rmsnorm_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 32-37: RMSNorm specific tests (per InputVecTutorial.md)
    // RMSNorm is a single-input function, so x1 should match x
    // Inject at specific positions: [0,1,2] and [D/2, D/2+1, D/2+2]
    bool should_inject_34 = (col <= 2) || (col >= dimension_ / 2 && col <= dimension_ / 2 + 2);

    switch (row) {
        case 32:
            val_x = ((col % 2) == 0) ? 1.0f : -1.0f;
            break;  // Zero mean, non-zero RMS: alternating [1.0, -1.0, 1.0, -1.0, ...]
        case 33:
            val_x = col * 0.001f;
            break;  // Near zero (small RMS): [0.000, 0.001, 0.002, ...]
        case 34: {  // Contains Inf/NaN
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
            val_x = should_inject_34 ? ((col % 3 == 0)   ? POS_INF
                                        : (col % 3 == 1) ? NEG_INF
                                                         : NAN_VAL)
                                     : dis(gen_);
            break;
        }
        case 35: {  // All values differ by epsilon (from zero): [0.0, EPSILON_RMS, 2*EPSILON_RMS, ...]
            const float EPSILON_RMS = 1e-5f;
            val_x = col * EPSILON_RMS;
            break;
        }
        case 36: {  // Large values
            std::uniform_real_distribution<float> dis(0.0f, BF16_MAX / 10.0f);
            val_x = dis(gen_);
            break;
        }
        case 37: {  // Small values (non-subnormal)
            std::uniform_real_distribution<float> dis(BF16_MIN_NORMAL, BF16_MIN_NORMAL * 10.0f);
            val_x = dis(gen_);
            break;
        }
    }
    val_x1 = val_x;  // RMSNorm is single-input, x1 mirrors x
}

void InputVecGenerator::generate_silu_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 38-41: SiLU specific tests (per InputVecTutorial.md)
    // SiLU is a single-input function, so x1 should match x
    // Inject at specific positions: [0,1,2,3] and [D/2, D/2+1, D/2+2, D/2+3]
    bool should_inject_41 = (col <= 3) || (col >= dimension_ / 2 && col <= dimension_ / 2 + 3);

    switch (row) {
        case 38: {  // Dense sampling around zero: [-5.0, ..., 0.0, ..., 5.0]
            float t = col / float(dimension_ - 1);
            val_x = -5.0f + t * 10.0f;  // Linear from -5 to 5
            break;
        }
        case 39: {  // Large positive values: [10.0, ..., 100.0]
            float t = col / float(dimension_ - 1);
            val_x = 10.0f + t * 90.0f;  // Linear from 10 to 100
            break;
        }
        case 40: {  // Large negative values: [-10.0, ..., -100.0]
            float t = col / float(dimension_ - 1);
            val_x = -10.0f - t * 90.0f;  // Linear from -10 to -100
            break;
        }
        case 41: {  // Contains Inf/-Inf/NaN
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
            val_x = should_inject_41 ? ((col % 4 == 0)   ? POS_INF
                                        : (col % 4 == 1) ? NEG_INF
                                        : (col % 4 == 2) ? NAN_VAL
                                                         : dis(gen_))
                                     : dis(gen_);
            break;
        }
    }
    val_x1 = val_x;  // SiLU is single-input, x1 mirrors x
}

void InputVecGenerator::generate_gelu_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 42-45: GELU specific tests (per InputVecTutorial.md)
    // GELU is a single-input function, so x1 should match x
    // Inject at specific positions: [0,1,2,3] and [D/2, D/2+1, D/2+2, D/2+3]
    bool should_inject_special = (col <= 3) || (col >= dimension_ / 2 && col <= dimension_ / 2 + 3);

    switch (row) {
        case 42: {  // Dense sampling around zero: [-5.0, ..., 0.0, ..., 5.0]
            float t = col / float(dimension_ - 1);
            val_x = -5.0f + t * 10.0f;  // Linear from -5 to 5
            break;
        }
        case 43: {  // Large positive values: [10.0, ..., 100.0]
            float t = col / float(dimension_ - 1);
            val_x = 10.0f + t * 90.0f;  // Linear from 10 to 100
            break;
        }
        case 44: {  // Large negative values: [-10.0, ..., -100.0]
            float t = col / float(dimension_ - 1);
            val_x = -10.0f - t * 90.0f;  // Linear from -10 to -100
            break;
        }
        case 45: {  // Contains Inf/-Inf/NaN
            std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
            val_x = should_inject_special ? ((col % 4 == 0)   ? POS_INF
                                             : (col % 4 == 1) ? NEG_INF
                                             : (col % 4 == 2) ? NAN_VAL
                                                              : dis(gen_))
                                          : dis(gen_);
            break;
        }
    }
    val_x1 = val_x;  // GELU is single-input, x1 mirrors x
}

void InputVecGenerator::generate_eltwise_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 46-53: EltwiseAdd/Mul specific tests (per InputVecTutorial.md)
    // X as first operand, X1 as second operand
    // For Row 50-53, inject only at [0,1]
    bool should_inject_50_53 = (col <= 1);

    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

    switch (row) {
        case 46: {  // Random positive
            std::uniform_real_distribution<float> pos_dis(0.0f, 10.0f);
            val_x = pos_dis(gen_);
            val_x1 = pos_dis(gen_);
            break;
        }
        case 47: {  // Random negative
            std::uniform_real_distribution<float> neg_dis(-10.0f, 0.0f);
            val_x = neg_dis(gen_);
            val_x1 = neg_dis(gen_);
            break;
        }
        case 48: {  // Large values (test overflow)
            std::uniform_real_distribution<float> large_dis(BF16_MAX / 2.0f, BF16_MAX);
            val_x = large_dis(gen_);
            val_x1 = large_dis(gen_);
            break;
        }
        case 49: {  // Small values (test underflow)
            std::uniform_real_distribution<float> small_dis(BF16_MIN_NORMAL,
                                                            BF16_MIN_NORMAL * 10.0f);
            val_x = small_dis(gen_);
            val_x1 = small_dis(gen_);
            break;
        }
        case 50:  // Contains +Inf (test Inf + (-Inf) or 0 * Inf)
            // Inject +Inf at specific positions (first 2 columns)
            val_x = should_inject_50_53 ? POS_INF : dis(gen_);
            val_x1 = should_inject_50_53 ? NEG_INF : dis(gen_);
            break;
        case 51:  // Contains -Inf
            // Inject -Inf at specific positions (first 2 columns)
            val_x = should_inject_50_53 ? NEG_INF : dis(gen_);
            val_x1 = dis(gen_);
            break;
        case 52:  // Contains NaN (test NaN propagation)
            // Inject NaN at specific positions (first 2 columns)
            val_x = should_inject_50_53 ? NAN_VAL : dis(gen_);
            val_x1 = dis(gen_);
            break;
        case 53:  // Contains 0.0 (test 0 * Inf)
            // Inject 0.0 at specific positions (first 2 columns)
            val_x = should_inject_50_53 ? 0.0f : dis(gen_);
            val_x1 = dis(gen_);
            break;
    }
}

void InputVecGenerator::generate_random_tests(int row, int col, float &val_x, float &val_x1)
{
    // Rows 54-63: Additional random and complex patterns (per InputVecTutorial.md)
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

    val_x = dis(gen_);
    val_x1 = dis(gen_);

    // Add some variety for the remaining rows
    if (row % 3 == 0) {
        std::uniform_real_distribution<float> wide_dis(-100.0f, 100.0f);
        val_x = wide_dis(gen_);
    } else if (row % 3 == 1) {
        // Alternating large and small
        val_x = ((row % 2) == 0) ? 100.0f : 0.01f;
    }
    // row % 3 == 2 keeps default dis(-5, 5)
}

// Standalone helper functions for test row selection
std::vector<int> select_test_rows(int module_config)
{
    std::vector<int> rows;
    // Add general test rows (0-19)
    for (int i = 0; i < 20; i++) {
        rows.push_back(i);
    }

    // Add function-specific rows (20-53)
    switch (module_config) {
        case CONFIG_ONLINE_SOFTMAX:
            // Rows 20-25: Softmax specific
            for (int i = 20; i <= 25; i++) rows.push_back(i);
            break;
        case CONFIG_LAYER_NORM:
            // Rows 26-31: LayerNorm specific
            for (int i = 26; i <= 31; i++) rows.push_back(i);
            break;
        case CONFIG_RMS_NORM:
            // Rows 32-37: RMSNorm specific
            for (int i = 32; i <= 37; i++) rows.push_back(i);
            break;
        case CONFIG_SILU:
            // Rows 38-41: SiLU specific
            for (int i = 38; i <= 41; i++) rows.push_back(i);
            break;
        case CONFIG_SIGMOID:
            // Sigmoid can use SiLU test rows (similar behavior)
            for (int i = 38; i <= 41; i++) rows.push_back(i);
            break;
        case CONFIG_GELU:
            // Rows 42-45: GELU specific
            for (int i = 42; i <= 45; i++) rows.push_back(i);
            break;
        case CONFIG_ELTWISE_ADD:
        case CONFIG_ELTWISE_MULT:
            // Rows 46-53: EltwiseAdd/Mul specific
            for (int i = 46; i <= 53; i++) rows.push_back(i);
            break;
        default:
            // No specific rows, use general only
            break;
    }

    // Add general random test rows (54-63)
    for (int i = 54; i < 64; i++) {
        rows.push_back(i);
    }
    return rows;
}
