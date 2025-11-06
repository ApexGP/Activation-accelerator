#ifndef INPUT_VEC_H
#define INPUT_VEC_H

#include <random>
#include <vector>

// Forward declaration of module config constants
// (defined in Utils/Config.h, but avoiding circular dependency)

/**
 * @brief Input Vector Generator for Activation Function Benchmark
 *
 * This module generates comprehensive test vectors following InputVecTutorial.md strategy.
 * The test vectors are designed to cover:
 * - Basic BF16 properties and special values (Rows 0-11)
 * - Random distributions (Rows 12-19)
 * - Function-specific tests (Rows 20-53)
 * - Additional complex patterns (Rows 54-63)
 */
class InputVecGenerator
{
public:
    /**
     * @brief Construct a new Input Vec Generator object
     *
     * @param length Length of test vectors (N, number of rows, default: 64)
     * @param dimension Dimension of each test vector (D, number of columns, default: 1)
     * @param seed Random seed for reproducibility (default: 2024)
     */
    InputVecGenerator(int length = 64, int dimension = 768, unsigned int seed = 42);

    /**
     * @brief Generate test vectors following InputVecTutorial.md strategy
     *
     * Generates two test vectors (x and x1) with comprehensive test cases:
     * - Rows 0-11: Basic BF16 properties and special values
     * - Rows 12-19: Random distributions
     * - Rows 20-25: Softmax specific tests
     * - Rows 26-31: LayerNorm specific tests
     * - Rows 32-37: RMSNorm specific tests
     * - Rows 38-41: SiLU specific tests
     * - Rows 42-45: GELU placeholder tests
     * - Rows 46-53: EltwiseAdd/Mul specific tests
     * - Rows 54-63: Additional random and complex patterns
     *
     * @param x_output Output vector for first input (x)
     * @param x1_output Output vector for second input (x1)
     */
    void generate(std::vector<float> &x_output, std::vector<float> &x1_output);

    /**
     * @brief Get the test vector length (number of rows)
     *
     * @return int Number of rows (N)
     */
    int get_length() const
    {
        return length_;
    }

    /**
     * @brief Get the dimension of each test vector (number of columns)
     *
     * @return int Number of columns (D)
     */
    int get_dimension() const
    {
        return dimension_;
    }

    /**
     * @brief Get total number of elements (N * D)
     *
     * @return int Total elements
     */
    int get_total_size() const
    {
        return length_ * dimension_;
    }

private:
    int length_;         // Number of rows (N)
    int dimension_;      // Number of columns (D)
    std::mt19937 gen_;   // Random number generator
    unsigned int seed_;  // Random seed

    // BF16 special values and constants
    static constexpr float BF16_MAX = 3.38953e+38f;
    static constexpr float BF16_MIN_NORMAL = 1.17549e-38f;
    const float POS_INF;
    const float NEG_INF;
    const float NAN_VAL;

    /**
     * @brief Round float to 1 decimal place
     *
     * @param x Input value
     * @return float Rounded value (or original if Inf/NaN)
     */
    float round_1dp(float x) const;

    /**
     * @brief Generate test data for rows 0-11 (Basic BF16 properties)
     *
     * @param row Row index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_basic_bf16_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 12-19 (Random distributions)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_random_distribution_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 20-25 (Softmax specific)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_softmax_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 26-31 (LayerNorm specific)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_layernorm_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 32-37 (RMSNorm specific)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_rmsnorm_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 38-41 (SiLU specific)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_silu_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 42-45 (GELU specific)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_gelu_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 46-53 (EltwiseAdd/Mul specific)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_eltwise_tests(int row, int col, float &val_x, float &val_x1);

    /**
     * @brief Generate test data for rows 54-63 (Random complex patterns)
     *
     * @param row Row index
     * @param col Column index
     * @param val_x Output for x value
     * @param val_x1 Output for x1 value
     */
    void generate_random_tests(int row, int col, float &val_x, float &val_x1);
};

/**
 * @brief Select test row indices for specific activation function
 *
 * According to InputVecTutorial.md:
 * - Rows 0-19: General tests (all functions)
 * - Rows 20-25: Softmax specific
 * - Rows 26-31: LayerNorm specific
 * - Rows 32-37: RMSNorm specific
 * - Rows 38-41: SiLU specific
 * - Rows 42-45: GELU specific (not implemented)
 * - Rows 46-53: EltwiseAdd/Mul specific
 * - Rows 54-63: General random tests (all functions)
 *
 * @param module_config Module configuration constant (from Config.h)
 * @return vector of selected row indices
 */
std::vector<int> select_test_rows(int module_config);

/**
 * @brief Extract subset of data based on selected row indices
 *
 * For a N×D matrix stored in row-major order, this function extracts
 * the complete rows specified by row_indices.
 *
 * @param full_data Complete N×D matrix (row-major, flattened)
 * @param row_indices Selected row indices
 * @param dimension Number of columns (D)
 * @return Subset of data corresponding to selected rows (length = row_indices.size() × D)
 */
template <typename T>
std::vector<T> extract_test_data(const std::vector<T> &full_data,
                                 const std::vector<int> &row_indices, int dimension)
{
    std::vector<T> subset;
    subset.reserve(row_indices.size() * dimension);
    for (int row : row_indices) {
        int row_offset = row * dimension;
        for (int col = 0; col < dimension; col++) {
            int idx = row_offset + col;
            if (idx >= 0 && idx < static_cast<int>(full_data.size())) {
                subset.push_back(full_data[idx]);
            }
        }
    }
    return subset;
}

#endif  // INPUT_VEC_H
