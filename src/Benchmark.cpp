#include "Benchmark.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>

#include "Utils/FormatUtils.h"

void eltwise_add_reference(const std::vector<float> &x0, const std::vector<float> &x1,
                           std::vector<float> &y)
{
    int len = x0.size();
    y.resize(len);
    for (int i = 0; i < len; i++) {
        // Convert to bf16 first (simulating hardware input)
        uint16_t x0_bf16 = float_to_bf16(x0[i]);
        uint16_t x1_bf16 = float_to_bf16(x1[i]);
        float x0_val = bf16_to_float(x0_bf16);
        float x1_val = bf16_to_float(x1_bf16);

        // Compute with bf16 precision
        float result = x0_val + x1_val;
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}

void eltwise_mult_reference(const std::vector<float> &x0, const std::vector<float> &x1,
                            std::vector<float> &y)
{
    int len = x0.size();
    y.resize(len);
    for (int i = 0; i < len; i++) {
        // Convert to bf16 first (simulating hardware input)
        uint16_t x0_bf16 = float_to_bf16(x0[i]);
        uint16_t x1_bf16 = float_to_bf16(x1[i]);
        float x0_val = bf16_to_float(x0_bf16);
        float x1_val = bf16_to_float(x1_bf16);

        // Compute with bf16 precision
        float result = x0_val * x1_val;
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}

void sigmoid_reference(const std::vector<float> &x, std::vector<float> &y)
{
    int len = x.size();
    y.resize(len);
    for (int i = 0; i < len; i++) {
        // Convert to bf16 first (simulating hardware input)
        uint16_t x_bf16 = float_to_bf16(x[i]);
        float x_val = bf16_to_float(x_bf16);

        // Compute with bf16 precision and saturation protection (match hardware behavior)
        float result;
        if (x_val > 88.0f) {
            // σ(x) ≈ 1.0 (exp(-x) underflows to 0)
            result = 1.0f;
        } else if (x_val < -88.0f) {
            // σ(x) ≈ 0.0 (exp(-x) overflows)
            result = 0.0f;
        } else {
            result = 1.0f / (1.0f + std::exp(-x_val));
        }
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}

void silu_reference(const std::vector<float> &x, std::vector<float> &y)
{
    int len = x.size();
    y.resize(len);
    for (int i = 0; i < len; i++) {
        // Convert to bf16 first (simulating hardware input)
        uint16_t x_bf16 = float_to_bf16(x[i]);
        float x_val = bf16_to_float(x_bf16);

        // Handle special values
        if (std::isnan(x_val)) {
            y[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        if (std::isinf(x_val)) {
            if (x_val > 0) {
                // SiLU(+∞) = +∞
                y[i] = std::numeric_limits<float>::infinity();
            } else {
                // SiLU(-∞) = 0
                y[i] = 0.0f;
            }
            continue;
        }

        // Compute with bf16 precision
        float sigmoid = 1.0f / (1.0f + std::exp(-x_val));
        float result = x_val * sigmoid;
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}

void gelu_reference(const std::vector<float> &x, std::vector<float> &y)
{
    int len = x.size();
    y.resize(len);
    const float GELU_CONST = 1.702f;

    for (int i = 0; i < len; i++) {
        // Convert to bf16 first (simulating hardware input)
        uint16_t x_bf16 = float_to_bf16(x[i]);
        float x_val = bf16_to_float(x_bf16);

        // Handle special values
        if (std::isnan(x_val)) {
            y[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        if (std::isinf(x_val)) {
            if (x_val > 0) {
                // GELU(+∞) = +∞
                y[i] = std::numeric_limits<float>::infinity();
            } else {
                // GELU(-∞) = 0
                y[i] = 0.0f;
            }
            continue;
        }

        // Compute GELU(x) ≈ x · σ(1.702x) with bf16 precision
        float scaled_x = GELU_CONST * x_val;

        // Sigmoid with saturation check
        float sigmoid;
        if (scaled_x > 88.0f) {
            sigmoid = 1.0f;
        } else if (scaled_x < -88.0f) {
            sigmoid = 0.0f;
        } else {
            sigmoid = 1.0f / (1.0f + std::exp(-scaled_x));
        }

        float result = x_val * sigmoid;
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}

void rms_norm_reference(const std::vector<float> &x, std::vector<float> &y, float eps)
{
    int len = x.size();
    y.resize(len);

    // Convert all inputs to bf16 first (simulating hardware input)
    std::vector<float> x_bf16_vals(len);
    bool has_special = false;
    for (int i = 0; i < len; i++) {
        uint16_t x_bf16 = float_to_bf16(x[i]);
        x_bf16_vals[i] = bf16_to_float(x_bf16);
        if (std::isinf(x_bf16_vals[i]) || std::isnan(x_bf16_vals[i])) {
            has_special = true;
        }
    }

    // If any element is inf/NaN, return all NaN
    if (has_special) {
        for (int i = 0; i < len; i++) {
            y[i] = std::numeric_limits<float>::quiet_NaN();
        }
        return;
    }

    float sum_sq = 0.0f;
    for (int i = 0; i < len; i++) {
        sum_sq += x_bf16_vals[i] * x_bf16_vals[i];
    }
    float mean_sq = sum_sq / len;
    // Use 1/sqrt instead of sqrt+div to match hardware implementation
    float rms_inv = 1.0f / std::sqrt(mean_sq + eps);

    for (int i = 0; i < len; i++) {
        float result = x_bf16_vals[i] * rms_inv;  // Multiply instead of divide
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}

void layer_norm_reference(const std::vector<float> &x, std::vector<float> &y, float eps)
{
    int len = x.size();
    y.resize(len);

    // Convert all inputs to bf16 first (simulating hardware input)
    std::vector<float> x_bf16_vals(len);
    bool has_special = false;
    for (int i = 0; i < len; i++) {
        uint16_t x_bf16 = float_to_bf16(x[i]);
        x_bf16_vals[i] = bf16_to_float(x_bf16);
        if (std::isinf(x_bf16_vals[i]) || std::isnan(x_bf16_vals[i])) {
            has_special = true;
        }
    }

    // If any element is inf/NaN, return all NaN
    if (has_special) {
        for (int i = 0; i < len; i++) {
            y[i] = std::numeric_limits<float>::quiet_NaN();
        }
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += x_bf16_vals[i];
    }
    float mean = sum / len;

    float var_sum = 0.0f;
    for (int i = 0; i < len; i++) {
        float diff = x_bf16_vals[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / len;
    // Use 1/sqrt instead of sqrt+div to match hardware implementation
    float std_inv = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < len; i++) {
        float result = (x_bf16_vals[i] - mean) * std_inv;  // Multiply instead of divide
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}

void online_softmax_reference(const std::vector<float> &x, std::vector<float> &y)
{
    int len = x.size();
    y.resize(len);

    // Convert all inputs to bf16 first (simulating hardware input)
    std::vector<float> x_bf16_vals(len);
    for (int i = 0; i < len; i++) {
        uint16_t x_bf16 = float_to_bf16(x[i]);
        x_bf16_vals[i] = bf16_to_float(x_bf16);
    }

    // ============================================================
    // Robust Softmax: Handle special values (NaN, +inf)
    // ============================================================

    // 1. Check for NaN
    for (int i = 0; i < len; i++) {
        if (std::isnan(x_bf16_vals[i])) {
            // Return all NaN (NaN propagation)
            uint16_t nan_bf16 = float_to_bf16(std::numeric_limits<float>::quiet_NaN());
            float nan_val = bf16_to_float(nan_bf16);
            for (int j = 0; j < len; j++) {
                y[j] = nan_val;
            }
            return;
        }
    }

    // 2. Check for +inf and count
    int num_pos_inf = 0;
    std::vector<int> pos_inf_indices;
    for (int i = 0; i < len; i++) {
        if (std::isinf(x_bf16_vals[i]) && x_bf16_vals[i] > 0) {
            num_pos_inf++;
            pos_inf_indices.push_back(i);
        }
    }

    // If there are +inf values, distribute probability equally among them
    if (num_pos_inf > 0) {
        float prob = 1.0f / num_pos_inf;
        uint16_t prob_bf16 = float_to_bf16(prob);
        float prob_val = bf16_to_float(prob_bf16);
        uint16_t zero_bf16 = float_to_bf16(0.0f);
        float zero_val = bf16_to_float(zero_bf16);

        for (int i = 0; i < len; i++) {
            y[i] = zero_val;
        }
        for (int idx : pos_inf_indices) {
            y[idx] = prob_val;
        }
        return;
    }

    // 3. Standard softmax for finite values (including -inf)
    // Find max among finite values only
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < len; i++) {
        if (std::isfinite(x_bf16_vals[i]) && x_bf16_vals[i] > max_val) {
            max_val = x_bf16_vals[i];
        }
    }

    // If all values are -inf, return NaN
    if (std::isinf(max_val) && max_val < 0) {
        uint16_t nan_bf16 = float_to_bf16(std::numeric_limits<float>::quiet_NaN());
        float nan_val = bf16_to_float(nan_bf16);
        for (int i = 0; i < len; i++) {
            y[i] = nan_val;
        }
        return;
    }

    // Compute exp and sum
    float sum = 0.0f;
    std::vector<float> exp_x(len);
    for (int i = 0; i < len; i++) {
        exp_x[i] = std::exp(x_bf16_vals[i] - max_val);  // exp(-inf) = 0 automatically
        sum += exp_x[i];
    }

    // Normalize
    for (int i = 0; i < len; i++) {
        float result = exp_x[i] / sum;
        uint16_t result_bf16 = float_to_bf16(result);
        y[i] = bf16_to_float(result_bf16);
    }
}
#include "Utils/Logger.h"
#include <iomanip>

void view_f32_rows(const std::vector<float> &data, int rows, int column,
                   int hex_first = 16, int f32_first = 16) {
  LOG("size: " << data.size() << " rows: " << rows << " column: " << column);
  for (int i = 0; i < rows; i++) {
    std::cout << "== Row " << std::dec << i << " ==\n";
    std::cout << "HEX (first " << hex_first << "):";
    for (int j = 0; j < hex_first; j++) {
      std::cout << std::hex
                << reinterpret_cast<const uint32_t *>(&data[i * column + j])[0]
                << " ";
    }
    std::cout << "\n F32 (first " << f32_first << "):";
    for (int j = 0; j < f32_first; j++) {
      std::cout << std::setw(6) << data[i * column + j] << " ";
    }
    std::cout << "\n\n";
  }
}
void view_u16_rows(const std::vector<uint16_t> &data, int rows, int column,
                   int hex_first = 16) {
  for (int i = 0; i < rows; i++) {
    std::cout << "== Row " << std::dec << i << " ==\n";
    std::cout << "HEX (first " << hex_first << "):";
    for (int j = 0; j < hex_first; j++) {
      std::cout << std::hex << data[i * column + j] << " ";
    }
    std::cout << "\n\n";
  }
}

struct error_pos {
    float err;
    int idx;
};
bool cmp(error_pos a, error_pos b) {
    return a.err > b.err;
}
BenchmarkResult verify_accuracy(const std::vector<float> &y_custom,
                                const std::vector<float> &y_reference,
                                const std::vector<int> test_rows, int D, float threshold)
{
    BenchmarkResult result;
    result.max_error = 0.0f;
    result.avg_error = 0.0f;

    int len = y_custom.size();
    if (len != y_reference.size() || len == 0) {
        result.passed = false;
        return result;
    }

    float sum_error = 0.0f;
    int valid_count = 0, max_error_index = 0, errors_cnt = 0;  // Count of non-NaN/Inf comparisons
    std::vector<error_pos> errors;

    for (int i = 0; i < len; i++) {
        float custom_val = y_custom[i];
        float ref_val = y_reference[i];
        float error = 0.0f;

        // Special value handling
        if (std::isnan(ref_val)) {
            // If reference is NaN, custom should also be NaN
            if (std::isnan(custom_val)) {
                error = 0.0f;  // Both NaN: correct
            } else {
                error = std::numeric_limits<float>::infinity();  // Mismatch: error
            }
        } else if (std::isinf(ref_val)) {
            // If reference is Inf, custom should also be Inf with same sign
            if (std::isinf(custom_val) && std::signbit(custom_val) == std::signbit(ref_val)) {
                error = 0.0f;  // Both Inf with same sign: correct
            } else {
                error = std::numeric_limits<float>::infinity();  // Mismatch: error
            }
        } else if (std::isnan(custom_val) || std::isinf(custom_val)) {
            // Reference is finite, but custom is NaN/Inf: error
            error = std::numeric_limits<float>::infinity();
        } else {
            // Both are finite: compute normal error
            error = std::abs(custom_val - ref_val);
            valid_count++;
        }

        // Only track finite errors in statistics
        if (std::isfinite(error)) {
            sum_error += error;
            if (error > result.max_error) {
                result.max_error = error;
                max_error_index = i;
            }
            // Debug: print large errors (threshold: 1.0 for meaningful debugging)
            if (error > threshold) {
                errors.push_back(error_pos{error, i});
                errors_cnt += 1;
            }
        } else if (error == std::numeric_limits<float>::infinity()) {
            // Special value mismatch detected - fail the test
            result.passed = false;
        }
    }

    // Compute average error only for valid (finite) comparisons
    if (valid_count > 0) {
        result.avg_error = sum_error / valid_count;
    } else {
        result.avg_error = std::numeric_limits<float>::quiet_NaN();
    }

    result.passed = (result.max_error < threshold);
    std::sort(errors.begin(), errors.end(), cmp);
    
    // Print the max 10 errors positions
    std::cout << "Max 10 Errors: " << std::endl;
    for (int i = 0; i < 10; i++) {
        int max_error_index = errors[i].idx;
        int row_num = max_error_index / D, col_num = max_error_index - row_num * D;
        std::cout << "Error at row " << test_rows[row_num] << " column " << col_num << " get "
                  << y_custom[max_error_index] << " expect " << y_reference[max_error_index]
                  << " error " << errors[i].err
                  << std::endl;
    }
    return result;
}
