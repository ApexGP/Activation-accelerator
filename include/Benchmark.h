#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstdint>
#include <vector>

// Reference implementations for accuracy verification

void eltwise_add_reference(const std::vector<float> &x0, const std::vector<float> &x1,
                           std::vector<float> &y);
void eltwise_mult_reference(const std::vector<float> &x0, const std::vector<float> &x1,
                            std::vector<float> &y);
void sigmoid_reference(const std::vector<float> &x, std::vector<float> &y);
void silu_reference(const std::vector<float> &x, std::vector<float> &y);
void gelu_reference(const std::vector<float> &x, std::vector<float> &y);
void rms_norm_reference(const std::vector<float> &x, std::vector<float> &y, float eps = 1e-6f);
void layer_norm_reference(const std::vector<float> &x, std::vector<float> &y, float eps = 1e-6f);
void online_softmax_reference(const std::vector<float> &x, std::vector<float> &y);

struct BenchmarkResult {
    float max_error;
    float avg_error;
    bool passed;  // Pass if error < threshold
};

// BF16 quantization threshold: 3.91e-3 (difference between 0x3F7F and 0x3F80)
// This accounts for the quantization gap when float32 results are truncated to bf16
BenchmarkResult verify_accuracy(const std::vector<float> &y_custom,
                                const std::vector<float> &y_reference,
                                const std::vector<int> test_rows, int D, float threshold=4e-3f);

#endif  // BENCHMARK_H
