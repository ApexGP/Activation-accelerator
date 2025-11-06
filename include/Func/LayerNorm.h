#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <cstdint>
#include <vector>

// Layer normalization: y[i] = (x[i] - mean) / sqrt(var + eps)
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void layer_norm(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len,
                float eps = 1e-6f);

#endif  // LAYERNORM_H
