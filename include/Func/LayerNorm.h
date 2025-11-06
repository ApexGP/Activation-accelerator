#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <cstdint>

// Layer normalization: y[i] = (x[i] - mean) / sqrt(var + eps)
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void layer_norm(uint16_t *x_bf16, uint16_t *y_bf16, const int len, float eps = 1e-6f);

#endif  // LAYERNORM_H
