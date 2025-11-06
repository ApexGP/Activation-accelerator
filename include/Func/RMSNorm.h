#ifndef RMSNORM_H
#define RMSNORM_H

#include <cstdint>

// RMS normalization: y[i] = x[i] / sqrt(mean(x^2) + eps)
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void rms_norm(uint16_t *x_bf16, uint16_t *y_bf16, const int len, float eps = 1e-6f);

#endif  // RMSNORM_H
