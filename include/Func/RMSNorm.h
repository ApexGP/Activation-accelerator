#ifndef RMSNORM_H
#define RMSNORM_H

#include <cstdint>
#include <vector>

// RMS normalization: y[i] = x[i] / sqrt(mean(x^2) + eps)
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void rms_norm(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len,
              float eps = 1e-6f);

#endif  // RMSNORM_H
