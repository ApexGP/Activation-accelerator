#ifndef SILU_H
#define SILU_H

#include <cstdint>
#include <vector>

// SiLU (Swish) activation: SiLU(x) = x * σ(x) = x / (1 + exp(-x))
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void silu_activation(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len);

#endif  // SILU_H
