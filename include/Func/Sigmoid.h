#ifndef SIGMOID_H
#define SIGMOID_H

#include <cstdint>
#include <vector>

// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void sigmoid_activation(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len);

#endif  // SIGMOID_H
