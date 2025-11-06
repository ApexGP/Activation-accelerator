#ifndef SIGMOID_H
#define SIGMOID_H

#include <cstdint>

// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void sigmoid(uint32_t x_25bit, uint32_t &y_25bit);

#endif  // SIGMOID_H
