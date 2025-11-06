#ifndef GELU_H
#define GELU_H

#include <cstdint>
#include <vector>

/**
 * @brief GELU Activation Function (Approximation)
 *
 * Formula: GELU(x) ≈ x · σ(1.702x)
 *          where σ(x) = 1 / (1 + exp(-x))
 *
 * This approximation is faster and more hardware-friendly than the
 * exact GELU formula involving error function (erf).
 *
 * Input/Output: BF16 (16-bit Brain Floating Point)
 * Internal Precision: 25-bit extended format
 *
 * @param x_bf16 Input vector in BF16 format
 * @param y Output vector in BF16 format (resized automatically)
 * @param len Length of input vector
 */
void gelu_activation(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len);

#endif  // GELU_H
