#ifndef GELU_H
#define GELU_H

#include <cstdint>

/**
 * @brief GELU Activation Function (Approximation)
 *
 * Formula: GELU(x) ≈ x · σ(1.702x)
 *          where σ(x) = 1 / (1 + exp(-x))
 *
 * This approximation is faster and more hardware-friendly than the
 * exact GELU formula involving error function (erf).
 *
 * Input/Output: bf16 (16-bit Brain Floating Point)
 * Internal Precision: 25-bit extended format
 *
 * @param x_bf16 Input vector in bf16 format
 * @param y_bf16 Output vector in bf16 format
 * @param len Length of input vector
 */
void gelu(uint16_t *x_bf16, uint16_t *y_bf16, const int len);

#endif  // GELU_H
