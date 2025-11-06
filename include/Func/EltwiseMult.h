#ifndef ELTWISEMULT_H
#define ELTWISEMULT_H

#include <cstdint>

/**
 * @brief Element-wise multiplication: y[i] = x0[i] * x1[i]
 *
 * @param x0_bf16 First input vector (bf16 format)
 * @param x1_bf16 Second input vector (bf16 format)
 * @param y_bf16 Output vector (bf16 format)
 * @param len Length of input vectors
 */
void eltwise_mult(uint16_t *x0_bf16, uint16_t *x1_bf16, uint16_t *y_bf16, const int len);

#endif  // ELTWISEMULT_H
