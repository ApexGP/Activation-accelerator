#ifndef ELTWISEMULT_H
#define ELTWISEMULT_H

#include <cstdint>
#include <vector>

/**
 * @brief Element-wise multiplication: y[i] = x0[i] * x1[i]
 *
 * @param x0_bf16 First input vector (bf16 format)
 * @param x1_bf16 Second input vector (bf16 format)
 * @param y Output vector (bf16 format)
 * @param len Length of input vectors
 */
void eltwise_mult(const std::vector<uint16_t> &x0_bf16, const std::vector<uint16_t> &x1_bf16,
                  std::vector<uint16_t> &y, int len);

#endif  // ELTWISEMULT_H
