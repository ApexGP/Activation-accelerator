#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cstdint>

// Safe Softmax implementation (output in bf16 format)
void online_softmax(uint16_t *x_bf16, uint16_t *y_bf16, const int len);

#endif  // SOFTMAX_H
