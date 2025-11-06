#ifndef ELTWISE_ADD_H
#define ELTWISE_ADD_H

#include <cstdint>

// Element-wise addition: y = x0 + x1
// Input: bf16 format (wE=8, wF=7)
// Output: bf16 format (internally computed with extended precision, truncated before output)
void eltwise_add(uint16_t *x0_bf16, uint16_t *x1_bf16, uint16_t *y_bf16, const int len);

#endif  // ELTWISE_ADD_H
