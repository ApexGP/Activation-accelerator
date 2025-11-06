#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cstdint>
#include <vector>

// Safe Softmax implementation (output in bf16 format)
void online_softmax(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y,
                    int exp_output_bits = 14);

#endif  // SOFTMAX_H
