#ifndef ACTIVATION_ACCELERATOR_H
#define ACTIVATION_ACCELERATOR_H

#include <cstdint>

#include "Utils/Config.h"

// Function declaration - 使用uint16数据类型
void activation_accelerator(uint16_t *in0, uint16_t *in1, uint16_t *out, uint32_t stage,
                            uint32_t config);

// Data size
#define DATA_SIZE (32 * 1024)

#endif  // ACTIVATION_ACCELERATOR_H
