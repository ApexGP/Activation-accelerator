#ifndef FORMAT_UTILS_H
#define FORMAT_UTILS_H

#include <cstdint>

// Data format conversion utilities

void print_hex24(const char *label, uint32_t value);
void print_hex16(const char *label, uint16_t value);

float bf16_to_float(uint16_t bf16);
uint16_t float_to_bf16(float val);

// 25-bit extended format conversion (wE=8, wF=14) for FPAdd/FPMult/Exp
// Format: exn(2) + sign(1) + exp(8) + mantissa(14) = 25-bit
// exn encoding: 0=zero, 1=normal, 2=infinity, 3=NaN
uint32_t float_to_extended_25bit(float val);
float extended_25bit_to_float(uint32_t val25);

// Direct conversion from 25-bit to bf16 (avoids intermediate float32 conversion)
uint16_t extended_25bit_to_bf16(uint32_t val25);

// Special value classification (used in computation flow)
enum class ValueType { Zero = 0, Normal = 1, PosInf = 2, NegInf = 3, NaN = 4 };

// Classify value type during 25-bit computation (runtime check)
ValueType classify_25bit(uint32_t val25);

#endif  // FORMAT_UTILS_H
