#include <iomanip>
#include <iostream>

#include "Utils/FormatUtils.h"

void print_hex24(const char *label, uint32_t value)
{
    std::cout << label << ": 0x" << std::hex << std::setw(6) << std::setfill('0')
              << (value & 0xFFFFFF) << std::dec << std::endl;
}

void print_hex16(const char *label, uint16_t value)
{
    std::cout << label << ": 0x" << std::hex << std::setw(4) << std::setfill('0') << value
              << std::dec << std::endl;
}

float bf16_to_float(uint16_t bf16)
{
    uint32_t f32_bits = ((uint32_t) bf16) << 16;
    return *(float *) &f32_bits;
}

uint16_t float_to_bf16(float val)
{
    uint32_t f32_bits = *(uint32_t *) &val;

    // Extract components
    uint32_t sign = (f32_bits >> 31) & 0x1;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    uint32_t mantissa_23bit = f32_bits & 0x7FFFFF;

    // Handle special cases (Inf, NaN)
    if (exp == 0xFF) {
        // Infinity or NaN: preserve as-is
        return (uint16_t) (f32_bits >> 16);
    }

    // Handle zero and denormals
    if (exp == 0) {
        // Zero or denormal: round to zero in bf16
        return (uint16_t) (sign << 15);
    }

    // Round mantissa from 23-bit to 7-bit using round-to-nearest-even
    uint32_t mantissa_7bit = mantissa_23bit >> 16;
    uint32_t dropped_bits = mantissa_23bit & 0xFFFF;  // Lower 16 bits being dropped
    uint32_t half = 0x8000;                           // 0.5 in fixed-point (bit 15)

    // Round-to-nearest-even (banker's rounding)
    if (dropped_bits > half) {
        // Greater than 0.5: round up
        mantissa_7bit++;
    } else if (dropped_bits == half) {
        // Exactly 0.5: round to even (if odd, round up)
        if (mantissa_7bit & 0x1) {
            mantissa_7bit++;
        }
    }
    // Less than 0.5: round down (no change)

    // Handle mantissa overflow (carry into exponent)
    if (mantissa_7bit > 0x7F) {
        mantissa_7bit = 0;
        exp++;
        if (exp >= 0xFF) {
            // Overflow to infinity
            exp = 0xFF;
            mantissa_7bit = 0;
        }
    }

    // Construct bf16: sign[15] | exp[14:7] | mantissa[6:0]
    return (uint16_t) ((sign << 15) | (exp << 7) | mantissa_7bit);
}

// Direct conversion from 25-bit extended format to bf16
// This avoids intermediate float32 conversion and preserves rounding precision
uint16_t extended_25bit_to_bf16(uint32_t val25)
{
    // Extract 25-bit components: exn(2) + sign(1) + exp(8) + mantissa(14)
    uint32_t exn = (val25 >> 23) & 0x3;
    uint32_t sign = (val25 >> 22) & 0x1;
    uint32_t exp = (val25 >> 14) & 0xFF;
    uint32_t mantissa_14bit = val25 & 0x3FFF;

    // Handle special cases
    if (exn == 0) {
        // Zero
        return (uint16_t) (sign << 15);
    } else if (exn == 2) {
        // Infinity
        return (uint16_t) ((sign << 15) | (0xFF << 7));
    } else if (exn == 3) {
        // NaN
        uint32_t mantissa_7bit =
            (mantissa_14bit >> 7) | 0x1;  // Preserve NaN with at least 1 bit set
        return (uint16_t) ((sign << 15) | (0xFF << 7) | mantissa_7bit);
    }

    // Normal number: round mantissa from 14-bit to 7-bit
    uint32_t mantissa_7bit = mantissa_14bit >> 7;
    uint32_t dropped_bits = mantissa_14bit & 0x7F;  // Lower 7 bits being dropped
    uint32_t half = 0x40;                           // 0.5 in fixed-point (bit 6)

    // Round-to-nearest-even (banker's rounding)
    if (dropped_bits > half) {
        // Greater than 0.5: round up
        mantissa_7bit++;
    } else if (dropped_bits == half) {
        // Exactly 0.5: round to even (if odd, round up)
        if (mantissa_7bit & 0x1) {
            mantissa_7bit++;
        }
    }
    // Less than 0.5: round down (no change)

    // Handle mantissa overflow (carry into exponent)
    if (mantissa_7bit > 0x7F) {
        mantissa_7bit = 0;
        exp++;
        if (exp >= 0xFF) {
            // Overflow to infinity
            exp = 0xFF;
            mantissa_7bit = 0;
        }
    }

    // Construct bf16: sign[15] | exp[14:7] | mantissa[6:0]
    return (uint16_t) ((sign << 15) | (exp << 7) | mantissa_7bit);
}

// Convert float to 25-bit extended format (wE=8, wF=14)
// Format: exn(2) + sign(1) + exp(8) + mantissa(14)
uint32_t float_to_extended_25bit(float val)
{
    uint32_t f32_bits = *(uint32_t *) &val;

    uint32_t sign = (f32_bits >> 31) & 0x1;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    uint32_t mantissa_23bit = f32_bits & 0x7FFFFF;

    // Handle special cases first
    uint32_t exn = 1;  // Normal number by default
    if (exp == 0 && mantissa_23bit == 0) {
        // Zero
        exn = 0;
        return ((exn << 23) | (sign << 22)) & 0x1FFFFFF;
    } else if (exp == 0xFF) {
        if (mantissa_23bit == 0) {
            // Infinity
            exn = 2;
        } else {
            // NaN
            exn = 3;
        }
        uint32_t mantissa_14bit = mantissa_23bit >> 9;
        return ((exn << 23) | (sign << 22) | (exp << 14) | mantissa_14bit) & 0x1FFFFFF;
    }

    // Round mantissa from 23-bit to 14-bit using round-to-nearest-even
    uint32_t mantissa_14bit = mantissa_23bit >> 9;
    uint32_t dropped_bits = mantissa_23bit & 0x1FF;  // Lower 9 bits being dropped
    uint32_t half = 0x100;                           // 0.5 in fixed-point (bit 8)

    // Round-to-nearest-even (banker's rounding)
    if (dropped_bits > half) {
        // Greater than 0.5: round up
        mantissa_14bit++;
    } else if (dropped_bits == half) {
        // Exactly 0.5: round to even (if odd, round up)
        if (mantissa_14bit & 0x1) {
            mantissa_14bit++;
        }
    }
    // Less than 0.5: round down (no change)

    // Handle mantissa overflow (carry into exponent)
    if (mantissa_14bit > 0x3FFF) {
        mantissa_14bit = 0;
        exp++;
        if (exp >= 0xFF) {
            // Overflow to infinity
            exn = 2;
            exp = 0xFF;
            mantissa_14bit = 0;
        }
    }

    // Construct 25-bit result: exn[24:23] | sign[22] | exp[21:14] | mantissa[13:0]
    return ((exn << 23) | (sign << 22) | (exp << 14) | mantissa_14bit) & 0x1FFFFFF;
}

// Extract float from 25-bit extended format
// Format: exn(2) + sign(1) + exp(8) + mantissa(14)
float extended_25bit_to_float(uint32_t val25)
{
    uint32_t exn = (val25 >> 23) & 0x3;
    uint32_t sign = (val25 >> 22) & 0x1;
    uint32_t exp = (val25 >> 14) & 0xFF;
    uint32_t mantissa_14bit = val25 & 0x3FFF;

    // Handle special cases based on exn (FPAdd/FPMult convention)
    // exn=0: zero, exn=1: normal, exn=2: infinity, exn=3: NaN
    if (exn == 0) {
        // Zero
        uint32_t f32_bits = (sign << 31);
        return *(float *) &f32_bits;
    } else if (exn == 2) {
        // Infinity
        uint32_t f32_bits = (sign << 31) | (0xFF << 23);
        return *(float *) &f32_bits;
    } else if (exn == 3) {
        // NaN
        uint32_t f32_bits = (sign << 31) | (0xFF << 23) | 0x400000;
        return *(float *) &f32_bits;
    }

    // Normal number (exn=1): extend mantissa from 14-bit to 23-bit
    uint32_t mantissa_23bit = mantissa_14bit << 9;
    uint32_t f32_bits = (sign << 31) | (exp << 23) | mantissa_23bit;
    return *(float *) &f32_bits;
}

// Classify value type using exn encoding (25-bit extended format)
// exn=0: zero, exn=1: normal, exn=2: infinity, exn=3: NaN
ValueType classify_25bit(uint32_t val25)
{
    uint32_t exn = (val25 >> 23) & 0x3;
    if (exn == 0) return ValueType::Zero;
    if (exn == 1) return ValueType::Normal;
    if (exn == 3) return ValueType::NaN;
    // exn == 2: infinity, check sign bit
    return ((val25 >> 22) & 0x1) ? ValueType::NegInf : ValueType::PosInf;
}
