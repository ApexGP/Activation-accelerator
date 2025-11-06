#include "Utils/ConstDiv.h"

// Define lookup table in source file for C++11 compatibility
constexpr uint32_t ConstDiv::ONE_THIRD_23BIT;

constexpr uint32_t ConstDiv::Inv_k_LUT_23BIT[ConstDiv::MAX_DIM + 1];

uint32_t ConstDiv::get_inv_k_25bit(int k)
{
    if (k < 1 || k > MAX_DIM) {
        return 0;  // Invalid index
    }

    // Add exn=01 (normal number) to convert 23-bit to 25-bit
    return (1U << 23) | Inv_k_LUT_23BIT[k];
}

uint32_t ConstDiv::get_one_third_25bit()
{
    // Add exn=01 (normal number) to convert 23-bit to 25-bit
    return (1U << 23) | ONE_THIRD_23BIT;
}
