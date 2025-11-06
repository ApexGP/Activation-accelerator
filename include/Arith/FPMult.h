#ifndef FPMULT_H
#define FPMULT_H

#include <cstdint>

// Floating-point multiplier for bf16 extended format
// Based on FloPoCo generated FPMult_8_14 (wF=14 upgrade)
// Input/Output format: 25-bit = exn(2) + sign(1) + exp(8) + mantissa(14)
// Pipeline: 2 stages
class FPMult
{
public:
    FPMult();
    void reset();

    // Set input operands (25-bit extended bf16 format)
    void set_input(uint32_t X, uint32_t Y);

    // Execute one clock cycle (2-stage pipeline)
    void clock_step();

    // Get result (25-bit extended bf16 format)
    uint32_t get_result() const
    {
#pragma HLS INLINE
        return R;
    }

private:
    // Input operands
    uint32_t X_input;
    uint32_t Y_input;

    // Pipeline stage 0 (cycle 0)
    uint32_t sign;
    uint8_t expX;
    uint8_t expY;
    uint16_t expSumPreSub;  // 10-bit
    uint16_t bias;          // 10-bit
    uint16_t expSum;        // 10-bit
    uint16_t sigX;          // 15-bit (1 + 14-bit mantissa)
    uint16_t sigY;          // 15-bit (1 + 14-bit mantissa)
    uint32_t sigProd;       // 30-bit (15 × 15)
    uint8_t excSel;         // 4-bit
    uint8_t exc;            // 2-bit
    bool norm;
    uint16_t expPostNorm;  // 10-bit
    uint32_t sigProdExt;   // 30-bit
    uint32_t expSig;       // 24-bit (10-bit exp + 14-bit mantissa)
    bool sticky;

    // Pipeline stage 1 (cycle 1) - delayed signals
    uint32_t sign_d1;
    uint8_t exc_d1;
    uint32_t sigProdExt_d1;
    bool sticky_d1;

    // Output
    uint32_t R;

    // Current pipeline cycle (0 or 1)
    int cycle;

    // Helper functions
    uint32_t multiply_15x15(uint16_t a, uint16_t b);   // 15-bit × 15-bit = 30-bit
    uint32_t round_add_24bit(uint32_t X, bool round);  // 24-bit adder with carry
};

#endif  // FPMULT_H
