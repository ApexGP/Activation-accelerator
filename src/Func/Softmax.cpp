#include "Arith/Exp.h"
#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Arith/FPSub.h"
#include "Func/Softmax.h"
#include "Utils/Config.h"
#include "Utils/FormatUtils.h"
#include "Utils/HlsVector.h"

// Online Softmax implementation (full 25-bit precision throughout, reduce format conversion)
void online_softmax(uint16_t *x_bf16, uint16_t *y_bf16, const int len)
{
    // ============================================================
    // Input conversion: bf16 → float → 25-bit (convert once)
    // ============================================================
    uint32_t x_25bit[len];
    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        x_25bit[i] = float_to_extended_25bit(bf16_to_float(x_bf16[i]));
    }

    // ============================================================
    // Initialize: m_old = -inf, s_old = 0 (25-bit format)
    // ============================================================
    // -inf in 25-bit: exn=2 (inf), sign=1 (negative), exp=0, mantissa=0
    uint32_t m_old = (2U << 23) | (1U << 22);  // exn=2, sign=1, exp=0, mant=0
    uint32_t s_old = 0U;                       // exn=0 (zero)

    // Record indices where inf appears during computation
    HlsVector<int, MAX_CAPACITY> computed_inf_indices;
    HlsVector<int, MAX_CAPACITY> input_pos_inf_indices;
    bool has_nan = false;
    bool all_neg_inf = true;

    // ============================================================
    // First pass: dynamically maintain max and sum (full 25-bit precision)
    // ============================================================
    for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
        uint32_t x_curr = x_25bit[i];
        ValueType x_type = classify_25bit(x_curr);

        // Check NaN
        if (x_type == ValueType::NaN) {
            has_nan = true;
            break;
        }

        // Check +inf (input inf has highest priority)
        if (x_type == ValueType::PosInf) {
            input_pos_inf_indices.push_back(i);
            all_neg_inf = false;
            continue;
        }

        // Check if not -inf
        if (x_type != ValueType::NegInf) {
            all_neg_inf = false;
        }

        // If there is already input +inf, skip subsequent computation
        if (!input_pos_inf_indices.empty()) {
            continue;
        }

        // Compute m_new = max(m_old, x_curr)
        // Use hardware comparison: determine magnitude through subtraction
        FPSub fpsub_cmp;
        fpsub_cmp.reset();
        fpsub_cmp.set_input(x_curr, m_old);
        for (int j = 0; j < 2; j++) {
            fpsub_cmp.clock_step();
        }
        uint32_t diff_cmp = fpsub_cmp.get_result_extended();

        // If x - m_old >= 0, then x is larger
        bool x_greater = ((diff_cmp >> 22) & 0x1) == 0;  // sign bit = 0

        uint32_t m_new = x_greater ? x_curr : m_old;
        bool m_changed = x_greater;

        if (m_changed) {
            // If s_old > 0 and m_old is not -inf, need to adjust s_old
            ValueType s_type = classify_25bit(s_old);
            ValueType m_old_type = classify_25bit(m_old);

            if (s_type != ValueType::Zero && m_old_type != ValueType::NegInf) {
                // Compute m_old - m_new (using FPSub)
                FPSub fpsub_diff;
                fpsub_diff.reset();
                fpsub_diff.set_input(m_old, m_new);
                for (int j = 0; j < 2; j++) {
                    fpsub_diff.clock_step();
                }
                uint32_t diff_25bit = fpsub_diff.get_result_extended();

                // Compute exp(m_old - m_new)
                Exp exp_module;
                exp_module.reset();
                exp_module.set_input(diff_25bit, true);
                for (int j = 0; j < 8; j++) {
                    exp_module.clock_step();
                }
                uint32_t exp_diff = exp_module.get_Exp_Ans_extended();

                // Compute s_old * exp(m_old - m_new) (using FPMult)
                FPMult mult_module;
                mult_module.set_input(s_old, exp_diff);
                mult_module.clock_step();  // Cycle 0
                mult_module.clock_step();  // Cycle 1
                s_old = mult_module.get_result();

            } else {
            }
        } else {
        }

        // Compute exp(x - m_new)
        FPSub fpsub_x;
        fpsub_x.reset();
        fpsub_x.set_input(x_curr, m_new);
        for (int j = 0; j < 2; j++) {
            fpsub_x.clock_step();
        }
        uint32_t x_minus_m = fpsub_x.get_result_extended();
        float x_minus_m_float = extended_25bit_to_float(x_minus_m);

        // Safe Softmax: if difference < -88, exp will underflow, skip directly
        if (x_minus_m_float < -88.0f) {
            m_old = m_new;
            continue;
        }

        Exp exp_module2;
        exp_module2.reset();
        exp_module2.set_input(x_minus_m, true);
        for (int j = 0; j < 8; j++) {
            exp_module2.clock_step();
        }
        uint32_t exp_val = exp_module2.get_Exp_Ans_extended();

        // Check if exp result overflows to +inf (computed inf)
        if (classify_25bit(exp_val) == ValueType::PosInf) {
            // Discard previous accumulation
            s_old = 0U;
            computed_inf_indices.clear();
            computed_inf_indices.push_back(i);
            m_old = m_new;
            continue;
        }

        // Accumulate: s_old = s_old + exp_val (using FPAdd)
        FPAdd fadd_module;
        fadd_module.reset();
        fadd_module.set_input(s_old, exp_val);
        for (int j = 0; j < 2; j++) {
            fadd_module.clock_step();
        }
        s_old = fadd_module.get_result_extended();

        m_old = m_new;
    }

    // ============================================================
    // Special case handling
    // ============================================================

    // 1. NaN handling
    if (has_nan) {
        uint16_t nan_bf16 = 0x7FC0;
        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
            y_bf16[i] = nan_bf16;
        }
        return;
    }

    // 2. Input +inf handling (highest priority)
    if (!input_pos_inf_indices.empty()) {
        float prob = 1.0f / input_pos_inf_indices.size();
        uint16_t prob_bf16 = float_to_bf16(prob);
        uint16_t zero_bf16 = 0x0000;

        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
            y_bf16[i] = zero_bf16;
        }
        for (int idx : input_pos_inf_indices) {
#pragma HLS PIPELINE II = 1
            y_bf16[idx] = prob_bf16;
        }
        return;
    }

    // 3. All -inf handling
    if (all_neg_inf) {
        uint16_t nan_bf16 = 0x7FC0;
        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
            y_bf16[i] = nan_bf16;
        }
        return;
    }

    // 4. Computed inf handling
    if (!computed_inf_indices.empty()) {
        float prob = 1.0f / computed_inf_indices.size();
        uint16_t prob_bf16 = float_to_bf16(prob);
        uint16_t zero_bf16 = 0x0000;

        for (int i = 0; i < len; i++) {
#pragma HLS PIPELINE II = 1
            y_bf16[i] = zero_bf16;
        }
        for (int idx : computed_inf_indices) {
#pragma HLS PIPELINE II = 1
            y_bf16[idx] = prob_bf16;
        }
        return;
    }

    // ============================================================
    // Second pass: compute final softmax output (full 25-bit precision)
    // ============================================================

    uint32_t m_final = m_old;
    uint32_t s_final = s_old;

    for (int i = 0; i < len; i++) {
        // Compute exp(x[i] - m_final)
        FPSub fpsub_final;
        fpsub_final.reset();
        fpsub_final.set_input(x_25bit[i], m_final);
        for (int j = 0; j < 2; j++) {
            fpsub_final.clock_step();
        }
        uint32_t diff_25bit = fpsub_final.get_result_extended();
        float diff_float = extended_25bit_to_float(diff_25bit);

        // Safe Softmax: if difference < -88, exp will underflow, output 0 directly
        if (diff_float < -88.0f) {
            y_bf16[i] = 0x0000;  // bf16 zero
            continue;
        }

        Exp exp_final;
        exp_final.reset();
        exp_final.set_input(diff_25bit, true);
        for (int j = 0; j < 8; j++) {
            exp_final.clock_step();
        }
        uint32_t exp_val = exp_final.get_Exp_Ans_extended();

        // Compute exp_val / s_final (convert to float for division, then convert back to bf16)
        float exp_float = extended_25bit_to_float(exp_val);
        float s_float = extended_25bit_to_float(s_final);
        float result_float = exp_float / s_float;

        y_bf16[i] = float_to_bf16(result_float);
    }
}
