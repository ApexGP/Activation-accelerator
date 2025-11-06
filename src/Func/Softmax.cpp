#include <iostream>
#include <vector>

#include "Arith/Exp.h"
#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Arith/FPSub.h"
#include "Func/Softmax.h"
#include "Utils/FormatUtils.h"
#include "Utils/Logger.h"
#define __REDUCED_LOG__

// Online Softmax implementation (全程使用 25-bit 精度，减少格式转换)
void online_softmax(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y,
                    int exp_output_bits)
{
    int len = x_bf16.size();
    y.resize(len);

#ifndef __REDUCED_LOG__
    LOG("========================================");
    LOG("Online Softmax Computation (25-bit precision)");
    LOG("Algorithm: Online Softmax with unified 25-bit format");
    LOG("Input length: " << len);
    LOG("========================================");
#endif

    // ============================================================
    // 输入转换：bf16 → float → 25-bit (只转换一次)
    // ============================================================
    std::vector<uint32_t> x_25bit(len);
    for (int i = 0; i < len; i++) {
        float x_float = bf16_to_float(x_bf16[i]);
        x_25bit[i] = float_to_extended_25bit(x_float);
    }

#ifndef __REDUCED_LOG__
    LOG("\nInput converted to 25-bit extended format");
#endif

    // ============================================================
    // 初始化：m_old = -inf, s_old = 0 (25-bit 格式)
    // ============================================================
    // -inf in 25-bit: exn=2 (inf), sign=1 (negative), exp=0, mantissa=0
    uint32_t m_old = (2U << 23) | (1U << 22);  // exn=2, sign=1, exp=0, mant=0
    uint32_t s_old = 0U;                       // exn=0 (zero)

    // 记录计算过程中出现 inf 的位置
    std::vector<int> computed_inf_indices;
    std::vector<int> input_pos_inf_indices;
    bool has_nan = false;
    bool all_neg_inf = true;

#ifndef __REDUCED_LOG__
    LOG("\n--- Pass 1: Maintain m_max and s_sum ---");
    LOG("Initialize: m_old = -inf (25-bit: 0x" << std::hex << m_old << std::dec << ")");
    LOG("            s_old = 0 (25-bit: 0x" << std::hex << s_old << std::dec << ")");
#endif

    // ============================================================
    // 第一遍：动态维护 max 和 sum (全程 25-bit)
    // ============================================================
    for (int i = 0; i < len; i++) {
        uint32_t x_curr = x_25bit[i];
        ValueType x_type = classify_25bit(x_curr);

#ifndef __REDUCED_LOG__
        LOG("\n[" << i << "] Processing x[" << i << "] (25-bit: 0x" << std::hex << x_curr
                  << std::dec << ")");
#endif

        // 检查 NaN
        if (x_type == ValueType::NaN) {
#ifndef __REDUCED_LOG__
            LOG("  ⚠️  NaN detected in input");
#endif
            has_nan = true;
            break;
        }

        // 检查 +inf（输入 inf 优先级最高）
        if (x_type == ValueType::PosInf) {
#ifndef __REDUCED_LOG__
            LOG("  +inf detected in input (highest priority)");
#endif
            input_pos_inf_indices.push_back(i);
            all_neg_inf = false;
            continue;
        }

        // 检查是否不是 -inf
        if (x_type != ValueType::NegInf) {
            all_neg_inf = false;
        }

        // 如果已经有输入 +inf，跳过后续计算
        if (!input_pos_inf_indices.empty()) {
            continue;
        }

        // 计算 m_new = max(m_old, x_curr)
        // 使用硬件比较：通过减法判断大小
        FPSub fpsub_cmp;
        fpsub_cmp.reset();
        fpsub_cmp.set_input(x_curr, m_old);
        for (int j = 0; j < 2; j++) {
            fpsub_cmp.clock_step();
        }
        uint32_t diff_cmp = fpsub_cmp.get_result_extended();

        // 如果 x - m_old >= 0，则 x 更大
        bool x_greater = ((diff_cmp >> 22) & 0x1) == 0;  // sign bit = 0

        uint32_t m_new = x_greater ? x_curr : m_old;
        bool m_changed = x_greater;

        if (m_changed) {
#ifndef __REDUCED_LOG__
            LOG("  m updated: 0x" << std::hex << m_old << " -> 0x" << m_new << std::dec);
#endif

            // 如果 s_old > 0 且 m_old 不是 -inf，需要调整 s_old
            ValueType s_type = classify_25bit(s_old);
            ValueType m_old_type = classify_25bit(m_old);

            if (s_type != ValueType::Zero && m_old_type != ValueType::NegInf) {
                // 计算 m_old - m_new (使用 FPSub)
                FPSub fpsub_diff;
                fpsub_diff.reset();
                fpsub_diff.set_input(m_old, m_new);
                for (int j = 0; j < 2; j++) {
                    fpsub_diff.clock_step();
                }
                uint32_t diff_25bit = fpsub_diff.get_result_extended();

#ifndef __REDUCED_LOG__
                LOG("  m_old - m_new = 0x" << std::hex << diff_25bit << std::dec);
#endif

                // 计算 exp(m_old - m_new)
                Exp exp_module;
                exp_module.reset();
                exp_module.set_input(diff_25bit, true);
                for (int j = 0; j < 8; j++) {
                    exp_module.clock_step();
                }
                uint32_t exp_diff = exp_module.get_Exp_Ans_extended();

#ifndef __REDUCED_LOG__
                LOG("  exp(m_old - m_new) = 0x" << std::hex << exp_diff << std::dec);
#endif

                // 计算 s_old * exp(m_old - m_new) (使用 FPMult)
                FPMult mult_module;
                mult_module.set_input(s_old, exp_diff);
                mult_module.clock_step();  // Cycle 0
                mult_module.clock_step();  // Cycle 1
                s_old = mult_module.get_result();

#ifndef __REDUCED_LOG__
                LOG("  s_old adjusted = 0x" << std::hex << s_old << std::dec);
#endif
            } else {
#ifndef __REDUCED_LOG__
                LOG("  s_old = 0 or m_old = -inf, no adjustment needed");
#endif
            }
        } else {
#ifndef __REDUCED_LOG__
            LOG("  m unchanged");
#endif
        }

        // 计算 exp(x - m_new)
        FPSub fpsub_x;
        fpsub_x.reset();
        fpsub_x.set_input(x_curr, m_new);
        for (int j = 0; j < 2; j++) {
            fpsub_x.clock_step();
        }
        uint32_t x_minus_m = fpsub_x.get_result_extended();
        float x_minus_m_float = extended_25bit_to_float(x_minus_m);

#ifndef __REDUCED_LOG__
        LOG("  x - m_new = 0x" << std::hex << x_minus_m << std::dec << " (" << x_minus_m_float
                               << ")");
#endif

        // Safe Softmax: 如果差值 < -88，exp 会下溢，直接跳过
        if (x_minus_m_float < -88.0f) {
#ifndef __REDUCED_LOG__
            LOG("  x - m_new < -88, exp underflow, skipping (treat as 0)");
#endif
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

#ifndef __REDUCED_LOG__
        LOG("  exp(x - m_new) = 0x" << std::hex << exp_val << std::dec);
#endif

        // 检查 exp 结果是否溢出为 +inf（计算 inf）
        if (classify_25bit(exp_val) == ValueType::PosInf) {
#ifndef __REDUCED_LOG__
            LOG("  ⚠️  exp overflow to +inf (computed inf detected)");
            LOG("  → Discarding previous s_old, recording computed inf");
#endif

            // 丢弃之前的累积
            s_old = 0U;
            computed_inf_indices.clear();
            computed_inf_indices.push_back(i);
            m_old = m_new;
            continue;
        }

        // 累加：s_old = s_old + exp_val (使用 FPAdd)
        FPAdd fadd_module;
        fadd_module.reset();
        fadd_module.set_input(s_old, exp_val);
        for (int j = 0; j < 2; j++) {
            fadd_module.clock_step();
        }
        s_old = fadd_module.get_result_extended();

#ifndef __REDUCED_LOG__
        LOG("  s updated = 0x" << std::hex << s_old << std::dec);
#endif

        m_old = m_new;
    }

    // ============================================================
    // 特殊情况处理
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Pass 1 Completed ---");
#endif

    // 1. NaN 处理
    if (has_nan) {
#ifndef __REDUCED_LOG__
        LOG("→ NaN detected: returning all NaN");
#endif
        uint16_t nan_bf16 = 0x7FC0;
        for (int i = 0; i < len; i++) {
            y[i] = nan_bf16;
        }
        return;
    }

    // 2. 输入 +inf 处理（优先级最高）
    if (!input_pos_inf_indices.empty()) {
#ifndef __REDUCED_LOG__
        LOG("→ Input +inf detected: uniform distribution over " << input_pos_inf_indices.size()
                                                                << " positions");
#endif
        float prob = 1.0f / input_pos_inf_indices.size();
        uint16_t prob_bf16 = float_to_bf16(prob);
        uint16_t zero_bf16 = 0x0000;

        for (int i = 0; i < len; i++) {
            y[i] = zero_bf16;
        }
        for (int idx : input_pos_inf_indices) {
            y[idx] = prob_bf16;
        }
        return;
    }

    // 3. 全部 -inf 处理
    if (all_neg_inf) {
#ifndef __REDUCED_LOG__
        LOG("→ All values are -inf: returning all NaN");
#endif
        uint16_t nan_bf16 = 0x7FC0;
        for (int i = 0; i < len; i++) {
            y[i] = nan_bf16;
        }
        return;
    }

    // 4. 计算 inf 处理
    if (!computed_inf_indices.empty()) {
#ifndef __REDUCED_LOG__
        LOG("→ Computed inf detected: uniform distribution over " << computed_inf_indices.size()
                                                                  << " positions");
#endif
        float prob = 1.0f / computed_inf_indices.size();
        uint16_t prob_bf16 = float_to_bf16(prob);
        uint16_t zero_bf16 = 0x0000;

        for (int i = 0; i < len; i++) {
            y[i] = zero_bf16;
        }
        for (int idx : computed_inf_indices) {
            y[idx] = prob_bf16;
        }
        return;
    }

    // ============================================================
    // 第二遍：计算最终 softmax 输出 (全程 25-bit)
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Pass 2: Compute final softmax output ---");
    LOG("m_final = 0x" << std::hex << m_old << std::dec);
    LOG("s_final = 0x" << std::hex << s_old << std::dec);
#endif

    uint32_t m_final = m_old;
    uint32_t s_final = s_old;

    for (int i = 0; i < len; i++) {
#ifndef __REDUCED_LOG__
        LOG("\n[" << i << "] Compute softmax[" << i << "]");
#endif

        // 计算 exp(x[i] - m_final)
        FPSub fpsub_final;
        fpsub_final.reset();
        fpsub_final.set_input(x_25bit[i], m_final);
        for (int j = 0; j < 2; j++) {
            fpsub_final.clock_step();
        }
        uint32_t diff_25bit = fpsub_final.get_result_extended();
        float diff_float = extended_25bit_to_float(diff_25bit);

        // Safe Softmax: 如果差值 < -88，exp 会下溢，直接输出 0
        if (diff_float < -88.0f) {
#ifndef __REDUCED_LOG__
            LOG("  x - m_final < -88, exp underflow, output = 0");
#endif
            y[i] = 0x0000;  // bf16 zero
            continue;
        }

        Exp exp_final;
        exp_final.reset();
        exp_final.set_input(diff_25bit, true);
        for (int j = 0; j < 8; j++) {
            exp_final.clock_step();
        }
        uint32_t exp_val = exp_final.get_Exp_Ans_extended();

#ifndef __REDUCED_LOG__
        LOG("  exp(x - m_final) = 0x" << std::hex << exp_val << std::dec);
#endif

        // 计算 exp_val / s_final (转换为 float 做除法，最后转回 bf16)
        float exp_float = extended_25bit_to_float(exp_val);
        float s_float = extended_25bit_to_float(s_final);
        float result_float = exp_float / s_float;

        y[i] = float_to_bf16(result_float);
#ifndef __REDUCED_LOG__
        LOG("  softmax[" << i << "] = " << result_float << " (bf16: 0x" << std::hex << y[i]
                         << std::dec << ")");
#endif
    }

#ifndef __REDUCED_LOG__
    LOG("\n========================================");
    LOG("Online Softmax Computation Completed");
    LOG("========================================\n");
#endif
}
