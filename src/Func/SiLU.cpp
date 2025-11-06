#include <vector>

#include "Arith/Exp.h"
#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Func/SiLU.h"
#include "Utils/FormatUtils.h"
#include "Utils/Logger.h"
#define __REDUCED_LOG__

void silu_activation(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len)
{
#ifndef __REDUCED_LOG__
    LOG("========================================");
    LOG("SiLU (Swish) Activation Computation (25-bit precision)");
    LOG("Formula: SiLU(x) = x * σ(x) = x / (1 + exp(-x))");
    LOG("Input length: " << len);
    LOG("Algorithm: Unified 25-bit format throughout");
    LOG("========================================");
#endif

    y.resize(len);

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
    // 处理每个元素（全程 25-bit）
    // ============================================================
    Exp exp_module;
    FPAdd fpadd_module;
    FPMult mult_module;

    // 预计算 1.0 的 25-bit 表示
    uint32_t one_25bit = float_to_extended_25bit(1.0f);

    for (int i = 0; i < len; i++) {
#ifndef __REDUCED_LOG__
        LOG("\n[" << i << "] Processing SiLU(x[" << i << "])");
        LOG("  x (25-bit): 0x" << std::hex << x_25bit[i] << std::dec);
#endif

        // 检查特殊值
        ValueType x_type = classify_25bit(x_25bit[i]);

        // SiLU(NaN) = NaN
        if (x_type == ValueType::NaN) {
#ifndef __REDUCED_LOG__
            LOG("  ⚠️  NaN detected → SiLU(x) = NaN");
#endif
            uint32_t nan_25bit = (3U << 23) | (0xFF << 14) | 0x1;
            float nan_float = extended_25bit_to_float(nan_25bit);
            y[i] = float_to_bf16(nan_float);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // SiLU(+∞) = +∞ * σ(+∞) = +∞ * 1 = +∞
        if (x_type == ValueType::PosInf) {
#ifndef __REDUCED_LOG__
            LOG("  +Inf detected → SiLU(+∞) = +∞");
#endif
            uint32_t pos_inf_25bit = (2U << 23) | (0xFF << 14);
            float pos_inf_float = extended_25bit_to_float(pos_inf_25bit);
            y[i] = float_to_bf16(pos_inf_float);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // SiLU(-∞) = -∞ * σ(-∞) = -∞ * 0 → 0 (by L'Hôpital's rule)
        if (x_type == ValueType::NegInf) {
#ifndef __REDUCED_LOG__
            LOG("  -Inf detected → SiLU(-∞) = 0.0");
#endif
            y[i] = float_to_bf16(0.0f);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // 数值稳定性检查（避免 exp 上溢/下溢）
        float x_float = extended_25bit_to_float(x_25bit[i]);

        // 对于 x > 88，SiLU(x) ≈ x（σ(x) ≈ 1）
        if (x_float > 88.0f) {
#ifndef __REDUCED_LOG__
            LOG("  x > 88, SiLU(x) ≈ x (σ(x) ≈ 1)");
#endif
            y[i] = float_to_bf16(x_float);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // 对于 x < -88，SiLU(x) ≈ 0（σ(x) ≈ 0）
        if (x_float < -88.0f) {
#ifndef __REDUCED_LOG__
            LOG("  x < -88, SiLU(x) ≈ 0.0 (σ(x) ≈ 0)");
#endif
            y[i] = float_to_bf16(0.0f);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // 正常情况：计算 -x（翻转符号位）
        uint32_t neg_x_25bit = x_25bit[i] ^ (1U << 22);  // Flip sign bit at position 22
#ifndef __REDUCED_LOG__
        LOG("  -x (25-bit): 0x" << std::hex << neg_x_25bit << std::dec);
#endif

        // 计算 exp(-x)
        exp_module.reset();
        exp_module.set_input(neg_x_25bit, true);

        for (int j = 0; j < 8; j++) {
            exp_module.clock_step();
        }

        uint32_t exp_neg_x_25bit = exp_module.get_Exp_Ans_extended();
#ifndef __REDUCED_LOG__
        LOG("  exp(-x) (25-bit): 0x" << std::hex << exp_neg_x_25bit << std::dec);
#endif

        // 检查 exp 结果是否溢出
        ValueType exp_type = classify_25bit(exp_neg_x_25bit);
        if (exp_type == ValueType::PosInf) {
            // exp(-x) = +∞，说明 x < 0 且 |x| 很大，SiLU(x) ≈ 0
#ifndef __REDUCED_LOG__
            LOG("  ⚠️  exp(-x) overflow → SiLU(x) ≈ 0.0");
#endif
            y[i] = float_to_bf16(0.0f);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // 计算 1 + exp(-x)（全程 25-bit）
        fpadd_module.reset();
        fpadd_module.set_input(one_25bit, exp_neg_x_25bit);

        for (int cycle = 0; cycle < 2; cycle++) {
            fpadd_module.clock_step();
        }

        uint32_t sum_25bit = fpadd_module.get_result_extended();
#ifndef __REDUCED_LOG__
        LOG("  1 + exp(-x) (25-bit): 0x" << std::hex << sum_25bit << std::dec);
#endif

        // 计算 σ(x) = 1 / (1 + exp(-x))
        float sum_float = extended_25bit_to_float(sum_25bit);
        float sigmoid_float = 1.0f / sum_float;
        uint32_t sigmoid_25bit = float_to_extended_25bit(sigmoid_float);
#ifndef __REDUCED_LOG__
        LOG("  σ(x) (25-bit): 0x" << std::hex << sigmoid_25bit << std::dec);
#endif

        // 计算 x * σ(x)（使用 FPMult 硬件模块，全程 25-bit）
        mult_module.set_input(x_25bit[i], sigmoid_25bit);
        mult_module.clock_step();  // Cycle 0
        mult_module.clock_step();  // Cycle 1

        uint32_t result_25bit = mult_module.get_result();
#ifndef __REDUCED_LOG__
        LOG("  x * σ(x) (25-bit): 0x" << std::hex << result_25bit << std::dec);
#endif

        // 输出转换：25-bit → float → bf16
        float result_float = extended_25bit_to_float(result_25bit);
        y[i] = float_to_bf16(result_float);
        // y[i] = extended_25bit_to_bf16(result_25bit);

#ifndef __REDUCED_LOG__
        LOG("  Final result SiLU(x) = 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
    }
#ifndef __REDUCED_LOG__
    LOG("\n========================================");
    LOG("SiLU Activation Computation Completed");
    LOG("========================================\n");
#endif
}
