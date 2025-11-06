#include <cmath>
#include <vector>

#include "Arith/Exp.h"
#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Func/GELU.h"
#include "Utils/FormatUtils.h"
#include "Utils/Logger.h"
#define __REDUCED_LOG__

void gelu_activation(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len)
{
#ifndef __REDUCED_LOG__
    LOG("========================================");
    LOG("GELU Activation Computation (25-bit precision)");
    LOG("Formula: GELU(x) ≈ x · σ(1.702x)");
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
    // 预计算常量
    // ============================================================
    const float GELU_CONST = 1.702f;
    uint32_t const_1702_25bit = float_to_extended_25bit(GELU_CONST);
    uint32_t one_25bit = float_to_extended_25bit(1.0f);

#ifndef __REDUCED_LOG__
    LOG("\nConstants:");
    LOG("  1.702 (25-bit): 0x" << std::hex << const_1702_25bit << std::dec);
    LOG("  1.0 (25-bit): 0x" << std::hex << one_25bit << std::dec);
#endif

    // ============================================================
    // 处理每个元素（全程 25-bit）
    // ============================================================
    FPMult mult_module;
    Exp exp_module;
    FPAdd fpadd_module;

    for (int i = 0; i < len; i++) {
#ifndef __REDUCED_LOG__
        if (i < 5 || i >= len - 2) {
            LOG("\n[" << i << "] Processing GELU(x[" << i << "])");
        } else if (i == 5) {
            LOG("\n  ...");
        }
#endif

        // 检查特殊值
        ValueType x_type = classify_25bit(x_25bit[i]);

        // GELU(NaN) = NaN
        if (x_type == ValueType::NaN) {
#ifndef __REDUCED_LOG__
            LOG("  ⚠️  NaN detected → GELU(x) = NaN");
#endif
            uint32_t nan_25bit = (3U << 23) | (0xFF << 14) | 0x1;
            float nan_float = extended_25bit_to_float(nan_25bit);
            y[i] = float_to_bf16(nan_float);
            continue;
        }

        float x_float = extended_25bit_to_float(x_25bit[i]);

        // GELU(+∞) = +∞ * σ(+∞) = +∞ * 1 = +∞
        if (x_type == ValueType::PosInf) {
#ifndef __REDUCED_LOG__
            LOG("  +Inf detected → GELU(+∞) = +∞");
#endif
            y[i] = float_to_bf16(INFINITY);
            continue;
        }

        // GELU(-∞) = -∞ * σ(-∞) = -∞ * 0 = NaN (undefined)
        // 但根据 GELU 定义，GELU(-∞) = 0
        if (x_type == ValueType::NegInf) {
#ifndef __REDUCED_LOG__
            LOG("  -Inf detected → GELU(-∞) = 0.0");
#endif
            y[i] = float_to_bf16(0.0f);
            continue;
        }

        // ============================================================
        // Step 1: 计算 1.702 * x
        // ============================================================
        mult_module.reset();
        mult_module.set_input(const_1702_25bit, x_25bit[i]);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t scaled_x_25bit = mult_module.get_result();

        if (i < 5 || i >= len - 2) {
            float scaled_x_float = extended_25bit_to_float(scaled_x_25bit);
#ifndef __REDUCED_LOG__
            LOG("  1.702 * x = " << scaled_x_float);
#endif
        }

        // ============================================================
        // Step 2: 计算 σ(1.702x) = 1 / (1 + exp(-1.702x))
        // ============================================================

        // 数值稳定性检查
        float scaled_x_float = extended_25bit_to_float(scaled_x_25bit);
        float sigmoid_result;

        if (scaled_x_float > 88.0f) {
            // σ(x) ≈ 1.0 (exp underflow)
            sigmoid_result = 1.0f;
#ifndef __REDUCED_LOG__
            if (i < 5 || i >= len - 2) {
                LOG("  σ(1.702x) ≈ 1.0 (x > 88)");
            }
#endif
        } else if (scaled_x_float < -88.0f) {
            // σ(x) ≈ 0.0 (exp overflow)
            sigmoid_result = 0.0f;
#ifndef __REDUCED_LOG__
            if (i < 5 || i >= len - 2) {
                LOG("  σ(1.702x) ≈ 0.0 (x < -88)");
            }
#endif
        } else {
            // 正常情况：计算 -1.702x（翻转符号位）
            uint32_t neg_scaled_x_25bit = scaled_x_25bit ^ (1U << 22);

            // 计算 exp(-1.702x)
            exp_module.reset();
            exp_module.set_input(neg_scaled_x_25bit, true);
            for (int j = 0; j < 8; j++) {
                exp_module.clock_step();
            }
            uint32_t exp_result_25bit = exp_module.get_Exp_Ans_extended();

            // 检查 exp 结果
            ValueType exp_type = classify_25bit(exp_result_25bit);
            if (exp_type == ValueType::PosInf) {
                // exp(-1.702x) = +∞，σ(x) ≈ 0
                sigmoid_result = 0.0f;
#ifndef __REDUCED_LOG__
                if (i < 5 || i >= len - 2) {
                    LOG("  exp overflow → σ(1.702x) ≈ 0.0");
                }
#endif
            } else {
                // 计算 1 + exp(-1.702x)
                fpadd_module.reset();
                fpadd_module.set_input(one_25bit, exp_result_25bit);
                for (int cycle = 0; cycle < 2; cycle++) {
                    fpadd_module.clock_step();
                }
                uint32_t sum_25bit = fpadd_module.get_result_extended();

                // 计算 σ = 1 / (1 + exp(-1.702x))
                float sum_float = extended_25bit_to_float(sum_25bit);
                sigmoid_result = 1.0f / sum_float;

#ifndef __REDUCED_LOG__
                if (i < 5 || i >= len - 2) {
                    LOG("  σ(1.702x) = 1 / " << sum_float << " = " << sigmoid_result);
                }
#endif
            }
        }

        // ============================================================
        // Step 3: 计算 x * σ(1.702x)
        // ============================================================
        uint32_t sigmoid_25bit = float_to_extended_25bit(sigmoid_result);

        mult_module.reset();
        mult_module.set_input(x_25bit[i], sigmoid_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t result_25bit = mult_module.get_result();

        // 输出转换：25-bit → float → bf16
        float result_float = extended_25bit_to_float(result_25bit);
        // y[i] = float_to_bf16(result_float);
        y[i] = extended_25bit_to_bf16(result_25bit);

#ifndef __REDUCED_LOG__
        if (i < 5 || i >= len - 2) {
            LOG("  GELU(x) = " << x_float << " * " << sigmoid_result << " = " << result_float);
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
        }
#endif
    }

#ifndef __REDUCED_LOG__
    LOG("\n========================================");
    LOG("GELU Activation Computation Completed");
    LOG("========================================\n");
#endif
}
