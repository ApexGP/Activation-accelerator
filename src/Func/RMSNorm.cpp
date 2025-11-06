#include <vector>

#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Arith/FPSqrt.h"
#include "Func/RMSNorm.h"
#include "Utils/ConstDiv.h"
#include "Utils/FormatUtils.h"
#include "Utils/Logger.h"
#define __REDUCED_LOG__

void rms_norm(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len, float eps)
{
#ifndef __REDUCED_LOG__
    LOG("========================================");
    LOG("RMS Normalization Computation (25-bit precision)");
    LOG("Formula: y[i] = x[i] / sqrt(mean(x^2) + eps)");
    LOG("Input length: " << len);
    LOG("Algorithm: Unified 25-bit format throughout");
    LOG("eps = " << eps);
    LOG("========================================");
#endif

    y.resize(len);

    // ============================================================
    // 输入转换：bf16 → float → 25-bit (只转换一次)
    // ============================================================
    std::vector<uint32_t> x_25bit(len);
    bool has_special = false;

    for (int i = 0; i < len; i++) {
        float x_float = bf16_to_float(x_bf16[i]);
        x_25bit[i] = float_to_extended_25bit(x_float);

        // 检查特殊值
        ValueType x_type = classify_25bit(x_25bit[i]);
        if (x_type == ValueType::NaN || x_type == ValueType::PosInf ||
            x_type == ValueType::NegInf) {
            has_special = true;
#ifndef __REDUCED_LOG__
            LOG("\n⚠️  Special value detected at index [" << i << "]");
#endif
            break;
        }
    }

#ifndef __REDUCED_LOG__
    LOG("\nInput converted to 25-bit extended format");
#endif

    // 如果有特殊值，输出全 NaN
    if (has_special) {
#ifndef __REDUCED_LOG__
        LOG("→ Input contains Inf/NaN, returning all NaN");
#endif
        uint32_t nan_25bit = (3U << 23) | (0xFF << 14) | 0x1;
        float nan_float = extended_25bit_to_float(nan_25bit);
        uint16_t nan_bf16 = float_to_bf16(nan_float);

        for (int i = 0; i < len; i++) {
            y[i] = nan_bf16;
        }
#ifndef __REDUCED_LOG__
        LOG("\n========================================");
        LOG("RMS Normalization Computation Completed (Special Case)");
        LOG("========================================\n");
#endif
        return;
    }

    // ============================================================
    // Stage 1: 计算平方和的均值 - 树状求和优化
    // 对于 D=768 = 3 × 2^8，使用树状结构减少累积误差
    // 公式: mean(x^2) = (s1 + s2 + s3) / 3
    // 其中 s1, s2, s3 是通过 8 层二分求和得到的三个子和
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Stage 1: Compute mean(x^2) using tree-based summation (768 = 3 × 2^8) ---");
#endif

    FPMult mult_module;
    FPAdd fadd_module;
    uint32_t mean_sq_25bit;

    // 为 768 = 3 × 256 进行优化
    if (len == 768) {
        // 首先计算所有 x[i]^2
        std::vector<uint32_t> x_sq_25bit(len);
        for (int i = 0; i < len; i++) {
            mult_module.set_input(x_25bit[i], x_25bit[i]);
            mult_module.clock_step();
            mult_module.clock_step();
            x_sq_25bit[i] = mult_module.get_result();

            if (i < 3 || i >= len - 2) {
                float x_val = extended_25bit_to_float(x_25bit[i]);
                float x_sq = extended_25bit_to_float(x_sq_25bit[i]);
#ifndef __REDUCED_LOG__
                LOG("  x[" << i << "]^2 = " << x_val << "^2 = " << x_sq);
#endif
            } else if (i == 3) {
#ifndef __REDUCED_LOG__
                LOG("  ...");
#endif
            }
        }

        std::vector<uint32_t> temp = x_sq_25bit;  // 复制平方值

        // 进行 8 层二分求和 (768 → 384 → 192 → 96 → 48 → 24 → 12 → 6 → 3)
        for (int level = 0; level < 8; level++) {
            int pairs = temp.size() / 2;
            std::vector<uint32_t> next_level;
            next_level.reserve(pairs + (temp.size() % 2));

            // 每对相加
            for (int i = 0; i < pairs; i++) {
                fadd_module.reset();
                fadd_module.set_input(temp[2 * i], temp[2 * i + 1]);
                fadd_module.clock_step();
                fadd_module.clock_step();
                uint32_t pair_sum = fadd_module.get_result_extended();

                // 右移 1 位 (除以2，通过指数减1实现)
                // 25-bit 格式: [exn:2][sign:1][exp:8][mantissa:14]
                uint32_t exn = (pair_sum >> 23) & 0x3;
                uint32_t sign = (pair_sum >> 22) & 0x1;
                uint32_t exp = (pair_sum >> 14) & 0xFF;
                uint32_t mantissa = pair_sum & 0x3FFF;

                // 除以2: exp -= 1 (只对正常数处理)
                if (exn == 1 && exp > 0) {  // 正常数且指数不为0
                    exp -= 1;
                    uint32_t halved = (exn << 23) | (sign << 22) | (exp << 14) | mantissa;
                    next_level.push_back(halved);
                } else {
                    // 特殊值或零，保持不变
                    next_level.push_back(pair_sum);
                }
            }

            // 如果有奇数个元素，最后一个直接传递（不需要除以2）
            if (temp.size() % 2 == 1) {
                next_level.push_back(temp.back());
            }

            temp = next_level;

#ifndef __REDUCED_LOG__
            if (level == 7) {
                LOG("  After 8 levels of pairwise summation, remaining " << temp.size()
                                                                         << " values:");
                for (size_t i = 0; i < temp.size(); i++) {
                    LOG("    s" << (i + 1) << " = " << extended_25bit_to_float(temp[i]));
                }
            }
#endif
        }

        // 现在 temp 应该有 3 个元素 (因为 768 = 3 × 256)
        if (temp.size() != 3) {
#ifndef __REDUCED_LOG__
            LOG("⚠️  Warning: Expected 3 elements after tree reduction, got " << temp.size());
#endif
        }

        // 将最后 3 个元素相加
        fadd_module.reset();
        fadd_module.set_input(temp[0], temp[1]);
        fadd_module.clock_step();
        fadd_module.clock_step();
        uint32_t sum_01 = fadd_module.get_result_extended();

        fadd_module.reset();
        fadd_module.set_input(sum_01, temp[2]);
        fadd_module.clock_step();
        fadd_module.clock_step();
        uint32_t sum_sq_25bit = fadd_module.get_result_extended();

        // 乘以 1/3 (使用 ConstDiv 查找表)
        uint32_t inv_3_25bit = ConstDiv::get_inv_k_25bit(3);

        mult_module.reset();
        mult_module.set_input(sum_sq_25bit, inv_3_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        mean_sq_25bit = mult_module.get_result();

        float sum_sq_float = extended_25bit_to_float(sum_sq_25bit);
        float mean_sq_float = extended_25bit_to_float(mean_sq_25bit);

#ifndef __REDUCED_LOG__
        LOG("  final_sum (s1+s2+s3) = " << sum_sq_float);
        LOG("  mean(x^2) = " << mean_sq_float << " (using tree-based summation + 1/3 LUT)");
#endif

    } else {
        // 对于非 768 的情况，使用传统顺序累加
#ifndef __REDUCED_LOG__
        LOG("  (Using sequential summation for len=" << len << ")");
#endif

        uint32_t sum_sq_25bit = 0U;  // Zero in 25-bit: exn=0

        for (int i = 0; i < len; i++) {
            // 计算 x[i]^2 (使用 FPMult)
            mult_module.set_input(x_25bit[i], x_25bit[i]);
            mult_module.clock_step();
            mult_module.clock_step();
            uint32_t x_sq_25bit = mult_module.get_result();

            // 累加到 sum_sq (使用 FPAdd)
            fadd_module.reset();
            fadd_module.set_input(sum_sq_25bit, x_sq_25bit);
            fadd_module.clock_step();
            fadd_module.clock_step();
            sum_sq_25bit = fadd_module.get_result_extended();
        }

        float sum_sq_float = extended_25bit_to_float(sum_sq_25bit);
#ifndef __REDUCED_LOG__
        LOG("  sum(x^2) = " << sum_sq_float);
#endif

        // 计算均值
        uint32_t inv_len_25bit = ConstDiv::get_inv_k_25bit(len);
        mult_module.reset();
        mult_module.set_input(sum_sq_25bit, inv_len_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        mean_sq_25bit = mult_module.get_result();

        float mean_sq_float = extended_25bit_to_float(mean_sq_25bit);
#ifndef __REDUCED_LOG__
        LOG("  mean(x^2) = " << mean_sq_float << " (using ConstDiv)");
#endif
    }

    // ============================================================
    // Stage 2: 计算 RMS_inv = 1/sqrt(mean(x^2) + eps) (使用 FPSqrt)
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Stage 2: Compute 1/sqrt(mean_sq + eps) using FPSqrt ---");
#endif

    // 计算 mean_sq + eps
    float mean_sq = extended_25bit_to_float(mean_sq_25bit);
    float var_eps = mean_sq + eps;
    uint32_t var_eps_25bit = float_to_extended_25bit(var_eps);

    // 计算 1/sqrt(variance) (FPSqrt 是状态机驱动)
    FPSqrt sqrt_module;
    sqrt_module.set_input(var_eps_25bit);
    while (!sqrt_module.get_valid_out()) {
        sqrt_module.clock_step();
    }
    uint32_t rms_inv_25bit = sqrt_module.get_result_extended();

    float rms_inv = extended_25bit_to_float(rms_inv_25bit);
#ifndef __REDUCED_LOG__
    LOG("  rms_inv = 1/sqrt(" << mean_sq << " + " << eps << ") = " << rms_inv);
#endif

    // ============================================================
    // Stage 4: 归一化 y[i] = x[i] * rms_inv（使用 FPMult）
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Stage 4: Normalization using FPMult ---");
#endif

    for (int i = 0; i < len; i++) {
        // 计算 x[i] * rms_inv (使用 FPMult)
        mult_module.reset();
        mult_module.set_input(x_25bit[i], rms_inv_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t result_25bit = mult_module.get_result();

        // 输出转换：25-bit → float → bf16
        float result_float = extended_25bit_to_float(result_25bit);
        y[i] = float_to_bf16(result_float);
        // y[i] = extended_25bit_to_bf16(result_25bit);

        if (i < 5 || i >= len - 2) {
            float x_float = extended_25bit_to_float(x_25bit[i]);
#ifndef __REDUCED_LOG__
            LOG("  y[" << i << "] = " << x_float << " * " << rms_inv << " = 0x" << std::hex << y[i]
                       << std::dec << " (bf16)");
#endif
        } else if (i == 5) {
#ifndef __REDUCED_LOG__
            LOG("  ...");
#endif
        }
    }

#ifndef __REDUCED_LOG__
    LOG("\n========================================");
    LOG("RMS Normalization Computation Completed");
    LOG("========================================\n");
#endif
}
