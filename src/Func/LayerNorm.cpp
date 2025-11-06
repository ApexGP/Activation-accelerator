#include <vector>

#include "Arith/FPAdd.h"
#include "Arith/FPMult.h"
#include "Arith/FPSqrt.h"
#include "Arith/FPSub.h"
#include "Func/LayerNorm.h"
#include "Utils/ConstDiv.h"
#include "Utils/FormatUtils.h"
#include "Utils/Logger.h"
#define __REDUCED_LOG__

void layer_norm(const std::vector<uint16_t> &x_bf16, std::vector<uint16_t> &y, int len, float eps)
{
#ifndef __REDUCED_LOG__
    LOG("========================================");
    LOG("Layer Normalization Computation (Optimized Algorithm)");
    LOG("Formula: y[i] = (x[i] - mean) / sqrt(variance + eps)");
    LOG("Input length: " << len);
    LOG("Algorithm:");
    LOG("  - Mean: Tree-based summation (768 = 3 * 2^8)");
    LOG("  - Variance: Welford's online algorithm");
    LOG("eps = " << eps);
    LOG("========================================");
#endif

    y.resize(len);

    // ============================================================
    // 输入转换：bf16 → 25-bit (只转换一次)
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
        LOG("Layer Normalization Computation Completed (Special Case)");
        LOG("========================================\n");
#endif
        return;
    }

    // ============================================================
    // Stage 1: 计算均值 - 树状求和优化
    // 对于 D=768 = 3 × 2^8，使用树状结构减少累积误差
    // 公式: μ = (s1 + s2 + s3) / 3
    // 其中 s1, s2, s3 是通过 8 层二分求和得到的三个子和
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Stage 1: Compute mean using tree-based summation (768 = 3 × 2^8) ---");
#endif

    FPAdd fadd_module;
    FPMult mult_module;
    uint32_t mean_25bit;

    // 为 768 = 3 × 256 进行优化
    if (len == 768) {
        std::vector<uint32_t> temp = x_25bit;  // 复制输入

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
        uint32_t sum_25bit = fadd_module.get_result_extended();

        // 乘以 1/3 (使用 ConstDiv 查找表)
        uint32_t inv_3_25bit = ConstDiv::get_inv_k_25bit(3);

        mult_module.reset();
        mult_module.set_input(sum_25bit, inv_3_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        mean_25bit = mult_module.get_result();

        float sum_float = extended_25bit_to_float(sum_25bit);
        float mean_float = extended_25bit_to_float(mean_25bit);

#ifndef __REDUCED_LOG__
        LOG("  final_sum (s1+s2+s3) = " << sum_float);
        LOG("  mean = " << mean_float << " (using tree-based summation + 1/3 LUT)");
#endif

    } else {
        // 对于非 768 的情况，使用传统顺序累加
#ifndef __REDUCED_LOG__
        LOG("  (Using sequential summation for len=" << len << ")");
#endif

        uint32_t sum_25bit = 0U;
        for (int i = 0; i < len; i++) {
            fadd_module.reset();
            fadd_module.set_input(sum_25bit, x_25bit[i]);
            fadd_module.clock_step();
            fadd_module.clock_step();
            sum_25bit = fadd_module.get_result_extended();
        }

        uint32_t inv_len_25bit = ConstDiv::get_inv_k_25bit(len);
        mult_module.reset();
        mult_module.set_input(sum_25bit, inv_len_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        mean_25bit = mult_module.get_result();

        float sum_float = extended_25bit_to_float(sum_25bit);
        float mean_float = extended_25bit_to_float(mean_25bit);

#ifndef __REDUCED_LOG__
        LOG("  sum = " << sum_float);
        LOG("  mean = " << mean_float << " (using ConstDiv FP32-based LUT)");
#endif
    }

    // ============================================================
    // Stage 2: 计算方差 - 传统两次遍历方法
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Stage 2: Compute variance using traditional method ---");
#endif

    FPSub fpsub_module;
    uint32_t var_sum_25bit = 0U;  // 初始化为 0

    for (int j = 0; j < len; j++) {
        // 计算 x[j] - mean
        fpsub_module.reset();
        fpsub_module.set_input(x_25bit[j], mean_25bit);
        for (int cycle = 0; cycle < 2; cycle++) {
            fpsub_module.clock_step();
        }
        uint32_t diff_25bit = fpsub_module.get_result_extended();

        // 计算 (x[j] - mean)^2
        mult_module.reset();
        mult_module.set_input(diff_25bit, diff_25bit);
        mult_module.clock_step();
        mult_module.clock_step();
        uint32_t diff_sq_25bit = mult_module.get_result();

        // 累加到 var_sum
        fadd_module.reset();
        fadd_module.set_input(var_sum_25bit, diff_sq_25bit);
        for (int cycle = 0; cycle < 2; cycle++) {
            fadd_module.clock_step();
        }
        var_sum_25bit = fadd_module.get_result_extended();

        if (j < 5 || j >= len - 2) {
            float x_val = extended_25bit_to_float(x_25bit[j]);
            float diff_val = extended_25bit_to_float(diff_25bit);
            float diff_sq = extended_25bit_to_float(diff_sq_25bit);
            float var_sum = extended_25bit_to_float(var_sum_25bit);
#ifndef __REDUCED_LOG__
            LOG("  x[" << j << "]=" << x_val << ", x-mean=" << diff_val
                       << ", (x-mean)^2=" << diff_sq << ", var_sum=" << var_sum);
#endif
        } else if (j == 5) {
#ifndef __REDUCED_LOG__
            LOG("  ...");
#endif
        }
    }

    // variance = var_sum / D (使用 ConstDiv FP32-based 查找表)
    uint32_t inv_len_25bit_var = ConstDiv::get_inv_k_25bit(len);

    mult_module.reset();
    mult_module.set_input(var_sum_25bit, inv_len_25bit_var);
    mult_module.clock_step();
    mult_module.clock_step();
    uint32_t variance_25bit = mult_module.get_result();

    float variance_float = extended_25bit_to_float(variance_25bit);
#ifndef __REDUCED_LOG__
    LOG("  final variance = " << variance_float << " (using ConstDiv FP32-based LUT)");
#endif

    // ============================================================
    // Stage 3: 计算标准差倒数 std_dev_inv = 1/sqrt(variance + eps)
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Stage 3: Compute 1/sqrt(variance + eps) using FPSqrt ---");
#endif

    // variance + eps
    uint32_t eps_25bit = float_to_extended_25bit(eps);
    fadd_module.reset();
    fadd_module.set_input(variance_25bit, eps_25bit);
    for (int cycle = 0; cycle < 2; cycle++) {
        fadd_module.clock_step();
    }
    uint32_t variance_plus_eps = fadd_module.get_result_extended();

    float var_eps_float = extended_25bit_to_float(variance_plus_eps);
#ifndef __REDUCED_LOG__
    LOG("  variance + eps = " << var_eps_float);
#endif

    // 计算 1/sqrt(variance + eps) 使用 FPSqrt 模块
    FPSqrt sqrt_module;
    sqrt_module.set_input(variance_plus_eps);

    // FPSqrt 需要多个周期（状态机驱动）
    while (!sqrt_module.get_valid_out()) {
        sqrt_module.clock_step();
    }

    uint32_t std_dev_inv_25bit = sqrt_module.get_result_extended();
    float std_dev_inv_float = extended_25bit_to_float(std_dev_inv_25bit);
#ifndef __REDUCED_LOG__
    LOG("  1/sqrt(variance + eps) = " << std_dev_inv_float);
#endif

    // ============================================================
    // Stage 4: 归一化 y[i] = (x[i] - mean) * std_dev_inv
    // ============================================================
#ifndef __REDUCED_LOG__
    LOG("\n--- Stage 4: Normalization ---");
#endif

    for (int i = 0; i < len; i++) {
        // 计算 x[i] - mean (使用树形求和的均值)
        fpsub_module.reset();
        fpsub_module.set_input(x_25bit[i], mean_25bit);

        for (int cycle = 0; cycle < 2; cycle++) {
            fpsub_module.clock_step();
        }

        uint32_t diff_25bit = fpsub_module.get_result_extended();

        // 计算 (x[i] - mean) * std_dev_inv
        mult_module.reset();
        mult_module.set_input(diff_25bit, std_dev_inv_25bit);
        mult_module.clock_step();
        mult_module.clock_step();

        uint32_t result_25bit = mult_module.get_result();

        // 输出转换：25-bit → float → bf16
        float result_float = extended_25bit_to_float(result_25bit);
        y[i] = float_to_bf16(result_float);
        // y[i] = extended_25bit_to_bf16(result_25bit);

#ifndef __REDUCED_LOG__
        if (i < 5 || i >= len - 2) {
            LOG("  y[" << i << "] = " << result_float << " = 0x" << std::hex << y[i] << std::dec
                       << " (bf16)");
        } else if (i == 5) {
            LOG("  ...");
        }
#endif
    }

#ifndef __REDUCED_LOG__
    LOG("\n========================================");
    LOG("Layer Normalization Computation Completed");
    LOG("========================================\n");
#endif
}
