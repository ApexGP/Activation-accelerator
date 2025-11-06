#include <vector>

#include "Arith/FPAdd.h"
#include "Func/EltwiseAdd.h"
#include "Utils/FormatUtils.h"
#include "Utils/Logger.h"
#define __REDUCED_LOG__

void eltwise_add(const std::vector<uint16_t> &x0_bf16, const std::vector<uint16_t> &x1_bf16,
                 std::vector<uint16_t> &y, int len)
{
#ifndef __REDUCED_LOG__
    LOG("========================================");
    LOG("Eltwise Add Computation (25-bit precision)");
    LOG("Formula: y = x0 + x1");
    LOG("Input length: " << len);
    LOG("Algorithm: Unified 25-bit format throughout");
    LOG("========================================");
#endif

    y.resize(len);

    // ============================================================
    // 输入转换：bf16 → float → 25-bit (只转换一次)
    // ============================================================
    std::vector<uint32_t> x0_25bit(len);
    std::vector<uint32_t> x1_25bit(len);

    for (int i = 0; i < len; i++) {
        float x0_float = bf16_to_float(x0_bf16[i]);
        float x1_float = bf16_to_float(x1_bf16[i]);
        x0_25bit[i] = float_to_extended_25bit(x0_float);
        x1_25bit[i] = float_to_extended_25bit(x1_float);
    }

#ifndef __REDUCED_LOG__
    LOG("\nInput converted to 25-bit extended format");
#endif

    // ============================================================
    // 处理每个元素（全程 25-bit）
    // ============================================================
    FPAdd fpadd_module;

    for (int i = 0; i < len; i++) {
#ifndef __REDUCED_LOG__
        LOG("\n[" << i << "] Processing x0[" << i << "] + x1[" << i << "]");
        LOG("  x0 (25-bit): 0x" << std::hex << x0_25bit[i] << std::dec);
        LOG("  x1 (25-bit): 0x" << std::hex << x1_25bit[i] << std::dec);
#endif
        // 使用 classify_25bit 检查特殊值
        ValueType x0_type = classify_25bit(x0_25bit[i]);
        ValueType x1_type = classify_25bit(x1_25bit[i]);

        // 特殊值处理（IEEE 754标准）
        // NaN + anything = NaN
        if (x0_type == ValueType::NaN || x1_type == ValueType::NaN) {
#ifndef __REDUCED_LOG__
            LOG("  ⚠️  NaN detected → result = NaN");
#endif
            // NaN in 25-bit: exn=3, sign=0, exp=0xFF, mantissa!=0
            uint32_t nan_25bit = (3U << 23) | (0xFF << 14) | 0x1;
            float nan_float = extended_25bit_to_float(nan_25bit);
            y[i] = float_to_bf16(nan_float);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // +Inf + (-Inf) = NaN
        if ((x0_type == ValueType::PosInf && x1_type == ValueType::NegInf) ||
            (x0_type == ValueType::NegInf && x1_type == ValueType::PosInf)) {
#ifndef __REDUCED_LOG__
            LOG("  ⚠️  +Inf + (-Inf) → result = NaN");
#endif
            uint32_t nan_25bit = (3U << 23) | (0xFF << 14) | 0x1;
            float nan_float = extended_25bit_to_float(nan_25bit);
            y[i] = float_to_bf16(nan_float);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // +Inf + anything (except -Inf) = +Inf
        if (x0_type == ValueType::PosInf || x1_type == ValueType::PosInf) {
#ifndef __REDUCED_LOG__
            LOG("  +Inf detected → result = +Inf");
#endif
            // +Inf in 25-bit: exn=2, sign=0, exp=0xFF, mantissa=0
            uint32_t pos_inf_25bit = (2U << 23) | (0xFF << 14);
            float pos_inf_float = extended_25bit_to_float(pos_inf_25bit);
            y[i] = float_to_bf16(pos_inf_float);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // -Inf + anything (except +Inf) = -Inf
        if (x0_type == ValueType::NegInf || x1_type == ValueType::NegInf) {
#ifndef __REDUCED_LOG__
            LOG("  -Inf detected → result = -Inf");
#endif
            // -Inf in 25-bit: exn=2, sign=1, exp=0xFF, mantissa=0
            uint32_t neg_inf_25bit = (2U << 23) | (1U << 22) | (0xFF << 14);
            float neg_inf_float = extended_25bit_to_float(neg_inf_25bit);
            y[i] = float_to_bf16(neg_inf_float);
#ifndef __REDUCED_LOG__
            LOG("  Result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif
            continue;
        }

        // 正常情况：使用 FPAdd 硬件模块（25-bit 精度）
        fpadd_module.reset();
        fpadd_module.set_input(x0_25bit[i], x1_25bit[i]);

        for (int cycle = 0; cycle < 2; cycle++) {
            fpadd_module.clock_step();
        }

        uint32_t result_25bit = fpadd_module.get_result_extended();
#ifndef __REDUCED_LOG__
        LOG("  FPAdd result (25-bit): 0x" << std::hex << result_25bit << std::dec);
#endif

        // 输出转换：25-bit → float → bf16
        float result_float = extended_25bit_to_float(result_25bit);
        y[i] = float_to_bf16(result_float);
        // y[i] = extended_25bit_to_bf16(result_25bit);
#ifndef  __REDUCED_LOG__
        LOG("  Final result: 0x" << std::hex << y[i] << std::dec << " (bf16)");
#endif // ! __REDUCED_LOG__
    }

#ifndef  __REDUCED_LOG__
    LOG("\n========================================");
    LOG("Eltwise Add Computation Completed");
    LOG("========================================\n");
#endif // ! __REDUCED_LOG__
}
