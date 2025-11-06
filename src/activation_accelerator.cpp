#include "Func/EltwiseAdd.h"
#include "Func/EltwiseMult.h"
#include "Func/GELU.h"
#include "Func/LayerNorm.h"
#include "Func/RMSNorm.h"
#include "Func/SiLU.h"
#include "Func/Softmax.h"
#include "Utils/Config.h"
#include "activation_accelerator.h"

/*
 * 0 Eltwise Add
 * 1 Eltwise Mult
 * 2 SiLU
 * 3 RMS Norm
 * 4 Layer Norm
 * 5 Softmax
 * 6 GELU
 */

/** 
 * @brief Activation Accelerator
 *
 * This function is used to accelerate the activation function which process 
 * data in a row every time. For example, if you want to activate Softmax which
 * receives input from Row 0-25 and 54-63, you need to call this function 36 times.
 *
 * @param in0 Input 0
 * @param in1 Input 1
 * @param out Output
 * @param stage Stage
 * @param config Configuration
 * 
 */
void activation_accelerator(uint16_t *in0, uint16_t *in1, uint16_t *out, uint32_t stage,
                            uint32_t config)
{
// HLS pragma directives for Vitis HLS synthesis
#pragma HLS INTERFACE m_axi port = in0 offset = slave bundle = gmem0 depth = 770
#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem1 depth = 770
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem2 depth = 770
#pragma HLS INTERFACE s_axilite port = stage
#pragma HLS INTERFACE s_axilite port = config
#pragma HLS INTERFACE s_axilite port = return

    static uint16_t buffer0_bf16[DIMENSION + 2];
    static uint16_t buffer1_bf16[DIMENSION + 2];
    static uint16_t x_bf16[DIMENSION];
    static uint16_t x1_bf16[DIMENSION];
    static uint16_t y_bf16[DIMENSION];

    if (stage == 0) {
        // Stage 0: Load data from PS to PL
        for (int i = 0; i < DIMENSION; i++) {
#pragma HLS PIPELINE II = 1
            buffer0_bf16[i] = in0[i];
            buffer1_bf16[i] = in1[i];
        }
    } else if (stage == 1) {
        // Stage 1: Compute

        for (int i = 0; i < DIMENSION; i++) {
#pragma HLS PIPELINE II = 1
            x_bf16[i] = buffer0_bf16[i];
            x1_bf16[i] = buffer1_bf16[i];
        }

        switch (config) {
            case 0:
                eltwise_add(x_bf16, x1_bf16, y_bf16, DIMENSION);
                break;

            case 1:
                eltwise_mult(x_bf16, x1_bf16, y_bf16, DIMENSION);
                break;

            case 2:
                silu(x_bf16, y_bf16, DIMENSION);
                break;

            case 3:
                rms_norm(x_bf16, y_bf16, DIMENSION);
                break;

            case 4:
                layer_norm(x_bf16, y_bf16, DIMENSION);
                break;

            case 5:
                online_softmax(x_bf16, y_bf16, DIMENSION);
                break;

            case 6:
                gelu(x_bf16, y_bf16, DIMENSION);
                break;

            default:
                break;
        }

    } else if (stage == 2) {
        // Stage 2: Load data from PL to PS
        for (int i = 0; i < DIMENSION; i++) {
#pragma HLS PIPELINE II = 1
            out[i] = y_bf16[i];
        }
    }
}
