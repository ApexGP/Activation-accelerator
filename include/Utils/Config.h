#ifndef CONFIG_H
#define CONFIG_H

// Module configuration
#define CONFIG_ELTWISE_ADD 0
#define CONFIG_ELTWISE_MULT 1
#define CONFIG_SILU 2
#define CONFIG_RMS_NORM 3
#define CONFIG_LAYER_NORM 4
#define CONFIG_ONLINE_SOFTMAX 5
#define CONFIG_GELU 6

// Data Input Vector - N * D
#define ROW 64
#define DIMENSION 768

// Max Capacity for HlsVector
#define MAX_CAPACITY 770

// Stage Definition
#define STAGE_LOAD 0     // Data loading stage
#define STAGE_COMPUTE 1  // Computation stage
#define STAGE_STORE 2    // Data storage stage

#endif  // CONFIG_H
