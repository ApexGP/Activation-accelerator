#include "Benchmark.h"
#include "Func/EltwiseAdd.h"
#include "Func/EltwiseMult.h"
#include "Func/GELU.h"
#include "Func/LayerNorm.h"
#include "Func/RMSNorm.h"
#include "Func/SiLU.h"
#include "Func/Sigmoid.h"
#include "Func/Softmax.h"
#include "Utils/Config.h"
#include "Utils/FormatUtils.h"
#include "Utils/InputVec.h"
#include "Utils/Logger.h"
#include <iomanip>
#include <iostream>
#include <vector>
#define __STATIC_INPUT__
#define DEFAULT_N 64
#define DEFAULT_D 768
#define STATIC_INPUT_ROOT "../data/bf16_vectors/"
// Convert bf16 encoded array to float array (similar to reference
// implementation)
void bf16_to_float_array(const std::vector<uint16_t> &in,
                         std::vector<float> &out) {
  out.resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = bf16_to_float(in[i]);
  }
}
void print_benchmark_result(const std::string &module_name,
                            const BenchmarkResult &result) {
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "Module: " << module_name << "\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "Max Error: " << std::scientific << std::setprecision(6)
            << result.max_error << "\n";
  std::cout << "Avg Error: " << std::scientific << std::setprecision(6)
            << result.avg_error << "\n";
  std::cout << "Precision: " << (result.passed ? "? Pass" : "? Fail")
            << " (threshold: 4e-3)\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  return;
  LOG("【" << module_name << "】Precision verification: "
           << (result.passed ? "Pass" : "Fail"));
  LOG("  Max error: " << std::scientific << std::setprecision(6)
                      << result.max_error);
  LOG("  Avg error: " << std::scientific << std::setprecision(6)
                      << result.avg_error);
}
int main(int argc, char *argv[]) {
  // Parse command-line arguments
  Config config;
  if (!config.parse_arguments(argc, argv)) {
    return 0;
  }
  // Initialize logging system (max log size: 5MB before archiving)
  Logger::getInstance().init("log", 10000);
  LOG("\n╔═══════════════════════════════════════════╗");
  LOG("║ Activation Function Precision Benchmark   ║");
  LOG("╚═══════════════════════════════════════════╝\n");
// Test data based on InputVecTutorial.md (N rows for comprehensive testing)
// Using InputVecGenerator for modular test data generation
// Configuration is managed in InputVec.h (default: N=64, D=768)
// For quick functional testing: use InputVecGenerator(64, 1, 2024)
// For full hardware testing: use InputVecGenerator() with defaults (64, 768,
// 2024)
#ifdef __STATIC_INPUT__ // Load input from file
  const int N = DEFAULT_N;
  const int D = DEFAULT_D;
  const int len = DEFAULT_N * DEFAULT_D;
  std::vector<uint16_t> x_bf16(len);
  std::vector<uint16_t> x1_bf16(len);
  // Load pre-generated test data from file
  std::ifstream ifs(STATIC_INPUT_ROOT "X_test_tensor_bf16.bin",
                    std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open test vector file" << std::endl;
    exit(1);
  }
  // Read N*D elements from file
  ifs.read(reinterpret_cast<char *>(x_bf16.data()), len * sizeof(uint16_t));
  ifs.close();
  // Load pre-generated test data from file
  std::ifstream ifs1(STATIC_INPUT_ROOT "Y_test_tensor_bf16.bin",
                     std::ios::binary);
  if (!ifs1.is_open()) {
    std::cerr << "Failed to open test vector file" << std::endl;
    exit(1);
  }
  ifs1.read(reinterpret_cast<char *>(x1_bf16.data()), len * sizeof(uint16_t));
  ifs1.close();
#else  // __STATIC_INPUT__
  InputVecGenerator input_gen;
  // Use default: N=64, D=768, seed=2024
  std::vector<float> x_input;
  std::vector<float> x1_input;
  input_gen.generate(x_input, x1_input);
  // Get configuration from InputVecGenerator
  const int N = input_gen.get_length();
  // Number of test rows
  const int D = input_gen.get_dimension();
  // Dimension per row
  const int len = input_gen.get_total_size();
  // Total elements (N * D)
  std::vector<uint16_t> x_bf16(len);
  std::vector<uint16_t> x1_bf16(len);
  // Note: For D>1, data is row-major flattened
  // Access: row i, col j -> x_input[i * D + j]
  // Convert to bf16
  for (int i = 0; i < len; i++) {
    x_bf16[i] = float_to_bf16(x_input[i]);
    x1_bf16[i] = float_to_bf16(x1_input[i]);
  }
#endif // __STATIC_INPUT__
  std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "Test Data Configuration\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  std::cout << "Matrix size: " << N << " × " << D << " (N×D)\n";
  std::cout << "Total elements: " << len << "\n";
  std::cout << "Input format: bf16 (wE=8, wF=7)\n";
  std::cout << "Precision threshold: 4e-3 (BF16 quantization tolerance)\n";
  std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  int total_tests = 0;
  int passed_tests = 0;
  // ================== Test 1: Element-wise Addition ==================
  if (config.should_run_module(CONFIG_ELTWISE_ADD)) {
    total_tests++;
    // Select appropriate test rows for EltwiseAdd
    std::vector<int> test_rows = select_test_rows(CONFIG_ELTWISE_ADD);
    LOG("EltwiseAdd test using "
        << test_rows.size()
        << " rows: general(0-19) + specific(46-53) + random(54-63)");
    // Process each test row independently (avoid NaN propagation across rows)
    std::vector<float> result_add;
    std::vector<float> result_add_ref;
    result_add.reserve(test_rows.size() * D);
    result_add_ref.reserve(test_rows.size() * D);
#ifdef __STATIC_INPUT__
    // Load pre-generated result from file
    std::ifstream ifs(STATIC_INPUT_ROOT "ref_add_bf16.bin", std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open test vector file" << std::endl;
      exit(1);
    }
    std::vector<uint16_t> interm_bf16(len);  // Answers that not been sliced
    ifs.read(reinterpret_cast<char *>(interm_bf16.data()), len * sizeof(uint16_t));
#endif
    for (int row : test_rows) {
        // Extract single row (D elements)
      std::vector<uint16_t> x_row_bf16(D);
      std::vector<uint16_t> x1_row_bf16(D);
      std::vector<float> x_row(D);
      std::vector<float> x1_row(D);
      int row_offset = row * D;
      for (int col = 0; col < D; col++) {
        x_row_bf16[col] = x_bf16[row_offset + col];
        x1_row_bf16[col] = x1_bf16[row_offset + col];
#ifndef __STATIC_INPUT__
        x_row[col] = x_input[row_offset + col];
        x1_row[col] = x1_input[row_offset + col];
#endif
      }
      // Custom implementation (process single row)
      std::vector<uint16_t> result_row_bf16;
      eltwise_add(x_row_bf16, x1_row_bf16, result_row_bf16, D);
      std::vector<float> result_row_ref(D);
#ifndef __STATIC_INPUT__
      // Standard library reference (process single row)
      eltwise_add_reference(x_row, x1_row, result_row_ref);
#else
      for (int col = 0; col < D; col++) {
        result_row_ref[col] = bf16_to_float(interm_bf16[D * row + col]);
      }
#endif
      // Convert custom result to float and append
      for (int col = 0; col < D; col++) {
        result_add.push_back(bf16_to_float(result_row_bf16[col]));
        result_add_ref.push_back(result_row_ref[col]);
      }
    }
    // Compare bf16-truncated values directly
    BenchmarkResult bench_result = verify_accuracy(result_add, result_add_ref, test_rows, D);
    print_benchmark_result("Element-wise Addition", bench_result);
    if (bench_result.passed)
      passed_tests++;
#ifdef __STATIC_INPUT__
    ifs.close();
#endif
  }
  // ================== Test 2: Element-wise Multiplication ==================
  if (config.should_run_module(CONFIG_ELTWISE_MULT)) {
    total_tests++;
    // Select appropriate test rows for EltwiseMult
    std::vector<int> test_rows = select_test_rows(CONFIG_ELTWISE_MULT);
    LOG("EltwiseMult test using "
        << test_rows.size()
        << " rows: general(0-19) + specific(46-53) + random(54-63)");
    // Process each test row independently (avoid NaN propagation across rows)
    std::vector<float> result_mult;
    std::vector<float> result_mult_ref;
    result_mult.reserve(test_rows.size() * D);
    result_mult_ref.reserve(test_rows.size() * D);
#ifdef __STATIC_INPUT__
    std::ifstream ifs(STATIC_INPUT_ROOT "ref_mul_bf16.bin", std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open test vector file" << std::endl;
      exit(1);
    }
    std::vector<uint16_t> interm_bf16(len); // Answers that not been sliced
    ifs.read(reinterpret_cast<char *>(interm_bf16.data()), len * sizeof(uint16_t));
#endif
    for (int row : test_rows) {
      // Extract single row (D elements)
      std::vector<uint16_t> x_row_bf16(D);
      std::vector<uint16_t> x1_row_bf16(D);
      std::vector<float> x_row(D);
      std::vector<float> x1_row(D);
      int row_offset = row * D;
      for (int col = 0; col < D; col++) {
        x_row_bf16[col] = x_bf16[row_offset + col];
        x1_row_bf16[col] = x1_bf16[row_offset + col];
#ifndef __STATIC_INPUT__
        x_row[col] = x_input[row_offset + col];
        x1_row[col] = x1_input[row_offset + col];
#endif
      }
      // Custom implementation (process single row)
      std::vector<uint16_t> result_row_bf16;
      eltwise_mult(x_row_bf16, x1_row_bf16, result_row_bf16, D);
      // Standard library reference (process single row)
      std::vector<float> result_row_ref(D);
#ifndef __STATIC_INPUT__
      eltwise_mult_reference(x_row, x1_row, result_row_ref);
#else
      for (int col = 0; col < D; col++) {
        result_row_ref[col] = bf16_to_float(interm_bf16[row * D + col]);
      }
#endif
		for (int i = 0; i < D; i++) {
			result_mult.push_back(bf16_to_float(result_row_bf16[i]));
			result_mult_ref.push_back(result_row_ref[i]);
		}
	}
      // Compare bf16-truncated values directly
      BenchmarkResult bench_result =
          verify_accuracy(result_mult, result_mult_ref, test_rows, D);
      print_benchmark_result("Element-wise Multiplication", bench_result);
      if (bench_result.passed)
        passed_tests++;
#ifdef __STATIC_INPUT__
      ifs.close();
#endif
  }
  // ================== Test 3: SiLU Activation ==================
  if (config.should_run_module(CONFIG_SILU)) {
    total_tests++;
    // Select appropriate test rows for Sigmoid (use SiLU rows)
    std::vector<int> test_rows = select_test_rows(CONFIG_SILU);
    LOG("SiLU test using "
        << test_rows.size()
        << " rows: general(0-19) + SiLU-like(38-41) + random(54-63)");
    // Process each test row independently (avoid NaN propagation across rows)
    std::vector<float> result_sigmoid;
    std::vector<float> result_sigmoid_ref;
    result_sigmoid.reserve(test_rows.size() * D);
    result_sigmoid_ref.reserve(test_rows.size() * D);
#ifdef __STATIC_INPUT__
    std::ifstream ifs(STATIC_INPUT_ROOT "ref_silu_bf16.bin",
                      std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open test vector file" << std::endl;
      exit(1);
    }
    std::vector<uint16_t> interm_bf16(len);  // Answers that not been sliced
    ifs.read(reinterpret_cast<char *>(interm_bf16.data()), len * sizeof(uint16_t));
#endif
    for (int row : test_rows) {
      // Extract single row (D elements)
      std::vector<uint16_t> x_row_bf16(D);
      std::vector<float> x_row(D);
      int row_offset = row * D;
      for (int col = 0; col < D; col++) {
        x_row_bf16[col] = x_bf16[row_offset + col];
#ifndef __STATIC_INPUT__
        x_row[col] = x_input[row_offset + col];
#endif
      }
      // Custom implementation (process single row)
      std::vector<uint16_t> result_row_bf16;
      silu_activation(x_row_bf16, result_row_bf16, D);
      // Standard library reference (process single row)
      std::vector<float> result_row_ref(D);
#ifndef __STATIC_INPUT__
      sigmoid_reference(x_row, result_row_ref);
#else
      for (int col = 0; col < D; col++) {
          result_row_ref[col] = bf16_to_float(interm_bf16[D * row + col]);
      }
#endif
      // Convert custom result to float and append
      for (int col = 0; col < D; col++) {
        result_sigmoid.push_back(bf16_to_float(result_row_bf16[col]));
        result_sigmoid_ref.push_back(result_row_ref[col]);
      }
    }
    // Compare bf16-truncated values directly
    BenchmarkResult bench_result =
        verify_accuracy(result_sigmoid, result_sigmoid_ref, test_rows, D);
    print_benchmark_result("SiLU Activation", bench_result);
    if (bench_result.passed)
      passed_tests++;
#ifdef __STATIC_INPUT__
    ifs.close();
#endif
  }
  // ================== Test 4: RMS Normalization ==================
  if (config.should_run_module(CONFIG_RMS_NORM)) {
    total_tests++;
    // Select appropriate test rows for RMSNorm
    std::vector<int> test_rows = select_test_rows(CONFIG_RMS_NORM);
    LOG("RMSNorm test using "
        << test_rows.size()
        << " rows: general(0-19) + specific(32-37) + random(54-63)");
    // Process each test row independently (RMSNorm operates on vectors, not
    // concatenated data)
    std::vector<float> result_rms;
    std::vector<float> result_rms_ref;
    result_rms.reserve(test_rows.size() * D);
    result_rms_ref.reserve(test_rows.size() * D);
#ifdef __STATIC_INPUT__
    std::ifstream ifs(STATIC_INPUT_ROOT "ref_rmsnorm_bf16.bin", std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open test vector file" << std::endl;
      exit(1);
    }
    std::vector<uint16_t> interm_bf16(len);  // Answers that not been sliced
    ifs.read(reinterpret_cast<char *>(interm_bf16.data()), len * sizeof(uint16_t));
#endif
    for (int row : test_rows) {
      // Extract single row (D elements)
      std::vector<uint16_t> x_row_bf16(D);
      std::vector<float> x_row(D);
      int row_offset = row * D;
      for (int col = 0; col < D; col++) {
        x_row_bf16[col] = x_bf16[row_offset + col];
#ifndef __STATIC_INPUT__
        x_row[col] = x_input[row_offset + col];
#endif
      }
      // Custom implementation (process single row)
      std::vector<uint16_t> result_row_bf16;
      rms_norm(x_row_bf16, result_row_bf16, D);
      // Standard library reference (process single row)
      std::vector<float> result_row_ref(D);
#ifdef __STATIC_INPUT__
      for (int col = 0; col < D; col++) {
          result_row_ref[col] = bf16_to_float(interm_bf16[D * row + col]);
      }
#else
      rms_norm_reference(x_row, result_row_ref);
#endif
      // Convert custom result to float and append
      if (result_row_bf16.size() != D) {
        std::cerr << "  [ERROR] Row " << row
                  << ": result_row_bf16.size() = " << result_row_bf16.size()
                  << " (expected " << D << ")\n";
      }
      for (int col = 0; col < D; col++) {
        result_rms.push_back(bf16_to_float(result_row_bf16[col]));
        result_rms_ref.push_back(result_row_ref[col]);
      }
    }
#ifdef __STATIC_INPUT__
    ifs.close();
#endif
    // Compare bf16-truncated values directly
    BenchmarkResult bench_result = verify_accuracy(result_rms, result_rms_ref, test_rows, D);
    print_benchmark_result("RMS Normalization", bench_result);
    if (bench_result.passed)
      passed_tests++;
  }
  // ================== Test 5: Layer Normalization ==================
  if (config.should_run_module(CONFIG_LAYER_NORM)) {
    total_tests++;
    // Select appropriate test rows for LayerNorm
    std::vector<int> test_rows = select_test_rows(CONFIG_LAYER_NORM);
    LOG("LayerNorm test using "
        << test_rows.size()
        << " rows: general(0-19) + specific(26-31) + random(54-63)");
    // Process each test row independently (LayerNorm operates on vectors, not
    // concatenated data)
    std::vector<float> result_layer;
    std::vector<float> result_layer_ref;
    result_layer.reserve(test_rows.size() * D);
    result_layer_ref.reserve(test_rows.size() * D);
#ifdef __STATIC_INPUT__
    std::ifstream ifs(STATIC_INPUT_ROOT "ref_layernorm_bf16.bin", std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open test vector file" << std::endl;
      exit(1);
    }
    std::vector<uint16_t> interm_bf16(len);  // Answers that not been sliced
    ifs.read(reinterpret_cast<char *>(interm_bf16.data()), len * sizeof(uint16_t));
#endif
    for (int row : test_rows) {
      // Extract single row (D elements)
      std::vector<uint16_t> x_row_bf16(D);
      std::vector<float> x_row(D);
      int row_offset = row * D;
      for (int col = 0; col < D; col++) {
        x_row_bf16[col] = x_bf16[row_offset + col];
#ifndef __STATIC_INPUT__
        x_row[col] = x_input[row_offset + col];
#endif
      }
      // Custom implementation (process single row)
      std::vector<uint16_t> result_row_bf16;
      layer_norm(x_row_bf16, result_row_bf16, D);
      // Standard library reference (process single row)
      std::vector<float> result_row_ref(D);
#ifdef __STATIC_INPUT__
      for (int col = 0; col < D; col++) {
          result_row_ref[col] = bf16_to_float(interm_bf16[D * row + col]);
      }
#else
      layer_norm_reference(x_row, result_row_ref);
#endif
      // Convert custom result to float and append
      for (int col = 0; col < D; col++) {
        result_layer.push_back(bf16_to_float(result_row_bf16[col]));
        result_layer_ref.push_back(result_row_ref[col]);
      }
    }
#ifdef __STATIC_INPUT__
    ifs.close();
#endif
    // Compare bf16-truncated values directly
    BenchmarkResult bench_result = verify_accuracy(result_layer, result_layer_ref, test_rows, D);
    print_benchmark_result("Layer Normalization", bench_result);
    if (bench_result.passed)
      passed_tests++;
  }
  // ================== Test 6: Online Softmax ==================
  if (config.should_run_module(CONFIG_ONLINE_SOFTMAX)) {
    total_tests++;
    // Select appropriate test rows for Softmax
    std::vector<int> test_rows = select_test_rows(CONFIG_ONLINE_SOFTMAX);
    LOG("Softmax test using "
        << test_rows.size()
        << " rows: general(0-19) + specific(20-25) + random(54-63)");
    // Process each test row independently (Softmax operates on vectors, not
    // concatenated data)
    std::vector<float> result_softmax;
    std::vector<float> result_softmax_ref;
    result_softmax.reserve(test_rows.size() * D);
    result_softmax_ref.reserve(test_rows.size() * D);
#ifdef __STATIC_INPUT__
    std::ifstream ifs(STATIC_INPUT_ROOT "ref_softmax_bf16.bin",
                      std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open test vector file" << std::endl;
      exit(1);
    }
    std::vector<uint16_t> interm_bf16(len);  // Answers that not been sliced
    ifs.read(reinterpret_cast<char *>(interm_bf16.data()), len * sizeof(uint16_t));
#endif
    for (int row : test_rows) {
      // Extract single row (D elements)
      std::vector<uint16_t> x_row_bf16(D);
      std::vector<float> x_row(D);
      int row_offset = row * D;
      for (int col = 0; col < D; col++) {
        x_row_bf16[col] = x_bf16[row_offset + col];
#ifndef __STATIC_INPUT__
        x_row[col] = x_input[row_offset + col];
#endif
      }
      // Custom implementation (process single row)
      std::vector<uint16_t> result_row_bf16;
      online_softmax(x_row_bf16, result_row_bf16);
      // Use default exp_output_bits=14
      // Standard library reference (process single row)
      std::vector<float> result_row_ref(D);
#ifdef __STATIC_INPUT__
      for (int col = 0; col < D; col++) {
          result_row_ref[col] = bf16_to_float(interm_bf16[D * row + col]);
      }
#else
      online_softmax_reference(x_row, result_row_ref);
#endif
      // Convert custom result to float and append
      for (int col = 0; col < D; col++) {
        result_softmax.push_back(bf16_to_float(result_row_bf16[col]));
        result_softmax_ref.push_back(result_row_ref[col]);
      }
    }
#ifdef __STATIC_INPUT__
    ifs.close();
#endif
    // Compare bf16-truncated values directly
    BenchmarkResult bench_result =
        verify_accuracy(result_softmax, result_softmax_ref, test_rows, D);
    print_benchmark_result("Online Softmax", bench_result);
    if (bench_result.passed)
      passed_tests++;
  }
  // ================== Test 7: GELU Activation ==================
  if (config.should_run_module(CONFIG_GELU)) {
    total_tests++;
    // Select appropriate test rows for GELU
    std::vector<int> test_rows = select_test_rows(CONFIG_GELU);
    LOG("GELU test using "
        << test_rows.size()
        << " rows: general(0-19) + specific(42-45) + random(54-63)");
    // Process each test row independently (avoid NaN propagation across rows)
    std::vector<float> result_gelu;
    std::vector<float> result_gelu_ref;
    result_gelu.reserve(test_rows.size() * D);
    result_gelu_ref.reserve(test_rows.size() * D);
#ifdef __STATIC_INPUT__
    std::ifstream ifs(STATIC_INPUT_ROOT "ref_gelu_bf16.bin", std::ios::binary);
    if (!ifs.is_open()) {
      std::cerr << "Failed to open test vector file" << std::endl;
      exit(1);
    }
    std::vector<uint16_t> interm_bf16(len);  // Answers that not been sliced
    ifs.read(reinterpret_cast<char *>(interm_bf16.data()), len * sizeof(uint16_t));
#endif
    for (int row : test_rows) {
      // Extract single row (D elements)
      std::vector<uint16_t> x_row_bf16(D);
      std::vector<float> x_row(D);
      int row_offset = row * D;
      for (int col = 0; col < D; col++) {
        x_row_bf16[col] = x_bf16[row_offset + col];
#ifndef __STATIC_INPUT__
        x_row[col] = x_input[row_offset + col];
#endif
      }
      // Custom implementation (process single row)
      std::vector<uint16_t> result_row_bf16;
      gelu_activation(x_row_bf16, result_row_bf16, D);
      // Standard library reference (process single row)
      std::vector<float> result_row_ref(D);
#ifdef __STATIC_INPUT__
      for (int col = 0; col < D; col++) {
          result_row_ref[col] = bf16_to_float(interm_bf16[D * row + col]);
      }
#else
      gelu_reference(x_row, result_row_ref);
#endif
      // Convert custom result to float and append
      for (int col = 0; col < D; col++) {
        result_gelu.push_back(bf16_to_float(result_row_bf16[col]));
        result_gelu_ref.push_back(result_row_ref[col]);
      }
    }
#ifdef __STATIC_INPUT__
    ifs.close();
#endif
    // Compare bf16-truncated values directly
    BenchmarkResult bench_result =
        verify_accuracy(result_gelu, result_gelu_ref, test_rows, D);
    print_benchmark_result("GELU Activation", bench_result);
    if (bench_result.passed)
      passed_tests++;
  }
  LOG("\n╔═══════════════════════════════════════════╗");
  LOG("║    Precision Verification Summary         ║");
  LOG("╚═══════════════════════════════════════════╝");
  LOG("Total tests: " << total_tests);
  LOG("Passed: " << passed_tests);
  LOG("Failed: " << (total_tests - passed_tests));
  LOG("Pass rate: " << std::fixed << std::setprecision(1)
                    << (total_tests > 0 ? (100.0 * passed_tests / total_tests)
                                        : 0.0)
                    << "%\n");
  Logger::getInstance().finalize();
  return (passed_tests == total_tests) ? 0 : 1;
}