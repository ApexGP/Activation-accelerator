cmake_minimum_required(VERSION 3.10)
project(ActivationAccelerator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCES
    ${PROJECT_SOURCE_DIR}/src/Arith/Analysis.cpp
    ${PROJECT_SOURCE_DIR}/src/Arith/Exp.cpp
    ${PROJECT_SOURCE_DIR}/src/Arith/FPAdd.cpp
    ${PROJECT_SOURCE_DIR}/src/Arith/FPSub.cpp
    ${PROJECT_SOURCE_DIR}/src/Arith/FPMult.cpp
    ${PROJECT_SOURCE_DIR}/src/Arith/FPDiv.cpp
    ${PROJECT_SOURCE_DIR}/src/Arith/FPSqrt.cpp
    ${PROJECT_SOURCE_DIR}/src/Benchmark.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/Sigmoid.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/SiLU.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/GELU.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/EltwiseAdd.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/EltwiseMult.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/RMSNorm.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/LayerNorm.cpp
    ${PROJECT_SOURCE_DIR}/src/Func/Softmax.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/Config.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/ConstDiv.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/FormatUtils.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/InputVec.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/Logger.cpp
)

add_library(ARITH STATIC ${SOURCES})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(main
    ${PROJECT_SOURCE_DIR}/main.cpp
)

target_link_libraries(main ARITH)

