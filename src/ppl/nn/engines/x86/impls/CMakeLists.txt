cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(PPLKernelX86)

option(PPL_USE_X86_OMP "Build x86 kernel with openmp support." OFF)
option(PPL_USE_X86_AVX512 "Build x86 kernel with openmp support." ON)

if(MSVC)
set(PPLKERNELX86_COMPILE_OPTIONS )
else()
set(PPLKERNELX86_COMPILE_OPTIONS "-Werror=return-type;-Wno-strict-aliasing")
endif()
set(PPLKERNELX86_COMPILE_DEFINITIONS )
set(PPLKERNELX86_INCLUDE_DIRECTORIES )
set(PPLKERNELX86_LINK_LIBRARIES )

if(HPCC_USE_OPENMP)
    set(PPL_USE_X86_OMP ON)
endif()

if(PPL_USE_X86_OMP)
    FIND_PACKAGE(OpenMP REQUIRED)
    list(APPEND PPLKERNELX86_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    list(APPEND PPLKERNELX86_COMPILE_DEFINITIONS PPL_USE_X86_OMP)
endif()

if(NOT ((CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9.2) OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6.0.0) OR (MSVC_VERSION GREATER 1910)))
    set(PPL_USE_X86_AVX512 OFF) # compiler does not support avx512
endif()

file(GLOB_RECURSE PPLKERNELX86_COMMON_SRC src/ppl/kernel/x86/common/*.cpp)
file(GLOB_RECURSE PPLKERNELX86_FP32_COMMON_SRC src/ppl/kernel/x86/fp32/*_fp32.cpp src/ppl/kernel/x86/fp32/*_fp32_common.cpp)
file(GLOB_RECURSE PPLKERNELX86_FP32_SSE_SRC src/ppl/kernel/x86/fp32/*_fp32_sse.cpp)
file(GLOB_RECURSE PPLKERNELX86_FP32_AVX_SRC src/ppl/kernel/x86/fp32/*_fp32_avx.cpp)
file(GLOB_RECURSE PPLKERNELX86_FP32_FMA_SRC src/ppl/kernel/x86/fp32/*_fp32_fma.cpp)
file(GLOB_RECURSE PPLKERNELX86_FP32_AVX512_SRC src/ppl/kernel/x86/fp32/*_fp32_avx512.cpp)

file(GLOB_RECURSE PPLKERNELX86_BOOL_COMMON_SRC src/ppl/kernel/x86/bool/*_bool.cpp src/ppl/kernel/x86/bool/*_bool_common.cpp)
file(GLOB_RECURSE PPLKERNELX86_BOOL_SSE_SRC src/ppl/kernel/x86/bool/*_bool_sse.cpp)
file(GLOB_RECURSE PPLKERNELX86_BOOL_AVX_SRC src/ppl/kernel/x86/bool/*_bool_avx.cpp)

file(GLOB_RECURSE PPLKERNELX86_INT64_COMMON_SRC src/ppl/kernel/x86/int64/*_int64.cpp src/int64/*_int64_common.cpp)
file(GLOB_RECURSE PPLKERNELX86_INT64_SSE_SRC src/ppl/kernel/x86/int64/*_int64_sse.cpp)
file(GLOB_RECURSE PPLKERNELX86_INT64_AVX_SRC src/ppl/kernel/x86/int64/*_int64_avx.cpp)

set(PPLKERNELX86_SSE_FLAGS )
set(PPLKERNELX86_AVX_FLAGS )
set(PPLKERNELX86_FMA_FLAGS )
set(PPLKERNELX86_AVX512_FLAGS )
if (CMAKE_COMPILER_IS_GNUCC)
    set(PPLKERNELX86_AVX512_FLAGS "-mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
    set(PPLKERNELX86_FMA_FLAGS "-mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
    set(PPLKERNELX86_AVX_FLAGS "-mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
endif()

set_source_files_properties(${PPLKERNELX86_FP32_SSE_SRC} ${PPLKERNELX86_BOOL_SSE_SRC} ${PPLKERNELX86_INT64_SSE_SRC} PROPERTIES
    COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${PPLKERNELX86_SSE_FLAGS}")
set_source_files_properties(${PPLKERNELX86_FP32_AVX_SRC} ${PPLKERNELX86_BOOL_AVX_SRC} ${PPLKERNELX86_INT64_AVX_SRC} PROPERTIES
    COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${AVX_ENABLED_FLAGS} ${PPLKERNELX86_AVX_FLAGS}")
set_source_files_properties(${PPLKERNELX86_FP32_FMA_SRC} PROPERTIES
    COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${AVX_ENABLED_FLAGS} ${FMA_ENABLED_FLAGS} ${PPLKERNELX86_FMA_FLAGS}")
if(PPL_USE_X86_AVX512)
    set_source_files_properties(${PPLKERNELX86_FP32_AVX512_SRC} PROPERTIES
        COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${AVX_ENABLED_FLAGS} ${FMA_ENABLED_FLAGS} ${AVX512_ENABLED_FLAGS} ${PPLKERNELX86_AVX512_FLAGS}")
endif()

set(PPLKERNELX86_SRC
    ${PPLKERNELX86_COMMON_SRC}
    ${PPLKERNELX86_FP32_COMMON_SRC}
    ${PPLKERNELX86_FP32_SSE_SRC}
    ${PPLKERNELX86_FP32_AVX_SRC}
    ${PPLKERNELX86_FP32_FMA_SRC}
    ${PPLKERNELX86_BOOL_COMMON_SRC}
    ${PPLKERNELX86_BOOL_SSE_SRC}
    ${PPLKERNELX86_BOOL_AVX_SRC}
    ${PPLKERNELX86_INT64_COMMON_SRC}
    ${PPLKERNELX86_INT64_SSE_SRC}
    ${PPLKERNELX86_INT64_AVX_SRC})

if (PPL_USE_X86_AVX512)
    list(APPEND PPLKERNELX86_SRC ${PPLKERNELX86_FP32_AVX512_SRC})
endif()

configure_file(include/ppl/kernel/x86/common/config.h.in ${PROJECT_BINARY_DIR}/include/ppl/kernel/x86/common/config.h @ONLY)
list(APPEND PPLKERNELX86_INCLUDE_DIRECTORIES ${PROJECT_BINARY_DIR}/include)

hpcc_populate_dep(ppl.common)
list(APPEND PPLKERNELX86_LINK_LIBRARIES pplcommon_static)

set(PPLNN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)
set(PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES ${PPLNN_SOURCE_DIR}/include ${PPLNN_SOURCE_DIR}/src)

add_library(PPLKernelX86 STATIC ${PPLKERNELX86_SRC})
target_compile_features(PPLKernelX86 PRIVATE cxx_std_11)
target_link_libraries(PPLKernelX86 ${PPLKERNELX86_LINK_LIBRARIES})
set_target_properties(PPLKernelX86 PROPERTIES
    DEBUG_POSTFIX d)
target_include_directories(PPLKernelX86
    PUBLIC include ${PPLKERNELX86_INCLUDE_DIRECTORIES}
    PRIVATE src ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_definitions(PPLKernelX86 PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
target_compile_options(PPLKernelX86 PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})

if(PPLNN_INSTALL)
    install(TARGETS PPLKernelX86 DESTINATION lib)
endif()

################### Test ###################

set(PPLNN_TOOLS_DIR ${PPLNN_SOURCE_DIR}/tools)
add_executable(test_conv2d test/test_conv2d.cpp ${PPLNN_TOOLS_DIR}/simple_flags.cc)
target_include_directories(test_conv2d
    PUBLIC include ${PPLKERNELX86_INCLUDE_DIRECTORIES}
    PRIVATE src ${PPLNN_TOOLS_DIR} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_options(test_conv2d PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})
target_compile_definitions(test_conv2d PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
target_compile_features(test_conv2d PRIVATE cxx_std_11)
target_link_libraries(test_conv2d PRIVATE PPLKernelX86 ${PPLKERNELX86_LINK_LIBRARIES})

add_executable(test_fc test/test_fc.cpp ${PPLNN_TOOLS_DIR}/simple_flags.cc)
target_include_directories(test_fc
    PUBLIC include ${PPLKERNELX86_INCLUDE_DIRECTORIES}
    PRIVATE src ${PPLNN_TOOLS_DIR} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_options(test_fc PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})
target_compile_definitions(test_fc PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
target_compile_features(test_fc PRIVATE cxx_std_11)
target_link_libraries(test_fc PRIVATE PPLKernelX86 ${PPLKERNELX86_LINK_LIBRARIES})

add_executable(test_gemm test/test_gemm.cpp ${PPLNN_TOOLS_DIR}/simple_flags.cc)
target_include_directories(test_gemm
    PUBLIC include ${PPLKERNELX86_INCLUDE_DIRECTORIES}
    PRIVATE src ${PPLNN_TOOLS_DIR} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_options(test_gemm PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})
target_compile_definitions(test_gemm PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
target_compile_features(test_gemm PRIVATE cxx_std_11)
target_link_libraries(test_gemm PRIVATE PPLKernelX86 ${PPLKERNELX86_LINK_LIBRARIES})

add_executable(test_gemm_v2 test/test_gemm_v2.cpp ${PPLNN_TOOLS_DIR}/simple_flags.cc)
target_include_directories(test_gemm_v2
    PUBLIC include ${PPLKERNELX86_INCLUDE_DIRECTORIES}
    PRIVATE src ${PPLNN_TOOLS_DIR} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_options(test_gemm_v2 PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})
target_compile_definitions(test_gemm_v2 PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
target_compile_features(test_gemm_v2 PRIVATE cxx_std_11)
target_link_libraries(test_gemm_v2 PRIVATE PPLKernelX86 ${PPLKERNELX86_LINK_LIBRARIES})
