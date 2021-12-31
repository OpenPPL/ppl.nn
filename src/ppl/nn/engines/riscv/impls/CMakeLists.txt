cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(PPLKernelRISCV)

# option(PPL_USE_RISCV_OMP "Build RISCV kernel with openmp support." OFF)

set(PPLKERNELRISCV_COMPILE_DEFINITIONS )
set(PPLKERNELRISCV_INCLUDE_DIRECTORIES )
set(PPLKERNELRISCV_LINK_LIBRARIES )

# if(HPCC_USE_OPENMP)
#     set(PPL_USE_RISCV_OMP ON)
# endif()

# if(PPL_USE_RISCV_OMP)
#     FIND_PACKAGE(OpenMP REQUIRED)
#     list(APPEND PPLKERNELRISCV_LINK_LIBRARIES OpenMP::OpenMP_CXX)
#     list(APPEND PPLKERNELRISCV_COMPILE_DEFINITIONS PPL_USE_RISCV_OMP)
# endif()

file(GLOB_RECURSE PPLKERNELRISCV_COMMON_SRC src/ppl/kernel/riscv/common/*.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP32_COMMON_SRC src/ppl/kernel/riscv/fp32/*_fp32.cpp src/ppl/kernel/riscv/fp32/*_fp32_common.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP32_VEC128_SRC src/ppl/kernel/riscv/fp32/*_fp32_vec128.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP16_COMMON_SRC src/ppl/kernel/riscv/fp16/*_fp16.cpp src/ppl/kernel/riscv/fp16/*_fp16_common.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP16_VEC128_SRC src/ppl/kernel/riscv/fp16/*_fp16_vec128.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_INT64_COMMON_SRC src/ppl/kernel/riscv/int64/*_int64.cpp src/ppl/kernel/riscv/int64/*_int64_common.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_INT64_VEC128_SRC src/ppl/kernel/riscv/int64/*_int64_vec128.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP16_ASM_SRC src/ppl/kernel/riscv/fp16/*.S)

set(PPLKERNELRISCV_VEC128_FLAGS )

set_source_files_properties(${PPLKERNELRISCV_FP16_VEC128_SRC} PROPERTIES COMPILE_FLAGS "${VEC128_ENABLED_FLAGS} ${PPLKERNELRISCV_VEC_FLAGS}")

set(PPLKERNELRISCV_SRC
    ${PPLKERNELRISCV_COMMON_SRC}
    ${PPLKERNELRISCV_FP32_COMMON_SRC}
    ${PPLKERNELRISCV_FP32_VEC128_SRC}
    ${PPLKERNELRISCV_FP16_COMMON_SRC}
    ${PPLKERNELRISCV_FP16_VEC128_SRC}
    ${PPLKERNELRISCV_INT64_COMMON_SRC}
    ${PPLKERNELRISCV_INT64_VEC128_SRC}
    ${PPLKERNELRISCV_FP16_ASM_SRC}
)

list(APPEND PPLKERNELRISCV_INCLUDE_DIRECTORIES ${PROJECT_BINARY_DIR}/include)

hpcc_populate_dep(pplcommon)
list(APPEND PPLKERNELRISCV_LINK_LIBRARIES pplcommon_static)

set(PPLNN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)
set(PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES ${PPLNN_SOURCE_DIR}/include ${PPLNN_SOURCE_DIR}/src)

add_library(pplkernelriscv_static STATIC ${PPLKERNELRISCV_SRC})

# set(CMAKE_CXX_FLAGS "-march=rv64gcvxtheadc -mabi=lp64d -mtune=c906 -DRVV_SPEC_0_7 -D__riscv_zfh=1 -static")

target_compile_features(pplkernelriscv_static PRIVATE cxx_std_11)
target_link_libraries(pplkernelriscv_static ${PPLKERNELRISCV_LINK_LIBRARIES})
target_include_directories(pplkernelriscv_static
    PUBLIC include ${PPLKERNELRISCV_INCLUDE_DIRECTORIES}
    PRIVATE src ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_definitions(pplkernelriscv_static PRIVATE ${PPLKERNELRISCV_COMPILE_DEFINITIONS})
target_compile_options(pplkernelriscv_static PRIVATE ${PPLKERNELRISCV_COMPILE_OPTIONS})

if(PPLNN_INSTALL)
    install(TARGETS pplkernelriscv_static DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/pplkernelriscv-config.cmake DESTINATION lib/cmake/ppl)
endif()

################### Test ###################

# set(PPLNN_TOOLS_DIR ${PPLNN_SOURCE_DIR}/tools)
# add_executable(test_conv2d test/test_conv2d.cpp ${PPLNN_TOOLS_DIR}/simple_flags.cc)
# target_include_directories(test_conv2d
#     PUBLIC include ${PPLKERNELRISCV_INCLUDE_DIRECTORIES}
#     PRIVATE src ${PPLNN_TOOLS_DIR} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
# target_compile_options(test_conv2d PRIVATE ${PPLKERNELRISCV_COMPILE_OPTIONS})
# target_compile_definitions(test_conv2d PRIVATE ${PPLKERNELRISCV_COMPILE_DEFINITIONS})
# target_compile_features(test_conv2d PRIVATE cxx_std_11)
# target_link_libraries(test_conv2d PRIVATE pplkernelriscv_static ${PPLKERNELRISCV_LINK_LIBRARIES})