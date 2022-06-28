cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(PPLKernelRISCV)

# option(PPL_USE_RISCV_OMP "Build RISCV kernel with openmp support." OFF)

set(PPLKERNELRISCV_COMPILE_DEFINITIONS )
set(PPLKERNELRISCV_INCLUDE_DIRECTORIES )
set(PPLKERNELRISCV_LINK_LIBRARIES )

file(GLOB_RECURSE PPLKERNELRISCV_COMMON_SRC src/ppl/kernel/riscv/common/*.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP32_COMMON_SRC src/ppl/kernel/riscv/fp32/*_fp32.cpp src/ppl/kernel/riscv/fp32/*_fp32_common.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP32_VEC128_SRC src/ppl/kernel/riscv/fp32/*_fp32_vec128.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP16_COMMON_SRC src/ppl/kernel/riscv/fp16/*_fp16.cpp src/ppl/kernel/riscv/fp16/*_fp16_common.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP16_VEC128_SRC src/ppl/kernel/riscv/fp16/*_fp16_vec128.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_INT64_COMMON_SRC src/ppl/kernel/riscv/int64/*_int64.cpp src/ppl/kernel/riscv/int64/*_int64_common.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_INT64_VEC128_SRC src/ppl/kernel/riscv/int64/*_int64_vec128.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_FP16_ASM_SRC src/ppl/kernel/riscv/fp16/*.S)
file(GLOB_RECURSE PPLKERNELRISCV_BOOL_COMMON_SRC src/ppl/kernel/riscv/bool/*_bool.cpp src/ppl/kernel/riscv/bool/*_bool_common.cpp)
file(GLOB_RECURSE PPLKERNELRISCV_BOOL_VEC128_SRC src/ppl/kernel/riscv/bool/*_bool_vec128.cpp)

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
    ${PPLKERNELRISCV_BOOL_COMMON_SRC}
    ${PPLKERNELRISCV_BOOL_VEC128_SRC}
    ${PPLKERNELRISCV_FP16_ASM_SRC}
)

list(APPEND PPLKERNELRISCV_INCLUDE_DIRECTORIES ${PROJECT_BINARY_DIR}/include)

hpcc_populate_dep(pplcommon)
list(APPEND PPLKERNELRISCV_LINK_LIBRARIES pplcommon_static)

set(__PPLNN_SOURCE_DIR__ ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)
set(PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES ${__PPLNN_SOURCE_DIR__}/include ${__PPLNN_SOURCE_DIR__}/src)
unset(__PPLNN_SOURCE_DIR__)

add_library(pplkernelriscv_static STATIC ${PPLKERNELRISCV_SRC})

# set(CMAKE_CXX_FLAGS "-march=rv64gcvxtheadc -mabi=lp64d -mtune=c906 -DRVV_SPEC_0_7 -D__riscv_zfh=1 -static")

target_compile_features(pplkernelriscv_static PRIVATE cxx_std_11)
target_link_libraries(pplkernelriscv_static ${PPLKERNELRISCV_LINK_LIBRARIES})
target_include_directories(pplkernelriscv_static
    PUBLIC include ${PPLKERNELRISCV_INCLUDE_DIRECTORIES}
    PRIVATE src ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_definitions(pplkernelriscv_static PRIVATE ${PPLKERNELRISCV_COMPILE_DEFINITIONS})
target_compile_options(pplkernelriscv_static PRIVATE ${PPLKERNELRISCV_COMPILE_OPTIONS})

if(CMAKE_COMPILER_IS_GNUCC)
    # requires a virtual destructor when having virtual functions, though it is not necessary for all cases
    target_compile_options(pplkernelriscv_static PRIVATE -Werror=non-virtual-dtor)
endif()

if(PPLNN_INSTALL)
    install(TARGETS pplkernelriscv_static DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/pplkernelriscv-config.cmake DESTINATION lib/cmake/ppl)
endif()
