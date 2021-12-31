set(CMAKE_CXX_FLAGS "-march=rv64gcvxtheadc -mabi=lp64d -mtune=c906 -DRVV_SPEC_0_7 -D__riscv_zfh=1 -static")
set(CMAKE_ASM_FLAGS "-march=rv64gcvxtheadc -mabi=lp64d -mtune=c906 -DRVV_SPEC_0_7 -D__riscv_zfh=1 -static")

file(GLOB_RECURSE PPLNN_RISCV_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/riscv/*.cc)
list(APPEND PPLNN_SOURCES ${PPLNN_RISCV_SRC})

add_subdirectory(src/ppl/nn/engines/riscv/impls)
list(APPEND PPLNN_LINK_LIBRARIES pplkernelriscv_static)

list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_RISCV)
