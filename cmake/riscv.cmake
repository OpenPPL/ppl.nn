set(PPLNN_USE_RISCV ON)

set(CMAKE_CXX_FLAGS "-march=rv64gcvxtheadc -mabi=lp64d -mtune=c906 -DRVV_SPEC_0_7 -D__riscv_zfh=1 -static")
set(CMAKE_ASM_FLAGS "-march=rv64gcvxtheadc -mabi=lp64d -mtune=c906 -DRVV_SPEC_0_7 -D__riscv_zfh=1 -static")

file(GLOB __PPLNN_RISCV_SRC__ src/ppl/nn/engines/riscv/*.cc)
file(GLOB_RECURSE __PPLNN_RISCV_SRC_RECURSE__
    src/ppl/nn/engines/riscv/kernels/*.cc
    src/ppl/nn/engines/riscv/optimizer/*.cc
    src/ppl/nn/engines/riscv/params/*.cc
    src/ppl/nn/engines/riscv/utils/*.cc)
add_library(pplnn_riscv_static STATIC ${__PPLNN_RISCV_SRC__} ${__PPLNN_RISCV_SRC_RECURSE__})
unset(__PPLNN_RISCV_SRC_RECURSE__)
unset(__PPLNN_RISCV_SRC__)

hpcc_populate_dep(ppl.riskv.kernel)
target_link_libraries(pplnn_riscv_static PUBLIC pplnn_basic_static pplkernelriscv_static)

target_compile_definitions(pplnn_riscv_static PUBLIC PPLNN_USE_RISCV)

target_link_libraries(pplnn_static INTERFACE pplnn_riscv_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/riscv DESTINATION include/ppl/nn/engines)
    install(TARGETS pplnn_riscv_static DESTINATION lib)
endif()
