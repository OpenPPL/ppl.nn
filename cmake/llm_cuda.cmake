if(NOT TARGET libprotobuf)
    hpcc_populate_dep(protobuf)
endif()

include(${HPCC_DEPS_DIR}/hpcc/cmake/cuda-common.cmake)

hpcc_populate_dep(ppl.llm.kernel.cuda)

file(GLOB_RECURSE __SRC__ src/ppl/nn/engines/llm_cuda/*.cc)
add_library(ppl_llm_cuda_static ${__SRC__})
target_link_libraries(ppl_llm_cuda_static PUBLIC pplnn_basic_static pplkernelcuda_static)
target_compile_definitions(ppl_llm_cuda_static PUBLIC PPLNN_USE_LLM_CUDA)

if(PPLNN_CUDA_ENABLE_NCCL)
    target_compile_definitions(ppl_llm_cuda_static PUBLIC PPLNN_CUDA_ENABLE_NCCL)
endif()

unset(__SRC__)

target_link_libraries(pplnn_static INTERFACE ppl_llm_cuda_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/llm_cuda DESTINATION include/ppl/nn/engines)
    install(TARGETS ppl_llm_cuda_static DESTINATION lib)
endif()
