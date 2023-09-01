hpcc_populate_dep(ppl.llm.kernel.cuda)

if(CUDA_VERSION VERSION_LESS "11.4")
    message(FATAL_ERROR "cuda verson [${CUDA_VERSION}] < min required [11.4]. >= 11.6 is recommended.")
endif()

file(GLOB_RECURSE __PPLNN_INTERNAL_CUDA_SRC__
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/cuda/*.cc)
list(APPEND PPLNN_SOURCE_EXTERNAL_CUDA_ENGINE_SOURCES ${__PPLNN_INTERNAL_CUDA_SRC__})
unset(__PPLNN_INTERNAL_CUDA_SRC__)
