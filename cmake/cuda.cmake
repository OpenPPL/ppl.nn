option(PPLNN_ENABLE_CUDA_JIT "enable cuda JIT support" ON)

# ----- required cuda version >= 10.2 ----- #

include(${HPCC_DEPS_DIR}/hpcc/cmake/cuda-common.cmake)

if(CUDA_VERSION VERSION_LESS "10.2")
    message(FATAL_ERROR "cuda verson [${CUDA_VERSION}] < min required [10.2]")
endif()

# ----- #

if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    hpcc_cuda_use_msvc_static_runtime()
endif()

file(GLOB_RECURSE PPLNN_CUDA_SRC src/ppl/nn/engines/cuda/*.cc)
list(APPEND PPLNN_SOURCES ${PPLNN_CUDA_SRC})

add_subdirectory(src/ppl/nn/engines/cuda/impls)
list(APPEND PPLNN_LINK_LIBRARIES pplkernelcuda_static)

set(PPLNN_USE_CUDA ON)
list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_CUDA)
