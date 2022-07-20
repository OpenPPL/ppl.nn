option(PPLNN_ENABLE_CUDA_JIT "enable cuda JIT support" ON)

include(${HPCC_DEPS_DIR}/hpcc/cmake/cuda-common.cmake)

if(CUDA_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "cuda verson [${CUDA_VERSION}] < min required [9.0]")
elseif(CUDA_VERSION VERSION_LESS "10.2")
    message(WARNNING " strongly recommend cuda >= 10.2, now is [${CUDA_VERSION}]")
endif()

# ----- #

if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    hpcc_cuda_use_msvc_static_runtime()
endif()

file(GLOB_RECURSE __PPLNN_CUDA_SRC__ src/ppl/nn/engines/cuda/*.cc)
list(APPEND PPLNN_SOURCES ${__PPLNN_CUDA_SRC__})
unset(__PPLNN_CUDA_SRC__)

add_subdirectory(src/ppl/nn/engines/cuda/impls)
list(APPEND PPLNN_LINK_LIBRARIES pplkernelcuda_static)

list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_CUDA)

list(APPEND PPLNN_COMPILE_DEFINITIONS
    PPLNN_CUDACC_VER_MAJOR=${CUDA_VERSION_MAJOR}
    PPLNN_CUDACC_VER_MINOR=${CUDA_VERSION_MINOR})
