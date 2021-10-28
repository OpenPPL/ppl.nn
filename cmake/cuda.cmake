option(PPLNN_ENABLE_CUDA_JIT "enable cuda JIT support" ON)

# ----- required cuda version >= 10.2 ----- #

find_package(CUDA REQUIRED)
if(CUDA_VERSION VERSION_LESS "10.2")
    message(FATAL_ERROR "cuda verson [${CUDA_VERSION}] < min required [10.2]")
endif()

# ----- #

file(GLOB_RECURSE PPLNN_CUDA_SRC src/ppl/nn/engines/cuda/*.cc)
list(APPEND PPLNN_SOURCES ${PPLNN_CUDA_SRC})

add_subdirectory(src/ppl/nn/engines/cuda/impls)
list(APPEND PPLNN_LINK_LIBRARIES pplkernelcuda_static)

list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_CUDA)
