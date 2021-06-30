if (NOT HPCC_USE_CUDA AND NOT WITH_CUDA)
    return()
endif()

# ----- required cuda version >= 10.2 ----- #

find_package(CUDA REQUIRED)
if(CUDA_VERSION VERSION_LESS "10.2")
    message(FATAL_ERROR "cuda verson [${CUDA_VERSION}] < min required [10.2]")
endif()

# ----- #

file(GLOB_RECURSE PPLNN_CUDA_SRC src/ppl/nn/engines/cuda/*.cc)
list(APPEND PPLNN_SOURCES ${PPLNN_CUDA_SRC})

add_subdirectory(src/ppl/nn/engines/cuda/impls)
list(APPEND PPLNN_LINK_LIBRARIES PPLCUDAKernel)

list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_CUDA)

# ----- install cuda engine ----- #

file(GLOB PPLNN_CUDA_PUBLIC_HEADERS
    src/ppl/nn/engines/cuda/cuda_options.h
    src/ppl/nn/engines/cuda/engine_factory.h)
install(FILES ${PPLNN_CUDA_PUBLIC_HEADERS}
    DESTINATION include/ppl/nn/engines/cuda)
