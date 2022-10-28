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
add_library(pplnn_cuda_static STATIC ${PPLNN_SOURCE_EXTERNAL_CUDA_ENGINE_SOURCES} ${__PPLNN_CUDA_SRC__})
unset(__PPLNN_CUDA_SRC__)

add_subdirectory(src/ppl/nn/engines/cuda/impls)
target_link_libraries(pplnn_cuda_static PUBLIC
    pplnn_basic_static pplkernelcuda_static ${PPLNN_SOURCE_EXTERNAL_CUDA_LINK_LIBRARIES})
target_include_directories(pplnn_cuda_static PRIVATE
    ${rapidjson_SOURCE_DIR}/include)
target_compile_definitions(pplnn_cuda_static PUBLIC
    PPLNN_USE_CUDA
    PPLNN_CUDACC_VER_MAJOR=${CUDA_VERSION_MAJOR}
    PPLNN_CUDACC_VER_MINOR=${CUDA_VERSION_MINOR})

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/cuda DESTINATION include/ppl/nn/engines)
    install(TARGETS pplnn_cuda_static DESTINATION lib)
endif()
