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

file(GLOB __PPLNN_CUDA_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/cuda/*.cc)
if(PPLNN_SOURCE_EXTERNAL_CUDA_ENGINE_SOURCES)
    list(REMOVE_ITEM __PPLNN_CUDA_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/cuda/default_register_resources.cc)
endif()

file(GLOB_RECURSE __PPLNN_CUDA_SRC_RECURSE__
    src/ppl/nn/engines/cuda/kernels/*.cc
    src/ppl/nn/engines/cuda/optimizer/*.cc
    src/ppl/nn/engines/cuda/params/*.cc
    src/ppl/nn/engines/cuda/pmx/*.cc)

if(PPLNN_ENABLE_CUDA_JIT)
file(GLOB_RECURSE __PPLNN_CUDA_MODULE_SRC_RECURSE__
    src/ppl/nn/engines/cuda/module/*.cc)
    list(APPEND __PPLNN_CUDA_SRC_RECURSE__ ${__PPLNN_CUDA_MODULE_SRC_RECURSE__})
    unset(__PPLNN_CUDA_MODULE_SRC_RECURSE__)
endif()

add_library(pplnn_cuda_static STATIC
    ${__PPLNN_CUDA_SRC__}
    ${__PPLNN_CUDA_SRC_RECURSE__}
    ${PPLNN_SOURCE_EXTERNAL_CUDA_ENGINE_SOURCES})
unset(__PPLNN_CUDA_SRC_RECURSE__)
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

target_link_libraries(pplnn_static INTERFACE pplnn_cuda_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/cuda DESTINATION include/ppl/nn/engines)
    install(TARGETS pplnn_cuda_static DESTINATION lib)
endif()
