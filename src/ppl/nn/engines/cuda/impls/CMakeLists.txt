cmake_minimum_required(VERSION 3.14)
project(PPLCUDAKernel)

set(_NVCC_FLAGS )
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "9")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
endif()
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
endif()
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "11")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_NVCC_FLAGS}")
unset(_NVCC_FLAGS)

set(__PPLCUDA_COMPILE_DEFINITIONS__)

# 2spk conv code generator
# idxn conv code generator
# swzl conv code generator
find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "cannot find python3")
endif()

if(PPLNN_ENABLE_CUDA_JIT)
    message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/gene_header.py ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv ${CMAKE_CURRENT_BINARY_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/int8_2spk/gen_2spk_int8_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_2spk
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/int8_idxn/gen_idxn_int8_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_idxn
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/int8_swzl/gen_swzl_int8_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_swzl
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/idxn/gen_idxn_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/swzl/gen_swzl_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

file(GLOB_RECURSE __CUDA_SRC__ tools/*.cu)

if(PPLNN_ENABLE_CUDA_JIT)
    set(__gene_kernel_src__ ${CMAKE_CURRENT_BINARY_DIR}/gene_header.cc src/nn/conv/gene_kernel.cc)
    if(CMAKE_COMPILER_IS_GNUCC)
        set_source_files_properties(${__gene_kernel_src__} PROPERTIES COMPILE_FLAGS -Werror=non-virtual-dtor)
    endif()
    list(APPEND __CUDA_SRC__ ${__gene_kernel_src__})
    unset(__gene_kernel_src__)
else()
    file(GLOB_RECURSE __CUDA_CONV_SRC__
        ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/f*_kernels.cu
        ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/idxn_kernels*.cu
        ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/f*_kernels*.cu
        ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/init*.cu
        ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/init*.cu
        ${CMAKE_CURRENT_BINARY_DIR}/conv/swzl/init*.cu)
    list(APPEND __CUDA_SRC__ ${__CUDA_CONV_SRC__})
    unset(__CUDA_CONV_SRC__)
endif()

if(PPLNN_ENABLE_CUDA_JIT)
    list(APPEND __PPLCUDA_COMPILE_DEFINITIONS__ PPLNN_ENABLE_CUDA_JIT)
endif()

file(GLOB_RECURSE __CUDA_CONV_SRC__
    ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_2spk/int8_f*_kernels.cu
    ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_2spk/init*.cu
    ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_idxn/int8_idxn_kernels*.cu
    ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_idxn/init*.cu
    ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_swzl/int8_f*_kernels.cu
    ${CMAKE_CURRENT_BINARY_DIR}/conv/int8_swzl/init*.cu)
list(APPEND __CUDA_SRC__ ${__CUDA_CONV_SRC__})
unset(__CUDA_CONV_SRC__)

file(GLOB_RECURSE __CUDA_COMMON_SRC__
    src/*.cu
    src/nn/conv/common/init_lut.cc)
list(APPEND __CUDA_SRC__ ${__CUDA_COMMON_SRC__})
unset(__CUDA_COMMON_SRC__)

add_library(pplkernelcuda_static STATIC ${__CUDA_SRC__} ${PPLNN_CUDA_EXTERNAL_SOURCES})
target_compile_features(pplkernelcuda_static PRIVATE cxx_std_11)

get_filename_component(__PPLNN_SOURCE_DIR__ "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES ${__PPLNN_SOURCE_DIR__}/include ${__PPLNN_SOURCE_DIR__}/src)
unset(__PPLNN_SOURCE_DIR__)

target_include_directories(pplkernelcuda_static PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_BINARY_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv
    ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES}
    ${PPLNN_CUDA_EXTERNAL_INCLUDE_DIRECTORIES})
target_link_directories(pplkernelcuda_static PUBLIC ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES} ${PPLNN_CUDA_EXTERNAL_LINK_DIRECTORIES})

# `PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES` is needed for generating pplkernecuda-config.cmake
set(PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES cuda cudart)
if(PPLNN_ENABLE_CUDA_JIT)
    list(APPEND PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES nvrtc)
endif()

target_link_libraries(pplkernelcuda_static PUBLIC ${PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES} ${PPLNN_CUDA_EXTERNAL_LINK_LIBRARIES})

hpcc_populate_dep(pplcommon)
target_link_libraries(pplkernelcuda_static PUBLIC pplcommon_static)

target_compile_definitions(pplkernelcuda_static PUBLIC ${__PPLCUDA_COMPILE_DEFINITIONS__})

if(PPLNN_INSTALL)
    install(TARGETS pplkernelcuda_static DESTINATION lib)

    set(__PPLNN_CMAKE_CONFIG_FILE__ ${CMAKE_CURRENT_BINARY_DIR}/generated/pplkernelcuda-config.cmake)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/pplkernelcuda-config.cmake.in
        ${__PPLNN_CMAKE_CONFIG_FILE__}
        @ONLY)
    install(FILES ${__PPLNN_CMAKE_CONFIG_FILE__} DESTINATION lib/cmake/ppl)
    unset(__PPLNN_CMAKE_CONFIG_FILE__)
endif()

unset(PPLNN_CUDA_TOOLKIT_LINK_LIBRARIES)
unset(__CUDA_SRC__)
