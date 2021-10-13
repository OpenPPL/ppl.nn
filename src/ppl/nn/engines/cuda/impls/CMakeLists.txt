cmake_minimum_required(VERSION 3.14)
project(PPLCUDAKernel)

set(_NVCC_FLAGS )
if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_NVCC_FLAGS}")

# 2spk conv code generator
# idxn conv code generator
find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "cannot find python3")
endif()
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/2spk/gen_2spk_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv/idxn/gen_idxn_kernel.py ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn
    OUTPUT_VARIABLE conv_kernels_codegen
    OUTPUT_STRIP_TRAILING_WHITESPACE)

file(GLOB_RECURSE CUDA_SRC tools/*.cu)

file(GLOB_RECURSE CUDA_SRC ${CUDA_SRC} ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/f*_kernels.cu
                                       ${CMAKE_CURRENT_BINARY_DIR}/conv/2spk/init*.cu
                                       ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/idxn_kernels*.cu
                                       ${CMAKE_CURRENT_BINARY_DIR}/conv/idxn/init*.cu)

file(GLOB_RECURSE CUDA_SRC ${CUDA_SRC} src/arithmetic/*.cu
                                       src/memory/*.cu
                                       src/reduce/*.cu
                                       src/reformat/*.cu
                                       src/nn/conv/conv.cu
                                       src/nn/conv/common/*.cu
                                       src/nn/conv/common/init_lut.cc
				                       src/unary/*.cu)
file(GLOB CUDA_SRC ${CUDA_SRC} src/nn/*.cu)
file(GLOB CUDA_SRC ${CUDA_SRC} src/nn/depthwise/*.cu)

add_library(pplkernelcuda_static STATIC ${CUDA_SRC})
target_compile_features(pplkernelcuda_static PRIVATE cxx_std_11)

set(PPLNN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)
set(PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES ${PPLNN_SOURCE_DIR}/include ${PPLNN_SOURCE_DIR}/src)

target_include_directories(pplkernelcuda_static PUBLIC
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_BINARY_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/conv
    ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_link_directories(pplkernelcuda_static PUBLIC ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(pplkernelcuda_static PUBLIC cuda ${CUDA_LIBRARIES} ${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES})

hpcc_populate_dep(ppl.common)
target_link_libraries(pplkernelcuda_static PUBLIC pplcommon_static)

set(_EXPECTED_CUDA_HOST_LINK_DIRECTORIES )
foreach(dir ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
    file(RELATIVE_PATH EXPECTED_RELATIVE_PATH ${CUDA_TOOLKIT_ROOT_DIR} ${dir})
    list(APPEND _EXPECTED_CUDA_HOST_LINK_DIRECTORIES ${EXPECTED_RELATIVE_PATH})
endforeach()
set(_EXPECTED_CUDA_HOST_INCLUDE_DIRECTORIES )
foreach(dir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    file(RELATIVE_PATH EXPECTED_RELATIVE_PATH ${CUDA_TOOLKIT_ROOT_DIR} ${dir})
    list(APPEND _EXPECTED_CUDA_HOST_INCLUDE_DIRECTORIES ${EXPECTED_RELATIVE_PATH})
endforeach()

if(PPLNN_INSTALL)
    install(TARGETS pplkernelcuda_static DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/pplkernelcuda-config.cmake DESTINATION lib/cmake/ppl)
endif()
