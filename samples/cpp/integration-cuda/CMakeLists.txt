cmake_minimum_required(VERSION 3.10)
project(pplnn-integration-cuda)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# cuda env initializations
find_package(CUDA REQUIRED)
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
endif()
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)

# import pplnn
get_filename_component(pplnn_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../pplnn-build/install/lib/cmake/ppl" ABSOLUTE)
# optional: disable unused devices to avoid linking extra deps
#set(PPLNN_USE_X86 OFF)
find_package(pplnn REQUIRED)

# sample target
add_executable(pplnn-integration-cuda main.cc)
target_include_directories(pplnn-integration-cuda PRIVATE
    ${PPLNN_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # imported from `enable_language(CUDA)`
target_link_directories(pplnn-integration-cuda PRIVATE
    ${PPLNN_LINK_DIRS}
    ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES}) # imported from `enable_language(CUDA)`
target_link_libraries(pplnn-integration-cuda PRIVATE ${PPLNN_LIBRARIES})
