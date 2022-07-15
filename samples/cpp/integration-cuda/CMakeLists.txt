cmake_minimum_required(VERSION 3.10)
project(pplnn-integration-cuda)

get_filename_component(pplnn_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../pplnn-build/install/lib/cmake/ppl" ABSOLUTE)
find_package(pplnn REQUIRED)

find_package(CUDA REQUIRED)

if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)

add_executable(pplnn-integration-cuda main.cc)
target_include_directories(pplnn-integration-cuda PRIVATE
    ${PPLNN_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # imported from `enable_language(CUDA)`
target_link_directories(pplnn-integration-cuda PRIVATE
    ${PPLNN_LINK_DIRS}
    ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES}) # imported from `enable_language(CUDA)`
target_link_libraries(pplnn-integration-cuda PRIVATE ${PPLNN_LIBRARIES})
