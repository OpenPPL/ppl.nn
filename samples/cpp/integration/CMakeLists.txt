cmake_minimum_required(VERSION 3.10)
project(pplnn-integration)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

get_filename_component(pplnn_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../pplnn-build/install/lib/cmake/ppl" ABSOLUTE)
# optional: disable unused devices to avoid linking extra deps
#set(PPLNN_USE_CUDA OFF)
find_package(pplnn REQUIRED)

add_executable(pplnn-integration main.cc)
target_include_directories(pplnn-integration PRIVATE ${PPLNN_INCLUDE_DIRS})
target_link_directories(pplnn-integration PRIVATE ${PPLNN_LINK_DIRS})
target_link_libraries(pplnn-integration PRIVATE ${PPLNN_LIBRARIES})
