cmake_minimum_required(VERSION 3.14)
project(ppl.nn)

# --------------------------------------------------------------------------- #

# options

option(PPLNN_BUILD_TESTS "build all tests" ON)
option(PPLNN_BUILD_TOOLS "build tools" ON)
option(PPLNN_BUILD_SAMPLES "build samples" ON)
option(PPLNN_INSTALL "install ppl headers and libs" ON)
option(PPLNN_ENABLE_KERNEL_PROFILING "enable profiling for each kernel" OFF)
option(PPLNN_ENABLE_PYTHON_API "enable python api support" OFF)
option(PPLNN_ENABLE_LUA_API "enable lua api support" OFF)

# --------------------------------------------------------------------------- #

# variables

set(PPLNN_SOURCES)
set(PPLNN_INCLUDE_DIRECTORIES)
set(PPLNN_COMPILE_DEFINITIONS)
set(PPLNN_LINK_LIBRARIES)
set(PPLNN_LINK_DIRECTORIES)

# --------------------------------------------------------------------------- #

# dependencies

include(cmake/deps.cmake)
include(cmake/protobuf.cmake)

hpcc_populate_dep(rapidjson)
hpcc_populate_dep(pplcommon)

# --------------------------------------------------------------------------- #

# framework related

include(cmake/version.cmake)

file(GLOB_RECURSE PPLNN_MODELS_SRC
    src/ppl/nn/models/*.cc)
file(GLOB_RECURSE PPLNN_BUILTIN_ENGINE_SRC
    src/ppl/nn/engines/common/*.cc)
file(GLOB_RECURSE PPLNN_OPUTILS_SRC
    src/ppl/nn/oputils/*.cc)
file(GLOB PPLNN_FRAMEWORK_SRC
    src/ppl/nn/common/*.cc
    src/ppl/nn/engines/*.cc
    src/ppl/nn/ir/*.cc
    src/ppl/nn/optimizers/*.cc
    src/ppl/nn/quantization/*.cc
    src/ppl/nn/runtime/*.cc
    src/ppl/nn/utils/*.cc
    ${PPLNN_OPUTILS_SRC}
    ${PPLNN_BUILTIN_ENGINE_SRC}
    ${PPLNN_MODELS_SRC})
list(APPEND PPLNN_SOURCES ${PPLNN_FRAMEWORK_SRC})

list(APPEND PPLNN_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

list(APPEND PPLNN_LINK_LIBRARIES pplcommon_static libprotobuf)

if(PPLNN_ENABLE_KERNEL_PROFILING)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_ENABLE_KERNEL_PROFILING)
endif()

# --------------------------------------------------------------------------- #

# engines

if(HPCC_USE_X86_64)
    include(cmake/x86.cmake)
endif()

if(HPCC_USE_CUDA)
    include(cmake/cuda.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPLNN_ENABLE_PYTHON_API)
    add_subdirectory(python)
endif()

if(PPLNN_ENABLE_LUA_API)
    add_subdirectory(lua)
endif()

add_library(pplnn_static STATIC ${PPLNN_SOURCES})
target_include_directories(pplnn_static PUBLIC ${PPLNN_INCLUDE_DIRECTORIES})
target_link_directories(pplnn_static PUBLIC ${PPLNN_LINK_DIRECTORIES})
target_link_libraries(pplnn_static PUBLIC ${PPLNN_LINK_LIBRARIES})
target_compile_definitions(pplnn_static PUBLIC ${PPLNN_COMPILE_DEFINITIONS})
target_compile_features(pplnn_static PRIVATE cxx_std_11)

target_include_directories(pplnn_static PRIVATE
    ${rapidjson_SOURCE_DIR}/include
    ${protobuf_SOURCE_DIR}/src)

# --------------------------------------------------------------------------- #

# installation

if(PPLNN_INSTALL)
    install(DIRECTORY include DESTINATION .)
    install(TARGETS pplnn_static DESTINATION lib)

    set(PPLNN_CMAKE_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/pplnn-config.cmake)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/pplnn-config.cmake.in
        ${PPLNN_CMAKE_CONFIG_FILE}
        @ONLY)
    install(FILES ${PPLNN_CMAKE_CONFIG_FILE} DESTINATION lib/cmake/ppl)
    unset(PPLNN_CMAKE_CONFIG_FILE)
endif()

# --------------------------------------------------------------------------- #

# tools, tests and samples

if(PPLNN_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(PPLNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(PPLNN_BUILD_SAMPLES)
    add_subdirectory(samples/cpp)
endif()
