cmake_minimum_required(VERSION 3.14)
project(ppl.nn)

# --------------------------------------------------------------------------- #

# options

option(PPLNN_BUILD_TESTS "build all tests" ON)
option(PPLNN_BUILD_TOOLS "build tools" ON)
option(PPLNN_BUILD_SAMPLES "build samples" ON)
option(PPLNN_INSTALL "install ppl headers and libs" ON)

option(PPLNN_ENABLE_KERNEL_PROFILING "enable profiling for each kernel" OFF)

option(PPLNN_ENABLE_ONNX_MODEL "enable onnx format support" ON)
option(PPLNN_ENABLE_PMX_MODEL "enable pmx format support. pmx format is under heavily developing and should not be used in production environment." OFF)

option(PPLNN_ENABLE_PYTHON_API "enable python api support" OFF)
option(PPLNN_ENABLE_LUA_API "enable lua api support" OFF)

option(PPLNN_HOLD_DEPS "don't update dependencies" OFF)

option(PPLNN_ENABLE_SANITIZE_OPTIONS "use -fsanitize options to check memory errors. Note that this option is only available for GCC and Clang." OFF)

if(MSVC)
    option(PPLNN_USE_MSVC_STATIC_RUNTIME "" ON)
endif()

option(PPLNN_USE_X86_64 "" OFF)
option(PPLNN_USE_AARCH64 "" OFF)
option(PPLNN_USE_ARMV7 "" OFF)
option(PPLNN_USE_RISCV64 "" OFF)
option(PPLNN_USE_CUDA "" OFF)

# deprecated options
if(HPCC_USE_X86_64 AND NOT PPLNN_USE_X86_64)
    message(FATAL_ERROR "`HPCC_USE_X86_64` is deprecated. use `PPLNN_USE_X86_64` instead.")
endif()
if(HPCC_USE_AARCH64 AND NOT PPLNN_USE_AARCH64)
    message(FATAL_ERROR "`HPCC_USE_AARCH64` is deprecated. use `PPLNN_USE_AARCH64` instead.")
endif()
if(HPCC_USE_CUDA AND NOT PPLNN_USE_CUDA)
    message(FATAL_ERROR "`HPCC_USE_CUDA` is deprecated. use `PPLNN_USE_CUDA` instead.")
endif()
if(HPCC_USE_RISCV AND NOT PPLNN_USE_RISCV64)
    message(FATAL_ERROR "`HPCC_USE_RISCV` is deprecated. use `PPLNN_USE_RISCV64` instead.")
endif()

# --------------------------------------------------------------------------- #

if(APPLE)
    message(FATAL_ERROR "MacOS is not supported currently.")
endif()

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

# --------------------------------------------------------------------------- #

# compiler related

if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    hpcc_use_msvc_static_runtime()
else()
    hpcc_use_msvc_dynamic_runtime()
endif()

# --------------------------------------------------------------------------- #

# framework related

include(cmake/rapidjson.cmake)
include(cmake/version.cmake)

set(PPLNN_MODELS_SRC)
if(PPLNN_ENABLE_ONNX_MODEL)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE PPLNN_MODEL_ONNX_SRC src/ppl/nn/models/onnx/*.cc)
    list(APPEND PPLNN_MODELS_SRC ${PPLNN_MODEL_ONNX_SRC})
endif()
if(PPLNN_ENABLE_PMX_MODEL)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_ENABLE_PMX_MODEL)
    file(GLOB_RECURSE PPLNN_MODEL_PMX_SRC src/ppl/nn/models/pmx/*.cc)
    list(APPEND PPLNN_MODELS_SRC ${PPLNN_MODEL_PMX_SRC})
endif()

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

list(APPEND PPLNN_LINK_LIBRARIES pplcommon_static)

if(PPLNN_ENABLE_KERNEL_PROFILING)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_ENABLE_KERNEL_PROFILING)
endif()

# --------------------------------------------------------------------------- #

# engines

if(PPLNN_USE_X86_64)
    include(cmake/x86.cmake)
endif()

if(PPLNN_USE_CUDA)
    include(cmake/cuda.cmake)
endif()

if(PPLNN_USE_RISCV64)
    include(cmake/riscv.cmake)
endif()

if(PPLNN_USE_AARCH64 OR PPLNN_USE_ARMV7)
    include(cmake/arm.cmake)
endif()

hpcc_populate_dep(pplcommon)

# --------------------------------------------------------------------------- #

if(PPLNN_ENABLE_PYTHON_API)
    add_subdirectory(python)
endif()

if(PPLNN_ENABLE_LUA_API)
    add_subdirectory(lua)
endif()

add_library(pplnn_static STATIC ${PPLNN_SOURCES} ${PPLNN_SOURCE_EXTERNAL_SOURCES})
target_include_directories(pplnn_static PUBLIC ${PPLNN_INCLUDE_DIRECTORIES} ${PPLNN_SOURCE_EXTERNAL_INCLUDE_DIRECTORIES})
target_compile_definitions(pplnn_static PUBLIC ${PPLNN_COMPILE_DEFINITIONS} ${PPLNN_SOURCE_EXTERNAL_COMPILE_DEFINITIONS})
target_compile_features(pplnn_static PRIVATE cxx_std_11)
target_link_directories(pplnn_static PUBLIC ${PPLNN_LINK_DIRECTORIES} ${PPLNN_SOURCE_EXTERNAL_LINK_DIRECTORIES})
target_link_libraries(pplnn_static PUBLIC ${PPLNN_LINK_LIBRARIES} ${PPLNN_SOURCE_EXTERNAL_LINK_LIBRARIES})

target_include_directories(pplnn_static PRIVATE
    ${rapidjson_SOURCE_DIR}/include)

if(PPLNN_ENABLE_ONNX_MODEL)
    include(cmake/protobuf.cmake)
    target_link_libraries(pplnn_static PUBLIC libprotobuf)
    target_include_directories(pplnn_static PRIVATE ${protobuf_SOURCE_DIR}/src)
endif()

if(PPLNN_ENABLE_PMX_MODEL)
    hpcc_populate_dep(flatbuffers)
    target_include_directories(pplnn_static PRIVATE ${flatbuffers_SOURCE_DIR}/include)
endif()

# --------------------------------------------------------------------------- #

# installation

if(PPLNN_INSTALL)
    include(cmake/install.cmake)
endif()

# --------------------------------------------------------------------------- #

# tools, tests and samples

if(PPLNN_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(PPLNN_BUILD_SAMPLES)
    add_subdirectory(samples/cpp)
endif()

if(PPLNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
