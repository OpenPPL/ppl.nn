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

option(PPLNN_HOLD_DEPS "don't update dependencies" OFF)

if(MSVC)
    option(PPLNN_USE_MSVC_STATIC_RUNTIME "" ON)
endif()

option(PPLNN_USE_X86_64 "" OFF)
option(PPLNN_USE_AARCH64 "" OFF)
option(PPLNN_USE_ARMV7 "" OFF)
option(PPLNN_USE_RISCV64 "" OFF)
option(PPLNN_USE_CUDA "" OFF)

# deprecated options
if(HPCC_USE_X86_64)
    message(FATAL_ERROR "`HPCC_USE_X86_64` is deprecated. use `PPLNN_USE_X86_64` instead.")
endif()
if(HPCC_USE_AARCH64)
    message(FATAL_ERROR "`HPCC_USE_AARCH64` is deprecated. use `PPLNN_USE_AARCH64` instead.")
endif()
if(HPCC_USE_CUDA)
    message(FATAL_ERROR "`HPCC_USE_CUDA` is deprecated. use `PPLNN_USE_CUDA` instead.")
endif()
if(HPCC_USE_RISCV)
    message(FATAL_ERROR "`HPCC_USE_RISCV` is deprecated. use `PPLNN_USE_RISCV64` instead.")
endif()
if(PPLNN_ENABLE_SANITIZE_OPTIONS)
    message(FATAL_ERROR "`PPLNN_ENABLE_SANITIZE_OPTIONS` is deprecated. use `HPCC_ENABLE_SANITIZE_OPTIONS` instead.")
endif()

# --------------------------------------------------------------------------- #

if(APPLE)
    message(FATAL_ERROR "MacOS is not supported currently.")
endif()

# --------------------------------------------------------------------------- #

# interface lib. MUST be placed before any other modules

add_library(pplnn_static INTERFACE)

# --------------------------------------------------------------------------- #

# dependencies

include(cmake/deps.cmake)
include(cmake/rapidjson.cmake)

# --------------------------------------------------------------------------- #

# compiler related

if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    hpcc_use_msvc_static_runtime()
else()
    hpcc_use_msvc_dynamic_runtime()
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

# pplcommon MUST be placed after engines because engines may set pplcommon options
hpcc_populate_dep(pplcommon)

# --------------------------------------------------------------------------- #

# pplnn basic

# ----- sources begin ----- #

file(GLOB_RECURSE __PPLNN_BUILTIN_ENGINE_SRC__
    src/ppl/nn/engines/common/*.cc)
file(GLOB_RECURSE __PPLNN_OPUTILS_SRC__
    src/ppl/nn/oputils/*.cc)
file(GLOB __PPLNN_SOURCES__
    src/ppl/nn/common/*.cc
    src/ppl/nn/engines/*.cc
    src/ppl/nn/ir/*.cc
    src/ppl/nn/optimizers/*.cc
    src/ppl/nn/quantization/*.cc
    src/ppl/nn/runtime/*.cc
    src/ppl/nn/utils/*.cc
    ${__PPLNN_BUILTIN_ENGINE_SRC__}
    ${__PPLNN_OPUTILS_SRC__})

if(PPLNN_ENABLE_PMX_MODEL)
    file(GLOB_RECURSE __PPLNN_MODEL_PMX_SRC__ src/ppl/nn/models/pmx/*.cc)
    # if external sources are set, remove `default_register_resources.cc`
    if(PPLNN_SOURCE_EXTERNAL_PMX_MODEL_SOURCES)
        list(REMOVE_ITEM __PPLNN_MODEL_PMX_SRC__ src/ppl/nn/models/pmx/default_register_resources.cc)
    endif()
    list(APPEND __PPLNN_SOURCES__ ${__PPLNN_MODEL_PMX_SRC__} ${PPLNN_SOURCE_EXTERNAL_PMX_MODEL_SOURCES})
    unset(__PPLNN_MODEL_PMX_SRC__)
endif()

add_library(pplnn_basic_static STATIC
    ${__PPLNN_SOURCES__}
    ${PPLNN_SOURCE_EXTERNAL_SOURCES})

unset(__PPLNN_SOURCES__)
unset(__PPLNN_OPUTILS_SRC__)
unset(__PPLNN_BUILTIN_ENGINE_SRC__)

# ----- sources end ----- #

if(PPLNN_ENABLE_PMX_MODEL)
    hpcc_populate_dep(flatbuffers)
    target_include_directories(pplnn_basic_static PUBLIC ${flatbuffers_SOURCE_DIR}/include)
    target_compile_definitions(pplnn_basic_static PUBLIC PPLNN_ENABLE_PMX_MODEL)
endif()

target_include_directories(pplnn_basic_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PPLNN_SOURCE_EXTERNAL_INCLUDE_DIRECTORIES})
target_compile_definitions(pplnn_basic_static PUBLIC
    ${PPLNN_SOURCE_EXTERNAL_COMPILE_DEFINITIONS})
target_link_directories(pplnn_basic_static PUBLIC
    ${PPLNN_SOURCE_EXTERNAL_LINK_DIRECTORIES})
target_link_libraries(pplnn_basic_static PUBLIC
    pplcommon_static
    ${PPLNN_SOURCE_EXTERNAL_LINK_LIBRARIES})

target_include_directories(pplnn_basic_static PRIVATE
    ${rapidjson_SOURCE_DIR}/include)

include(cmake/version.cmake)
target_compile_definitions(pplnn_basic_static PUBLIC
    PPLNN_VERSION_MAJOR=${PPLNN_VERSION_MAJOR}
    PPLNN_VERSION_MINOR=${PPLNN_VERSION_MINOR}
    PPLNN_VERSION_PATCH=${PPLNN_VERSION_PATCH}
    PPLNN_COMMIT_STR="${PPLNN_COMMIT_STR}")

if(PPLNN_ENABLE_KERNEL_PROFILING)
    target_compile_definitions(pplnn_basic_static PUBLIC PPLNN_ENABLE_KERNEL_PROFILING)
endif()

target_link_libraries(pplnn_static INTERFACE pplnn_basic_static)

# --------------------------------------------------------------------------- #

# model formats

if(PPLNN_ENABLE_ONNX_MODEL)
    include(cmake/onnx_model.cmake)
endif()

# --------------------------------------------------------------------------- #

# language bindings

if(PPLNN_ENABLE_PYTHON_API)
    add_subdirectory(python)
endif()

# --------------------------------------------------------------------------- #

# installations

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
