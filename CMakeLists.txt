cmake_minimum_required(VERSION 3.18)
project(ppl.nn.llm)

option(PPLNN_USE_LLM_CUDA "" OFF)

# refer to ppl.nn for more available options
option(PPLNN_ENABLE_ONNX_MODEL "" ON)
option(PPLNN_ENABLE_PMX_MODEL "" ON)
option(PPLNN_ENABLE_PYTHON_API "" OFF)
option(PPLNN_INSTALL "" ON)

if(MSVC)
    option(PPLNN_USE_MSVC_STATIC_RUNTIME "" ON)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(PPLNN_USE_LLM_CUDA)
    if(NOT CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    set(PPLCOMMON_USE_CUDA ON)
endif()

# --------------------------------------------------------------------------- #

include(cmake/deps.cmake)
string(REPLACE "-Werror=non-virtual-dtor" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# --------------------------------------------------------------------------- #

# compiler related

if(PPLNN_USE_MSVC_STATIC_RUNTIME)
    hpcc_use_msvc_static_runtime()
else()
    hpcc_use_msvc_dynamic_runtime()
endif()

# --------------------------------------------------------------------------- #

set(PPLNN_LLM_SRC)
set(PPLNN_LLM_DEFINITIONS)
set(PPLNN_LLM_LINK_LIBRARIES)
set(PPLNN_LLM_INCLUDE_DIRECTORIES)
set(PPLNN_LLM_LINK_DIRECTORIES)

file(GLOB_RECURSE __EXTRA_SRC__ src/ppl/nn/oputils/*.cc)
list(APPEND PPLNN_LLM_SRC ${__EXTRA_SRC__})
unset(__EXTRA_SRC__)

# --------------------------------------------------------------------------- #

if(PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE __PPLNN_ONNX_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/models/onnx/*.cc)
    list(APPEND PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES ${__PPLNN_ONNX_SRC__})
    unset(__PPLNN_ONNX_SRC__)
endif()

list(APPEND PPLNN_SOURCE_EXTERNAL_SOURCES
    ${PPLNN_LLM_SRC})
list(APPEND PPLNN_SOURCE_EXTERNAL_INCLUDE_DIRECTORIES
    ${PPLNN_LLM_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
list(APPEND PPLNN_SOURCE_EXTERNAL_LINK_DIRECTORIES
    ${PPLNN_LLM_LINK_DIRECTORIES})
list(APPEND PPLNN_SOURCE_EXTERNAL_COMPILE_DEFINITIONS
    ${PPLNN_LLM_DEFINITIONS})
list(APPEND PPLNN_SOURCE_EXTERNAL_LINK_LIBRARIES
    ${PPLNN_LLM_LINK_LIBRARIES})

if(NOT TARGET libprotobuf)
    hpcc_populate_dep(protobuf)
    set(PPL_LLM_PROTOC_EXECUTABLE ${protobuf_BINARY_DIR}/protoc)
endif()

if(NOT PPLNN_ONNX_GENERATED_LIBS)
    if(NOT PPL_LLM_PROTOC_EXECUTABLE)
        message(FATAL_ERROR "`PPL_LLM_PROTOC_EXECUTABLE` is not specified.")
    endif()
    # generate new onnx.pb.* for pplnn
    set(__LLM_GENERATED_DIR__ ${CMAKE_CURRENT_BINARY_DIR}/generated)
    file(MAKE_DIRECTORY ${__LLM_GENERATED_DIR__})

    set(__PROTO_DIR__ ${HPCC_DEPS_DIR}/pplnn/src/ppl/nn/models/onnx/proto)
    set(__ONNX_GENERATED_FILES__ "${__LLM_GENERATED_DIR__}/onnx.pb.h;${__LLM_GENERATED_DIR__}/onnx.pb.cc")
    add_custom_command(
        OUTPUT ${__ONNX_GENERATED_FILES__}
        COMMAND ${PPL_LLM_PROTOC_EXECUTABLE}
        ARGS --cpp_out ${__LLM_GENERATED_DIR__} -I ${__PROTO_DIR__}
        ${__PROTO_DIR__}/onnx.proto
        DEPENDS protoc ${__PROTO_DIR__}/onnx.proto)
    add_library(pplnn_onnx_generated_static STATIC ${__ONNX_GENERATED_FILES__})
    target_link_libraries(pplnn_onnx_generated_static PUBLIC libprotobuf)
    target_include_directories(pplnn_onnx_generated_static PUBLIC ${__LLM_GENERATED_DIR__})
    set(PPLNN_ONNX_GENERATED_LIBS pplnn_onnx_generated_static)

    unset(__ONNX_GENERATED_FILES__)
    unset(__PROTO_DIR__)
    unset(__LLM_GENERATED_DIR__)
endif()

# --------------------------------------------------------------------------- #

# python config before pplnn
if(PPLNN_ENABLE_PYTHON_API)
    include(python/CMakeLists.txt)
endif()

# --------------------------------------------------------------------------- #

hpcc_get_git_info(
    GIT_HASH_OUTPUT PPLNN_COMMIT_STR
    GIT_TAG_OUTPUT PPLNN_VERSION_STR)

if(NOT PPLNN_VERSION_STR)
    set(PPLNN_VERSION_STR "v0.0.0")
endif()

if(NOT PPLNN_COMMIT_STR)
    set(PPLNN_COMMIT_STR "llm-unknown")
else()
    set(PPLNN_COMMIT_STR "llm-${PPLNN_COMMIT_STR}")
endif()

hpcc_populate_dep(pplnn)

# --------------------------------------------------------------------------- #

if(PPLNN_USE_LLM_CUDA)
    include(cmake/llm_cuda.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPLNN_INSTALL)
    include(cmake/install.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPLNN_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
