if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

# --------------------------------------------------------------------------- #

if(CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
        message(FATAL_ERROR "gcc >= 4.9.0 is required.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0.0)
        message(FATAL_ERROR "clang >= 6.0.0 is required.")
    endif()
endif()

# --------------------------------------------------------------------------- #

if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/opt/libomp/include -Wno-unused-command-line-argument")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/opt/libomp/include -Wno-unused-command-line-argument")
    endif()
endif()

# --------------------------------------------------------------------------- #

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

if(PPLNN_HOLD_DEPS)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# --------------------------------------------------------------------------- #

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(FATAL_ERROR "git is required.")
endif()

set(__HPCC_COMMIT__ e6e445437fba399cc86867536fdf4252bbefae61)

if(PPLNN_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLNN_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    if(NOT PPLNN_DEP_HPCC_GIT)
        set(PPLNN_DEP_HPCC_GIT "https://github.com/openppl-public/hpcc.git")
    endif()
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLNN_DEP_HPCC_GIT}
        GIT_TAG ${__HPCC_COMMIT__}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

unset(__HPCC_COMMIT__)

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable pplcommon tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable pplcommon benchmark")
set(PPLCOMMON_ENABLE_PYTHON_API ${PPLNN_ENABLE_PYTHON_API})
set(PPLCOMMON_ENABLE_LUA_API ${PPLNN_ENABLE_LUA_API})
set(PPLCOMMON_HOLD_DEPS ${PPLNN_HOLD_DEPS})
set(PPLCOMMON_USE_X86_64 ${PPLNN_USE_X86_64})
set(PPLCOMMON_USE_AARCH64 ${PPLNN_USE_AARCH64})
set(PPLCOMMON_USE_ARMV7 ${PPLNN_USE_ARMV7})
set(PPLCOMMON_USE_CUDA ${PPLNN_USE_CUDA})

set(__PPLCOMMON_COMMIT__ 0f01f5913fbb4eb9bde528cf52044b3926688a22)

if(PPLNN_DEP_PPLCOMMON_PKG)
    hpcc_declare_pkg_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_PKG})
else()
    if(NOT PPLNN_DEP_PPLCOMMON_GIT)
        set(PPLNN_DEP_PPLCOMMON_GIT "https://github.com/openppl-public/ppl.common.git")
    endif()
    hpcc_declare_git_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_GIT}
        ${__PPLCOMMON_COMMIT__})
endif()

unset(__PPLCOMMON_COMMIT__)

# --------------------------------------------------------------------------- #

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "disable tests")
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "disable installation")

set(__FLATBUFFERS_TAG__ v2.0.0)

if(PPLNN_DEP_FLATBUFFERS_PKG)
    hpcc_declare_pkg_dep(flatbuffers
        ${PPLNN_DEP_FLATBUFFERS_PKG})
else()
    if(NOT PPLNN_DEP_FLATBUFFERS_GIT)
        set(PPLNN_DEP_FLATBUFFERS_GIT "https://github.com/google/flatbuffers.git")
    endif()
    hpcc_declare_git_dep(flatbuffers
        ${PPLNN_DEP_FLATBUFFERS_GIT}
        ${__FLATBUFFERS_TAG__})
endif()

unset(__FLATBUFFERS_TAG__)

# --------------------------------------------------------------------------- #

set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")
set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "")

if(MSVC)
    if(PPLNN_USE_MSVC_STATIC_RUNTIME)
        set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "")
    else()
        set(protobuf_BUILD_SHARED_LIBS ON CACHE BOOL "")
    endif()
endif()

set(__PROTOBUF_TAG__ v3.11.4)

if(PPLNN_DEP_PROTOBUF_PKG)
    hpcc_declare_pkg_dep(protobuf
        ${PPLNN_DEP_PROTOBUF_PKG})
else()
    if(NOT PPLNN_DEP_PROTOBUF_GIT)
        set(PPLNN_DEP_PROTOBUF_GIT "https://github.com/protocolbuffers/protobuf.git")
    endif()
    hpcc_declare_git_dep(protobuf
        ${PPLNN_DEP_PROTOBUF_GIT}
        ${__PROTOBUF_TAG__})
endif()

unset(__PROTOBUF_TAG__)

# --------------------------------------------------------------------------- #

set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "disable rapidjson tests")
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "disable rapidjson examples")
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "disable rapidjson docs")

set(__RAPIDJSON_COMMIT__ 2e8f5d897d9d461a7273b4b812b0127f321b1dcf)

if(PPLNN_DEP_RAPIDJSON_PKG)
    hpcc_declare_pkg_dep(rapidjson
        ${PPLNN_DEP_RAPIDJSON_PKG})
else()
    if(NOT PPLNN_DEP_RAPIDJSON_GIT)
        set(PPLNN_DEP_RAPIDJSON_GIT "https://github.com/Tencent/rapidjson.git")
    endif()
    hpcc_declare_git_dep(rapidjson
        ${PPLNN_DEP_RAPIDJSON_GIT}
        ${__RAPIDJSON_COMMIT__})
endif()

unset(__RAPIDJSON_COMMIT__)

# --------------------------------------------------------------------------- #

set(PYBIND11_INSTALL OFF CACHE BOOL "disable pybind11 installation")
set(PYBIND11_TEST OFF CACHE BOOL "disable pybind11 tests")
set(PYBIND11_NOPYTHON ON CACHE BOOL "do not find python")
set(PYBIND11_FINDPYTHON OFF CACHE BOOL "do not find python")

set(__PYBIND11_TAG__ v2.9.2)

if(PPLNN_DEP_PYBIND11_PKG)
    hpcc_declare_pkg_dep(pybind11
        ${PPLNN_DEP_PYBIND11_PKG})
else()
    if(NOT PPLNN_DEP_PYBIND11_GIT)
        set(PPLNN_DEP_PYBIND11_GIT "https://github.com/pybind/pybind11.git")
    endif()
    hpcc_declare_git_dep(pybind11
        ${PPLNN_DEP_PYBIND11_GIT}
        ${__PYBIND11_TAG__})
endif()

unset(__PYBIND11_TAG__)

# --------------------------------------------------------------------------- #

set(LUACPP_INSTALL OFF CACHE BOOL "")
set(LUACPP_BUILD_TESTS OFF CACHE BOOL "")

set(__LUACPP_COMMIT__ f381a4702017b61ee9662ae9fa7bceec8b5c7b32)

if(PPLNN_DEP_LUACPP_PKG)
    hpcc_declare_pkg_dep(luacpp
        ${PPLNN_DEP_LUACPP_PKG})
else()
    if(NOT PPLNN_DEP_LUACPP_GIT)
        set(PPLNN_DEP_LUACPP_GIT "https://github.com/ouonline/lua-cpp.git")
    endif()
    hpcc_declare_git_dep(luacpp
        ${PPLNN_DEP_LUACPP_GIT}
        ${__LUACPP_COMMIT__})
endif()

unset(__LUACPP_COMMIT__)

# --------------------------------------------------------------------------- #

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(__GOOGLETEST_TAG__ release-1.10.0)

if(PPLNN_DEP_GOOGLETEST_PKG)
    hpcc_declare_pkg_dep(googletest
        ${PPLNN_DEP_GOOGLETEST_PKG})
else()
    if(NOT PPLNN_DEP_GOOGLETEST_GIT)
        set(PPLNN_DEP_GOOGLETEST_GIT "https://github.com/google/googletest.git")
    endif()
    hpcc_declare_git_dep(googletest
        ${PPLNN_DEP_GOOGLETEST_GIT}
        ${__GOOGLETEST_TAG__})
endif()

unset(__GOOGLETEST_TAG__)
