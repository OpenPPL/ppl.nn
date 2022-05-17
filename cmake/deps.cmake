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

set(__HPCC_COMMIT__ 266579679761dd2c37440c700fb5602187056fce)

if(PPLNN_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLNN_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
elseif(PPLNN_DEP_HPCC_GIT)
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLNN_DEP_HPCC_GIT}
        GIT_TAG ${__HPCC_COMMIT__}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    FetchContent_Declare(hpcc
        URL https://github.com/openppl-public/hpcc/archive/${__HPCC_COMMIT__}.zip
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

set(__PPLCOMMON_COMMIT__ d2096bc104172b7c853d680e3b2b9c5981a13df7)

if(PPLNN_DEP_PPLCOMMON_PKG)
    hpcc_declare_pkg_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_PKG})
elseif(PPLNN_DEP_PPLCOMMON_GIT)
    hpcc_declare_git_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_GIT}
        ${__PPLCOMMON_COMMIT__})
else()
    hpcc_declare_pkg_dep(pplcommon
        https://github.com/openppl-public/ppl.common/archive/${__PPLCOMMON_COMMIT__}.zip)
endif()

unset(__PPLCOMMON_COMMIT__)

# --------------------------------------------------------------------------- #

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "disable tests")
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "disable installation")

set(__FLATBUFFERS_TAG__ v2.0.0)

if(PPLNN_DEP_FLATBUFFERS_PKG)
    hpcc_declare_pkg_dep(flatbuffers
        ${PPLNN_DEP_FLATBUFFERS_PKG})
elseif(PPLNN_DEP_FLATBUFFERS_GIT)
    hpcc_declare_git_dep(flatbuffers
        ${PPLNN_DEP_FLATBUFFERS_GIT}
        ${__FLATBUFFERS_TAG__})
else()
    hpcc_declare_pkg_dep(flatbuffers
        https://github.com/google/flatbuffers/archive/refs/tags/${__FLATBUFFERS_TAG__}.zip)
endif()

unset(__FLATBUFFERS_TAG__)

# --------------------------------------------------------------------------- #

set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")
set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "")

set(__PROTOBUF_TAG__ v3.12.4)

if(PPLNN_DEP_PROTOBUF_PKG)
    hpcc_declare_pkg_dep(protobuf
        ${PPLNN_DEP_PROTOBUF_PKG})
elseif(PPLNN_DEP_PROTOBUF_GIT)
    hpcc_declare_git_dep(protobuf
        ${PPLNN_DEP_PROTOBUF_GIT}
        ${__PROTOBUF_TAG__})
else()
    hpcc_declare_pkg_dep(protobuf
        https://github.com/protocolbuffers/protobuf/releases/download/${__PROTOBUF_TAG__}/protobuf-cpp-3.12.4.zip)
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
elseif(PPLNN_DEP_RAPIDJSON_GIT)
    hpcc_declare_git_dep(rapidjson
        ${PPLNN_DEP_RAPIDJSON_GIT}
        ${__RAPIDJSON_COMMIT__})
else()
    hpcc_declare_pkg_dep(rapidjson
        https://github.com/Tencent/rapidjson/archive/${__RAPIDJSON_COMMIT__}.zip)
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
elseif(PPLNN_DEP_PYBIND11_GIT)
    hpcc_declare_git_dep(pybind11
        ${PPLNN_DEP_PYBIND11_GIT}
        ${__PYBIND11_TAG__})
else()
    hpcc_declare_pkg_dep(pybind11
        https://github.com/pybind/pybind11/archive/refs/tags/${__PYBIND11_TAG__}.zip)
endif()

unset(__PYBIND11_TAG__)

# --------------------------------------------------------------------------- #

set(LUACPP_INSTALL OFF CACHE BOOL "")
set(LUACPP_BUILD_TESTS OFF CACHE BOOL "")

set(__LUACPP_COMMIT__ 09070faa6de6cd5dbb3a247d66667d369f89f3c8)

if(PPLNN_DEP_LUACPP_PKG)
    hpcc_declare_pkg_dep(luacpp
        ${PPLNN_DEP_LUACPP_PKG})
elseif(PPLNN_DEP_LUACPP_GIT)
    hpcc_declare_git_dep(luacpp
        ${PPLNN_DEP_LUACPP_GIT}
        ${__LUACPP_COMMIT__})
else()
    hpcc_declare_pkg_dep(luacpp
        https://github.com/ouonline/lua-cpp/archive/${__LUACPP_COMMIT__}.zip)
endif()

unset(__LUACPP_COMMIT__)

# --------------------------------------------------------------------------- #

set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

set(__GOOGLETEST_TAG__ release-1.10.0)

if(PPLNN_DEP_GOOGLETEST_PKG)
    hpcc_declare_pkg_dep(googletest
        ${PPLNN_DEP_GOOGLETEST_PKG})
elseif(PPLNN_DEP_GOOGLETEST_GIT)
    hpcc_declare_git_dep(googletest
        ${PPLNN_DEP_GOOGLETEST_GIT}
        ${__GOOGLETEST_TAG__})
else()
    hpcc_declare_pkg_dep(googletest
        https://github.com/google/googletest/archive/refs/tags/${__GOOGLETEST_TAG__}.zip)
endif()

unset(__GOOGLETEST_TAG__)
