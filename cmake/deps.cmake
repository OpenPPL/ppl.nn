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

# --------------------------------------------------------------------------- #

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(FATAL_ERROR "git is required.")
endif()

FetchContent_Declare(hpcc
    GIT_REPOSITORY https://github.com/openppl-public/hpcc.git
    GIT_TAG 97b154551d2cac09fa4558cb933f6372c4da83dc
    SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
    SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild
    UPDATE_DISCONNECTED True)

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable ppl.common tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable ppl.common benchmark")
set(PPLCOMMON_ENABLE_PYTHON_API ${PPLNN_ENABLE_PYTHON_API})
set(PPLCOMMON_ENABLE_LUA_API ${PPLNN_ENABLE_LUA_API})

hpcc_declare_git_dep(ppl.common
    https://github.com/openppl-public/ppl.common.git
    e022bdd45d620c4160365aa2895e6e17ba343af8)

# --------------------------------------------------------------------------- #

set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")

hpcc_declare_pkg_dep(protobuf
    https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.12.4.zip
    35baf103b2bbc166502c2e431d9d543b)

# --------------------------------------------------------------------------- #

set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "disable rapidjson tests")
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "disable rapidjson examples")
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "disable rapidjson docs")

hpcc_declare_pkg_dep(rapidjson
    https://github.com/Tencent/rapidjson/archive/2e8f5d897d9d461a7273b4b812b0127f321b1dcf.zip
    aadb4462dab0f019a5522ae4489ee1aa)

# --------------------------------------------------------------------------- #

set(PYBIND11_INSTALL OFF CACHE BOOL "disable pybind11 installation")
set(PYBIND11_TEST OFF CACHE BOOL "disable pybind11 tests")
set(PYBIND11_NOPYTHON ON CACHE BOOL "do not find python")
set(PYBIND11_FINDPYTHON OFF CACHE BOOL "do not find python")

hpcc_declare_pkg_dep(pybind11
    https://github.com/pybind/pybind11/archive/refs/tags/v2.7.0.zip
    267807f790ef598ef912a79aceefdc10)

# --------------------------------------------------------------------------- #

set(LUACPP_INSTALL OFF CACHE BOOL "")

hpcc_declare_pkg_dep(luacpp
    https://github.com/ouonline/lua-cpp/archive/6e9327be6594e54facba750178b13d86bf1a71fb.zip
    818e2f7890eb3fe92c7026d9bb4394b2)

# --------------------------------------------------------------------------- #

set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

hpcc_declare_pkg_dep(googletest
    https://github.com/google/googletest/archive/refs/tags/release-1.10.0.zip
    82358affdd7ab94854c8ee73a180fc53)
