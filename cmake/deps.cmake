if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

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
    GIT_TAG v0.1.0
    GIT_SHALLOW TRUE
    SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
    SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild
    UPDATE_COMMAND "")

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable ppl.common tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable ppl.common benchmark")

hpcc_declare_git_dep(ppl.common
    https://github.com/openppl-public/ppl.common.git
    v0.1.0)

# --------------------------------------------------------------------------- #

set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")

hpcc_declare_git_dep(protobuf
    https://github.com/protocolbuffers/protobuf.git
    v3.1.0)

# --------------------------------------------------------------------------- #

set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "disable rapidjson tests")
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "disable rapidjson examples")
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "disable rapidjson docs")

hpcc_declare_git_dep(rapidjson
    https://github.com/Tencent/rapidjson.git
    master)

# --------------------------------------------------------------------------- #

hpcc_declare_git_dep(googletest
    https://github.com/google/googletest.git
    release-1.10.0)
