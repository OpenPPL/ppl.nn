if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

# forces to install libraries to `lib`, not `lib64` or others
set(CMAKE_INSTALL_LIBDIR lib)

# --------------------------------------------------------------------------- #

if(CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
        message(FATAL_ERROR "gcc >= 4.9.0 is required.")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 10.3.0)
        message(FATAL_ERROR "gcc 10.3.0 has known bugs. use another version >= 9.4.0.")
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

if(NOT PPLNN_DEP_HPCC_VERSION)
    set(PPLNN_DEP_HPCC_VERSION master)
endif()

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
        GIT_TAG ${PPLNN_DEP_HPCC_VERSION}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable pplcommon tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable pplcommon benchmark")
if(PPLNN_ENABLE_PYTHON_API)
    set(PPLCOMMON_ENABLE_PYTHON_API ON)
endif()
if(PPLNN_HOLD_DEPS)
    set(PPLCOMMON_HOLD_DEPS ON)
endif()
if(PPLNN_USE_X86_64)
    set(PPLCOMMON_USE_X86_64 ON)
endif()
if(PPLNN_USE_AARCH64)
    set(PPLCOMMON_USE_AARCH64 ON)
elseif(PPLNN_USE_ARMV7)
    set(PPLCOMMON_USE_ARMV7 ON)
endif()
if(PPLNN_USE_CUDA)
    set(PPLCOMMON_USE_CUDA ON)
endif()
if(PPLNN_CUDA_ENABLE_NCCL)
    set(PPLCOMMON_ENABLE_NCCL ON)
endif()

if(NOT PPLNN_DEP_PPLCOMMON_VERSION)
    set(PPLNN_DEP_PPLCOMMON_VERSION master)
endif()

if(PPLNN_DEP_PPLCOMMON_PKG)
    hpcc_declare_pkg_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_PKG})
else()
    if(NOT PPLNN_DEP_PPLCOMMON_GIT)
        set(PPLNN_DEP_PPLCOMMON_GIT "https://github.com/openppl-public/ppl.common.git")
    endif()
    hpcc_declare_git_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_GIT}
        ${PPLNN_DEP_PPLCOMMON_VERSION})
endif()

# --------------------------------------------------------------------------- #

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "disable tests")
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "disable installation")
set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "")
set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "")
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "")

set(__FLATBUFFERS_TAG__ v2.0.8)

if(PPLNN_DEP_FLATBUFFERS_PKG)
    hpcc_declare_pkg_dep(flatbuffers
        ${PPLNN_DEP_FLATBUFFERS_PKG})
else()
    if(NOT PPLNN_DEP_FLATBUFFERS_GIT)
        set(PPLNN_DEP_FLATBUFFERS_GIT "https://github.com/google/flatbuffers.git")
    endif()
    hpcc_declare_git_dep_depth1(flatbuffers
        ${PPLNN_DEP_FLATBUFFERS_GIT}
        ${__FLATBUFFERS_TAG__})
endif()

unset(__FLATBUFFERS_TAG__)

# --------------------------------------------------------------------------- #

set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")

if(MSVC)
    if(PPLNN_USE_MSVC_STATIC_RUNTIME)
        set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "")
    else()
        set(protobuf_BUILD_SHARED_LIBS ON CACHE BOOL "")
    endif()
endif()

if(PPLNN_DEP_PROTOBUF_VERSION)
    set(__PROTOBUF_TAG__ ${PPLNN_DEP_PROTOBUF_VERSION})
else()
    set(__PROTOBUF_TAG__ v3.1.0)
endif()

if(PPLNN_DEP_PROTOBUF_PKG)
    hpcc_declare_pkg_dep(protobuf
        ${PPLNN_DEP_PROTOBUF_PKG})
else()
    if(NOT PPLNN_DEP_PROTOBUF_GIT)
        set(PPLNN_DEP_PROTOBUF_GIT "https://github.com/protocolbuffers/protobuf.git")
    endif()
    hpcc_declare_git_dep_depth1(protobuf
        ${PPLNN_DEP_PROTOBUF_GIT}
        ${__PROTOBUF_TAG__})
endif()

unset(__PROTOBUF_TAG__)

# --------------------------------------------------------------------------- #

set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "disable rapidjson tests")
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "disable rapidjson examples")
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "disable rapidjson docs")

set(__RAPIDJSON_COMMIT__ 06d58b9e848c650114556a23294d0b6440078c61)

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
    hpcc_declare_git_dep_depth1(pybind11
        ${PPLNN_DEP_PYBIND11_GIT}
        ${__PYBIND11_TAG__})
endif()

unset(__PYBIND11_TAG__)

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
    hpcc_declare_git_dep_depth1(googletest
        ${PPLNN_DEP_GOOGLETEST_GIT}
        ${__GOOGLETEST_TAG__})
endif()

unset(__GOOGLETEST_TAG__)

# --------------------------------------------------------------------------- #

if(PPLNN_USE_X86_64 OR PPLNN_USE_AARCH64 OR PPLNN_USE_ARMV7 OR PPLNN_USE_RISCV64)
    if(NOT PPLNN_DEP_PPLCPUKERNEL_VERSION)
        set(PPLNN_DEP_PPLCPUKERNEL_VERSION master)
    endif()

    if(PPLNN_DEP_PPLCPUKERNEL_PKG)
        hpcc_declare_pkg_dep(ppl.kernel.cpu
            ${PPLNN_DEP_PPLCPUKERNEL_PKG})
    else()
        if(NOT PPLNN_DEP_PPLCPUKERNEL_GIT)
            set(PPLNN_DEP_PPLCPUKERNEL_GIT "https://github.com/openppl-public/ppl.kernel.cpu.git")
        endif()
        hpcc_declare_git_dep_depth1(ppl.kernel.cpu
            ${PPLNN_DEP_PPLCPUKERNEL_GIT}
            ${PPLNN_DEP_PPLCPUKERNEL_VERSION})
    endif()
endif()

# --------------------------------------------------------------------------- #

if(PPLNN_USE_CUDA)
    if(NOT PPLNN_DEP_PPLCUDAKERNEL_VERSION)
        set(PPLNN_DEP_PPLCUDAKERNEL_VERSION master)
    endif()

    if(PPLNN_DEP_PPLCUDAKERNEL_PKG)
        hpcc_declare_pkg_dep(ppl.kernel.cuda
            ${PPLNN_DEP_PPLCUDAKERNEL_PKG})
    else()
        if(NOT PPLNN_DEP_PPLCUDAKERNEL_GIT)
            set(PPLNN_DEP_PPLCUDAKERNEL_GIT "https://github.com/openppl-public/ppl.kernel.cuda.git")
        endif()
        hpcc_declare_git_dep_depth1(ppl.kernel.cuda
            ${PPLNN_DEP_PPLCUDAKERNEL_GIT}
            ${PPLNN_DEP_PPLCUDAKERNEL_VERSION})
    endif()
endif()

# --------------------------------------------------------------------------- #

if(NOT PPLNN_DEP_PPL_LLM_KERNEL_CUDA_VERSION)
    set(PPLNN_DEP_PPL_LLM_KERNEL_CUDA_VERSION master)
endif()

if(PPLNN_DEP_PPL_LLM_KERNEL_CUDA_PKG)
    hpcc_declare_pkg_dep(ppl.llm.kernel.cuda
        ${PPLNN_DEP_PPL_LLM_KERNEL_CUDA_PKG})
else()
    if(NOT PPLNN_DEP_PPL_LLM_KERNEL_CUDA_GIT)
        set(PPLNN_DEP_PPL_LLM_KERNEL_CUDA_GIT "https://github.com/openppl-public/ppl.llm.kernel.cuda.git")
    endif()
    hpcc_declare_git_dep_depth1(ppl.llm.kernel.cuda
        ${PPLNN_DEP_PPL_LLM_KERNEL_CUDA_GIT}
        ${PPLNN_DEP_PPL_LLM_KERNEL_CUDA_VERSION})
endif()
