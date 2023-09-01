if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

# forces to install libraries to `lib`, not `lib64` or others
set(CMAKE_INSTALL_LIBDIR lib)

# --------------------------------------------------------------------------- #

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

if(PPLNN_HOLD_DEPS)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# --------------------------------------------------------------------------- #

if(CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.4.0)
        message(FATAL_ERROR "gcc >= 9.4.0 is required.")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 10.3.0)
        message(FATAL_ERROR "gcc 10.3.0 has known bugs. use another version >= 9.4.0.")
    endif()
endif()

# --------------------------------------------------------------------------- #

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(FATAL_ERROR "git is required.")
endif()

set(__HPCC_COMMIT__ master)

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

set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")

if(MSVC)
    if(PPLNN_USE_MSVC_STATIC_RUNTIME)
        set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "")
    else()
        set(protobuf_BUILD_SHARED_LIBS ON CACHE BOOL "")
    endif()
endif()

if(PPLNN_PROTOBUF_VERSION)
    set(__PROTOBUF_TAG__ ${PPLNN_PROTOBUF_VERSION})
else()
    set(__PROTOBUF_TAG__ v23.4)
endif()

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

set(PPLNN_BUILD_TESTS OFF)
set(PPLNN_BUILD_SAMPLES OFF)

set(__PPLNN_COMMIT__ master)

if(PPLNN_DEP_PPLNN_PKG)
    hpcc_declare_pkg_dep(pplnn
        ${PPLNN_DEP_PPLNN_PKG})
else()
    if(NOT PPLNN_DEP_PPLNN_GIT)
        set(PPLNN_DEP_PPLNN_GIT "https://github.com/openppl-public/ppl.nn.git")
    endif()
    hpcc_declare_git_dep(pplnn
        ${PPLNN_DEP_PPLNN_GIT}
        ${__PPLNN_COMMIT__})
endif()

unset(__PPLNN_COMMIT__)

# --------------------------------------------------------------------------- #

set(__LLM_KERNEL_CUDA_COMMIT__ master)

if(PPLNN_DEP_PPL_LLM_KERNEL_CUDA_PKG)
    hpcc_declare_pkg_dep(ppl.llm.kernel.cuda
        ${PPLNN_DEP_PPL_LLM_KERNEL_CUDA_PKG})
else()
    if(NOT PPLNN_DEP_PPL_LLM_KERNEL_CUDA_GIT)
        set(PPLNN_DEP_PPL_LLM_KERNEL_CUDA_GIT "https://github.com/openppl-public/ppl.llm.kernel.cuda.git")
    endif()
    hpcc_declare_git_dep(ppl.llm.kernel.cuda
        ${PPLNN_DEP_PPL_LLM_KERNEL_CUDA_GIT}
        ${__LLM_KERNEL_CUDA_COMMIT__})
endif()

unset(__LLM_KERNEL_CUDA_COMMIT__)
