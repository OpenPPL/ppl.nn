hpcc_populate_dep(pybind11)

set(__python_soabi__ )

if(NOT PYTHON3_INCLUDE_DIRS)
    find_package(Python3 COMPONENTS Development QUIET)
    if(NOT Python3_Development_FOUND)
        if(MSVC)
            message(FATAL_ERROR "python dev lib is required. please install python3 development files or specify `PYTHON3_INCLUDE_DIRS` and `PYTHON3_LIBRARIES` manually.")
        else()
            message(FATAL_ERROR "python dev lib is required. please install python3 development files (python3-dev on Debian/Ubuntu or python3-devel on RetHat/CentOS) or specify `PYTHON3_INCLUDE_DIRS` manually.")
        endif()
    endif()

    # `Python3_SOABI` is provided by find_package(Python3) in cmake >= 3.17
    set(__python_soabi__ ${Python3_SOABI})

    set(PYTHON3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    if(MSVC)  # cannot link static libraries built without -fPIC on linux
        set(PYTHON3_LIBRARIES ${Python3_LIBRARIES})
    endif()
endif()

if(MSVC)
    if(NOT PYTHON3_LIBRARIES)
        message(FATAL_ERROR "please specify `PYTHON3_INCLUDE_DIRS` and `PYTHON3_LIBRARIES` manually.")
    endif()
endif()

# --------------------------------------------------------------------------- #

file(GLOB PPLNN_PYTHON_API_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/py_pplnn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/engines/py_engine.cc)

if(PPLNN_USE_X86)
    file(GLOB_RECURSE PPLNN_PYTHON_X86_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/x86/*.cc)
    list(APPEND PPLNN_PYTHON_API_SRC ${PPLNN_PYTHON_X86_SRC})
endif()

if(PPLNN_USE_CUDA)
    file(GLOB_RECURSE PPLNN_PYTHON_CUDA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/cuda/*.cc)
    list(APPEND PPLNN_PYTHON_API_SRC ${PPLNN_PYTHON_CUDA_SRC})
endif()

if(PPLNN_USE_RISCV)
    file(GLOB_RECURSE PPLNN_PYTHON_RISCV_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/riscv/*.cc)
    list(APPEND PPLNN_PYTHON_API_SRC ${PPLNN_PYTHON_RISCV_SRC})
endif()

if(PPLNN_USE_ARM)
    file(GLOB_RECURSE PPLNN_PYTHON_ARM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/engines/arm/*.cc)
    list(APPEND PPLNN_PYTHON_API_SRC ${PPLNN_PYTHON_ARM_SRC})
endif()

if(PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE PPLNN_PYTHON_ONNX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/models/onnx/*.cc)
    list(APPEND PPLNN_PYTHON_API_SRC ${PPLNN_PYTHON_ONNX_SRC})
endif()

if(PPLNN_ENABLE_PMX_MODEL)
    file(GLOB_RECURSE PPLNN_PYTHON_PMX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/models/pmx/*.cc)
    list(APPEND PPLNN_PYTHON_API_SRC ${PPLNN_PYTHON_PMX_SRC})
endif()

if(NOT PPLNN_PYTHON_API_EXTERNAL_SOURCES)
    set(PPLNN_PYTHON_API_EXTERNAL_SOURCES py_dummy_load_resources.cc)
endif()

add_library(pypplnn_shared SHARED ${PPLNN_PYTHON_API_SRC} ${PPLNN_PYTHON_API_EXTERNAL_SOURCES})
target_link_libraries(pypplnn_shared PUBLIC pplnn_static ${PYTHON3_LIBRARIES})
target_include_directories(pypplnn_shared PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${pybind11_SOURCE_DIR}/include
    ${PYTHON3_INCLUDE_DIRS})
target_compile_features(pypplnn_shared PRIVATE cxx_std_11)

# --------------------------------------------------------------------------- #

# rename target for python import
set_target_properties(pypplnn_shared PROPERTIES PREFIX "")
set_target_properties(pypplnn_shared PROPERTIES OUTPUT_NAME "nn")

if(__python_soabi__)
    set(__so_suffix__ ".${__python_soabi__}")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(pypplnn_shared PROPERTIES SUFFIX "${__so_suffix__}.so") # using .so instead of .dylib on macosx
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(pypplnn_shared PROPERTIES SUFFIX "${__so_suffix__}.pyd") # using .pyd instead of .dll
endif()

unset(__so_suffix__)
unset(__python_soabi__)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_options(pypplnn_shared PUBLIC "LINKER:-undefined,dynamic_lookup") # required by pybind11
endif()

if(PPLNN_INSTALL)
    install(TARGETS pypplnn_shared DESTINATION lib/pyppl)
endif()
