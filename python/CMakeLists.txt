hpcc_populate_dep(pybind11)

set(__python_soabi__ )

if(NOT PYTHON3_INCLUDE_DIRS)
    find_package(Python3 ${PPLNN_PYTHON3_VERSION} COMPONENTS Development QUIET)
    if(NOT Python3_Development_FOUND)
        if(MSVC)
            message(FATAL_ERROR "python dev lib is required. please install python3 development files or specify `PYTHON3_INCLUDE_DIRS` and `PYTHON3_LIBRARIES` manually.")
        else()
            message(FATAL_ERROR "python dev lib is required. please install python3 development files (python3-dev on Debian/Ubuntu or python3-devel on RetHat/CentOS) or specify `PYTHON3_INCLUDE_DIRS` manually.")
        endif()
    endif()

    if(Python3_SOABI)
        # `Python3_SOABI` is provided by find_package(Python3) in cmake >= 3.17
        set(__python_soabi__ ${Python3_SOABI})
    else()
        find_package(Python3 ${PPLNN_PYTHON3_VERSION} COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import sysconfig;print(sysconfig.get_config_var('SOABI'))"
                OUTPUT_VARIABLE __python_soabi__
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
    endif()

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

file(GLOB __PPLNN_PYTHON_API_SRC__
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/py_pplnn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/engines/py_engine.cc)

if(PPLNN_USE_X86)
    file(GLOB_RECURSE __PPLNN_PYTHON_X86_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/x86/*.cc)
    list(APPEND __PPLNN_PYTHON_API_SRC__ ${__PPLNN_PYTHON_X86_SRC__})
    unset(__PPLNN_PYTHON_X86_SRC__)
endif()

if(PPLNN_USE_CUDA)
    file(GLOB_RECURSE __PPLNN_PYTHON_CUDA_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/cuda/*.cc)
    list(APPEND __PPLNN_PYTHON_API_SRC__ ${__PPLNN_PYTHON_CUDA_SRC__})
    unset(__PPLNN_PYTHON_CUDA_SRC__)
endif()

if(PPLNN_USE_RISCV)
    file(GLOB_RECURSE __PPLNN_PYTHON_RISCV_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/riscv/*.cc)
    list(APPEND __PPLNN_PYTHON_API_SRC__ ${__PPLNN_PYTHON_RISCV_SRC__})
    unset(__PPLNN_PYTHON_RISCV_SRC__)
endif()

if(PPLNN_USE_ARM)
    file(GLOB_RECURSE __PPLNN_PYTHON_ARM_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/engines/arm/*.cc)
    list(APPEND __PPLNN_PYTHON_API_SRC__ ${__PPLNN_PYTHON_ARM_SRC__})
    unset(__PPLNN_PYTHON_ARM_SRC__)
endif()

if(PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB_RECURSE __PPLNN_PYTHON_ONNX_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/models/onnx/*.cc)
    list(APPEND __PPLNN_PYTHON_API_SRC__ ${__PPLNN_PYTHON_ONNX_SRC__})
    unset(__PPLNN_PYTHON_ONNX_SRC__)
endif()

if(PPLNN_ENABLE_PMX_MODEL)
    file(GLOB_RECURSE __PPLNN_PYTHON_PMX_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/models/pmx/*.cc)
    list(APPEND __PPLNN_PYTHON_API_SRC__ ${__PPLNN_PYTHON_PMX_SRC__})
    unset(__PPLNN_PYTHON_PMX_SRC__)
endif()

if(NOT PPLNN_SOURCE_EXTERNAL_PYTHON_SOURCES)
    set(PPLNN_SOURCE_EXTERNAL_PYTHON_SOURCES py_dummy_load_resources.cc)
endif()

add_library(pypplnn_shared SHARED ${__PPLNN_PYTHON_API_SRC__} ${PPLNN_SOURCE_EXTERNAL_PYTHON_SOURCES})
unset(__PPLNN_PYTHON_API_SRC__)

target_link_libraries(pypplnn_shared
    PUBLIC ${PYTHON3_LIBRARIES}
    PRIVATE pplnn_static ${PPLNN_SOURCE_EXTERNAL_PYTHON_LINK_LIBRARIES})
target_include_directories(pypplnn_shared PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${pybind11_SOURCE_DIR}/include
    ${PPLNN_SOURCE_EXTERNAL_PYTHON_INCLUDE_DIRECTORIES}
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
