hpcc_populate_dep(pybind11)

if(NOT PYTHON3_INCLUDE_DIRS)
    find_package(Python3 COMPONENTS Development QUIET)
    if(NOT Python3_Development_FOUND)
        if(MSVC)
            message(FATAL_ERROR "python dev lib is required. please install python3 development files or specify `PYTHON3_INCLUDE_DIRS` and `PYTHON3_LIBRARIES` manually.")
        else()
            message(FATAL_ERROR "python dev lib is required. please install python3 development files (python3-dev on Debian/Ubuntu or python3-devel on RetHat/CentOS) or specify `PYTHON3_INCLUDE_DIRS` manually.")
        endif()
    endif()
    set(PYTHON3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})

    if(MSVC)
        set(PYTHON3_LIBRARIES ${Python3_LIBRARIES})
    endif()
endif()

if(MSVC)
    if(NOT PYTHON3_LIBRARIES)
        message(FATAL_ERROR "please specify `PYTHON3_INCLUDE_DIRS` and `PYTHON3_LIBRARIES` manually.")
    endif()
endif()

file(GLOB_RECURSE PPLNN_PYTHON_API_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_library(pypplnn_shared SHARED ${PPLNN_PYTHON_API_SRC})
target_link_libraries(pypplnn_shared PUBLIC pplnn_static ${PYTHON3_LIBRARIES})
target_include_directories(pypplnn_shared PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${pybind11_SOURCE_DIR}/include
    ${PYTHON3_INCLUDE_DIRS})
target_compile_features(pypplnn_shared PRIVATE cxx_std_11)

# rename target for python import
set_target_properties(pypplnn_shared PROPERTIES PREFIX "")
set_target_properties(pypplnn_shared PROPERTIES OUTPUT_NAME "nn")

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    target_link_options(pypplnn_shared PUBLIC "LINKER:-undefined,dynamic_lookup") # required by pybind11
    set_target_properties(pypplnn_shared PROPERTIES SUFFIX ".so") # using .so instead of .dylib
endif()

if(PPLNN_INSTALL)
    install(TARGETS pypplnn_shared DESTINATION lib/pyppl)
endif()
