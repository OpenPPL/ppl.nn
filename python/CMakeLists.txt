hpcc_populate_dep(pybind11)

if(NOT PYTHON3_INCLUDE_DIRS)
    find_package(Python3 COMPONENTS Development REQUIRED)
    if(NOT Python3_Development_FOUND)
        message("python dev lib is required.")
    endif()
    set(PYTHON3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
endif()

file(GLOB_RECURSE PPLNN_PYTHON_API_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_library(pypplnn_shared SHARED ${PPLNN_PYTHON_API_SRC})
target_link_libraries(pypplnn_shared PUBLIC pplnn_static)
target_include_directories(pypplnn_shared PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${pybind11_SOURCE_DIR}/include
    ${PYTHON3_INCLUDE_DIRS})

# rename target for python import
set_target_properties(pypplnn_shared PROPERTIES PREFIX "")
set_target_properties(pypplnn_shared PROPERTIES OUTPUT_NAME "pypplnn")
