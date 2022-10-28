cmake_minimum_required(VERSION 3.10)

if(TARGET "pplkernelriscv_static")
    return()
endif()

add_library(pplkernelriscv_static STATIC IMPORTED)

if(NOT TARGET "pplcommon_static")
    include(${CMAKE_CURRENT_LIST_DIR}/pplcommon-config.cmake)
endif()

get_filename_component(__PPLNN_RISCV_LIB_PATH__ "${CMAKE_CURRENT_LIST_DIR}/../../../lib/@HPCC_STATIC_LIB_PREFIX@pplkernelriscv_static@HPCC_STATIC_LIB_SUFFIX@" ABSOLUTE)
set_target_properties(pplkernelriscv_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pplcommon_static"
    IMPORTED_LOCATION "${__PPLNN_RISCV_LIB_PATH__}"
    IMPORTED_LOCATION_DEBUG "${__PPLNN_RISCV_LIB_PATH__}"
    IMPORTED_LOCATION_RELEASE "${__PPLNN_RISCV_LIB_PATH__}")
unset(__PPLNN_RISCV_LIB_PATH__)
