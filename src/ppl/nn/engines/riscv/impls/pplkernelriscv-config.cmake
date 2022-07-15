cmake_minimum_required(VERSION 3.10)

if(TARGET "pplkernelriscv_static")
    return()
endif()

add_library(pplkernelriscv_static STATIC IMPORTED)

if(NOT TARGET "pplcommon_static")
    include(${CMAKE_CURRENT_LIST_DIR}/pplcommon-config.cmake)
endif()

get_filename_component(__PPLKERNELRISCV_PACKAGE_DIR__ "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set_target_properties(pplkernelriscv_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pplcommon_static"
    IMPORTED_LOCATION "${__PPLKERNELRISCV_PACKAGE_DIR__}/lib/libpplkernelriscv_static.a"
    IMPORTED_LOCATION_DEBUG "${__PPLKERNELRISCV_PACKAGE_DIR__}/lib/libpplkernelriscv_static.a"
    IMPORTED_LOCATION_RELEASE "${__PPLKERNELRISCV_PACKAGE_DIR__}/lib/libpplkernelriscv_static.a")
unset(__PPLKERNELRISCV_PACKAGE_DIR__)
