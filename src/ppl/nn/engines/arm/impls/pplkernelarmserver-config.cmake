cmake_minimum_required(VERSION 3.10)

if(TARGET "pplkernelarmserver_static")
    return()
endif()

set(__PPLKERNELARMSERVER_PACKAGE_DIR__ "${CMAKE_CURRENT_LIST_DIR}/../../..")

add_library(pplkernelarmserver_static STATIC IMPORTED)

if(NOT TARGET "pplcommon_static")
    include(${CMAKE_CURRENT_LIST_DIR}/pplcommon-config.cmake)
endif()

set_target_properties(pplkernelarmserver_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pplcommon_static"
    IMPORTED_LOCATION "${__PPLKERNELARMSERVER_PACKAGE_DIR__}/lib/libpplkernelarmserver_static.a"
    IMPORTED_LOCATION_DEBUG "${__PPLKERNELARMSERVER_PACKAGE_DIR__}/lib/libpplkernelarmserver_static.a"
    IMPORTED_LOCATION_RELEASE "${__PPLKERNELARMSERVER_PACKAGE_DIR__}/lib/libpplkernelarmserver_static.a")

unset(__PPLKERNELARMSERVER_PACKAGE_DIR__)
