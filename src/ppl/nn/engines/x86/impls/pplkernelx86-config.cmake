cmake_minimum_required(VERSION 3.10)

if(TARGET "pplkernelx86_static")
    return()
endif()

set(__PPLKERNELX86_PACKAGE_DIR__ "${CMAKE_CURRENT_LIST_DIR}/../../..")

add_library(pplkernelx86_static STATIC IMPORTED)

if(NOT TARGET "pplcommon_static")
    include(${CMAKE_CURRENT_LIST_DIR}/pplcommon-config.cmake)
endif()

set_target_properties(pplkernelx86_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pplcommon_static")

if(MSVC)
    set_target_properties(pplkernelx86_static PROPERTIES
        IMPORTED_LOCATION "${__PPLKERNELX86_PACKAGE_DIR__}/lib/libpplkernelx86_static.lib"
        IMPORTED_LOCATION_DEBUG "${__PPLKERNELX86_PACKAGE_DIR__}/lib/libpplkernelx86_static.lib"
        IMPORTED_LOCATION_RELEASE "${__PPLKERNELX86_PACKAGE_DIR__}/lib/libpplkernelx86_static.lib")
else()
    set_target_properties(pplkernelx86_static PROPERTIES
        IMPORTED_LOCATION "${__PPLKERNELX86_PACKAGE_DIR__}/lib/libpplkernelx86_static.a"
        IMPORTED_LOCATION_DEBUG "${__PPLKERNELX86_PACKAGE_DIR__}/lib/libpplkernelx86_static.a"
        IMPORTED_LOCATION_RELEASE "${__PPLKERNELX86_PACKAGE_DIR__}/lib/libpplkernelx86_static.a")
endif()

unset(__PPLKERNELX86_PACKAGE_DIR__)
