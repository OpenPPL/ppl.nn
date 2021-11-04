cmake_minimum_required(VERSION 3.10)

if(TARGET "pplkernelcuda_static")
    return()
endif()

set(__PPLKERNELCUDA_PACKAGE_DIR__ "${CMAKE_CURRENT_LIST_DIR}/../../..")

add_library(pplkernelcuda_static STATIC IMPORTED)

if(NOT TARGET "pplcommon_static")
    include(${CMAKE_CURRENT_LIST_DIR}/pplcommon-config.cmake)
endif()

set_target_properties(pplkernelcuda_static PROPERTIES
    INTERFACE_LINK_LIBRARIES "pplcommon_static")

if(MSVC)
    set_target_properties(pplkernelcuda_static PROPERTIES
        IMPORTED_LOCATION "${__PPLKERNELCUDA_PACKAGE_DIR__}/lib/libpplkernelcuda_static.lib"
        IMPORTED_LOCATION_DEBUG "${__PPLKERNELCUDA_PACKAGE_DIR__}/lib/libpplkernelcuda_static.lib"
        IMPORTED_LOCATION_RELEASE "${__PPLKERNELCUDA_PACKAGE_DIR__}/lib/libpplkernelcuda_static.lib")
else()
    set_target_properties(pplkernelcuda_static PROPERTIES
        IMPORTED_LOCATION "${__PPLKERNELCUDA_PACKAGE_DIR__}/lib/libpplkernelcuda_static.a"
        IMPORTED_LOCATION_DEBUG "${__PPLKERNELCUDA_PACKAGE_DIR__}/lib/libpplkernelcuda_static.a"
        IMPORTED_LOCATION_RELEASE "${__PPLKERNELCUDA_PACKAGE_DIR__}/lib/libpplkernelcuda_static.a")
endif()

unset(__PPLKERNELCUDA_PACKAGE_DIR__)
