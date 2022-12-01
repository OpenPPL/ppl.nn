cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(PPLKernelArmServer)

if(HPCC_USE_OPENMP AND NOT PPLNN_USE_OPENMP)
    message(FATAL_ERROR "`HPCC_USE_OPENMP` is deprecated. use `PPLNN_USE_OPENMP` instead.")
endif()

# `PPLKERNELARMSERVER_COMPILE_OPTIONS`, `PPLKERNELARMSERVER_COMPILE_DEFINITIONS` and `PPLKERNELARMSERVER_LINK_LIBRARIES` are reserved

#### #### #### #### ####
### dependencies #######
#### #### #### #### ####
if(PPLNN_USE_OPENMP)
    find_package(Threads REQUIRED)
    find_package(OpenMP REQUIRED)
    list(APPEND PPLKERNELARMSERVER_LINK_LIBRARIES OpenMP::OpenMP_CXX)
endif()

list(APPEND PPLKERNELARMSERVER_LINK_LIBRARIES pplcommon_static)

#### #### #### #### ####
### file globbing ######
#### #### #### #### ####
file(GLOB_RECURSE PPLKERNELARMSERVER_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*)
if(PPLNN_USE_ARMV7)
    foreach(file ${PPLKERNELARMSERVER_SRC})
        if(${file} MATCHES ".*ppl/kernel/arm_server/conv2d/neon/fp../.*")
            LIST(REMOVE_ITEM PPLKERNELARMSERVER_SRC ${file})
        endif()
        if(${file} MATCHES ".*ppl/kernel/arm_server/fc/neon/fp../.*")
            LIST(REMOVE_ITEM PPLKERNELARMSERVER_SRC ${file})
        endif()
        if(${file} MATCHES ".*ppl/kernel/arm_server/gemm/neon/.+/.*")
            LIST(REMOVE_ITEM PPLKERNELARMSERVER_SRC ${file})
        endif()
    endforeach()
endif()

list(APPEND PPLKERNELARMSERVER_SOURCES ${PPLKERNELARMSERVER_SRC})
list(APPEND PPLKERNELARMSERVER_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/include)
list(APPEND PPLKERNELARMSERVER_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src)

#### #### #### #### ####
### definitions ########
#### #### #### #### ####
if(PPLNN_USE_OPENMP)
    list(APPEND PPLKERNELARMSERVER_COMPILE_DEFINITIONS PPL_USE_ARM_SERVER_OMP)
endif()

if(PPLNN_USE_AARCH64)
    list(APPEND PPLKERNELARMSERVER_COMPILE_DEFINITIONS PPLNN_USE_AARCH64)
endif()

if(PPLNN_USE_ARMV7)
    list(APPEND PPLKERNELARMSERVER_COMPILE_DEFINITIONS PPLNN_USE_ARMV7)
endif()

if(PPLNN_USE_ARMV8_2_FP16)
    list(APPEND PPLKERNELARMSERVER_COMPILE_DEFINITIONS PPLNN_USE_ARMV8_2_FP16)
endif()
if(PPLNN_USE_ARMV8_2_BF16)
    list(APPEND PPLKERNELARMSERVER_COMPILE_DEFINITIONS PPLNN_USE_ARMV8_2_BF16)
endif()
if(PPLNN_USE_ARMV8_2_I8MM)
    list(APPEND PPLKERNELARMSERVER_COMPILE_DEFINITIONS PPLNN_USE_ARMV8_2_I8MM)
endif()

if(PPLNN_ENABLE_KERNEL_PROFILING)
    list(APPEND PPLKERNELARMSERVER_COMPILE_DEFINITIONS PPLNN_ENABLE_KERNEL_PROFILING)
endif()

#### #### #### #### ####
### compile options ####
#### #### #### #### ####
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND PPLKERNELARMSERVER_COMPILE_OPTIONS "-fno-rtti;-fno-exceptions;-fno-builtin;-Wall;-Wextra;-Wno-unused-parameter;-Wno-unknown-pragmas")
endif()

if(PPLNN_USE_ARMV8_2_FP16)
    set(PPL_KERNEL_ARMV8_2_MARCH "-march=armv8.2-a+fp16")
    if (PPLNN_USE_ARMV8_2_BF16)
        set(PPL_KERNEL_ARMV8_2_MARCH "${PPL_KERNEL_ARMV8_2_MARCH}+bf16")
    endif()
    if (PPLNN_USE_ARMV8_2_I8MM)
        set(PPL_KERNEL_ARMV8_2_MARCH "${PPL_KERNEL_ARMV8_2_MARCH}+i8mm")
    endif()
    list(APPEND PPLKERNELARMSERVER_COMPILE_OPTIONS ${PPL_KERNEL_ARMV8_2_MARCH})
endif()

# dependencies
include(cmake/deps.cmake)
hpcc_populate_dep(pplcommon)

#### #### #### #### ####
### targets ############
#### #### #### #### ####
add_library(pplkernelarm_static STATIC ${PPLKERNELARMSERVER_SOURCES})
target_link_libraries(pplkernelarm_static PRIVATE ${PPLKERNELARMSERVER_LINK_LIBRARIES})
target_include_directories(pplkernelarm_static
    PUBLIC ${PPLKERNELARMSERVER_INCLUDE_DIRECTORIES}
    PRIVATE ${PPLCOMMON_INCLUDES})
target_compile_options(pplkernelarm_static PRIVATE ${PPLKERNELARMSERVER_COMPILE_OPTIONS})
target_compile_definitions(pplkernelarm_static PRIVATE ${PPLKERNELARMSERVER_COMPILE_DEFINITIONS})

if(CMAKE_COMPILER_IS_GNUCC)
    # requires a virtual destructor when having virtual functions, though it is not necessary for all cases
    target_compile_options(pplkernelarm_static PRIVATE -Werror=non-virtual-dtor)
endif()

if(PPLNN_INSTALL)
    install(TARGETS pplkernelarm_static DESTINATION lib)

    # `PPLKERNELARMSERVER_LINK_LIBRARIES` is needed for generating pplkernearmserver-config.cmake
    set(__PPLNN_CMAKE_CONFIG_FILE__ ${CMAKE_CURRENT_BINARY_DIR}/generated/pplkernelarm-config.cmake)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/pplkernelarm-config.cmake.in
        ${__PPLNN_CMAKE_CONFIG_FILE__}
        @ONLY)
    install(FILES ${__PPLNN_CMAKE_CONFIG_FILE__} DESTINATION lib/cmake/ppl)
    unset(__PPLNN_CMAKE_CONFIG_FILE__)
endif()
