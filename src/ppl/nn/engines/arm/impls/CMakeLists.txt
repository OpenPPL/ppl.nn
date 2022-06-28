cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(PPLKernelArmServer)

option(PPL_USE_ARM_SERVER_OMP "Build arm server kernel with openmp support." ON)

if(HPCC_USE_OPENMP AND NOT PPLNN_USE_OPENMP)
    message(FATAL_ERROR "`HPCC_USE_OPENMP` is deprecated. use `PPLNN_USE_OPENMP` instead.")
endif()

if(PPLNN_USE_OPENMP)
    set(PPL_USE_ARM_SERVER_OMP ON)
endif()

set(PPL_KERNEL_ARM_SERVER_COMPILE_OPTIONS )
set(PPL_KERNEL_ARM_SERVER_COMPILE_DEFINITIONS )
set(PPL_KERNEL_ARM_SERVER_LINK_LIBRARIES )

#### #### #### #### ####
### dependencies #######
#### #### #### #### ####
if(PPL_USE_ARM_SERVER_OMP)
    find_package(Threads REQUIRED)
    find_package(OpenMP REQUIRED)
    list(APPEND PPL_KERNEL_ARM_SERVER_LINK_LIBRARIES OpenMP::OpenMP_CXX)
endif()

list(APPEND PPL_KERNEL_ARM_SERVER_LINK_LIBRARIES pplcommon_static)

set(__PPLNN_SOURCE_DIR__ ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)
set(PPLNN_ARM_ENGINE_INCLUDE_DIRECTORIES ${__PPLNN_SOURCE_DIR__}/src/ ${__PPLNN_SOURCE_DIR__}/include/)
unset(__PPLNN_SOURCE_DIR__)

#### #### #### #### ####
### file globbing ######
#### #### #### #### ####
file(GLOB_RECURSE PPL_KERNEL_ARM_SERVER_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*)
if(PPLNN_USE_ARMV7)
    foreach(file ${PPL_KERNEL_ARM_SERVER_SRC})
        if(${file} MATCHES ".*ppl/kernel/arm_server/conv2d/neon/fp../.*")
            LIST(REMOVE_ITEM PPL_KERNEL_ARM_SERVER_SRC ${file})
        endif()
        if(${file} MATCHES ".*ppl/kernel/arm_server/fc/neon/fp../.*")
            LIST(REMOVE_ITEM PPL_KERNEL_ARM_SERVER_SRC ${file})
        endif()
        if(${file} MATCHES ".*ppl/kernel/arm_server/gemm/neon/.+/.*")
            LIST(REMOVE_ITEM PPL_KERNEL_ARM_SERVER_SRC ${file})
        endif()
    endforeach()
endif()

list(APPEND PPL_KERNEL_ARM_SERVER_SOURCES ${PPL_KERNEL_ARM_SERVER_SRC})
list(APPEND PPL_KERNEL_ARM_SERVER_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/include)
list(APPEND PPL_KERNEL_ARM_SERVER_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src)

#### #### #### #### ####
### definitions ########
#### #### #### #### ####
if(PPL_USE_ARM_SERVER_OMP)
    list(APPEND PPL_KERNEL_ARM_SERVER_COMPILE_DEFINITIONS PPL_USE_ARM_SERVER_OMP)
endif()

if(PPLNN_USE_AARCH64)
    list(APPEND PPL_KERNEL_ARM_SERVER_COMPILE_DEFINITIONS PPLNN_USE_AARCH64)
endif()

if(PPLNN_USE_ARMV7)
    list(APPEND PPL_KERNEL_ARM_SERVER_COMPILE_DEFINITIONS PPLNN_USE_ARMV7)
endif()

if(PPLNN_USE_ARMV8_2_FP16)
    list(APPEND PPL_KERNEL_ARM_SERVER_COMPILE_DEFINITIONS PPLNN_USE_ARMV8_2_FP16)
endif()

if(PPLNN_ENABLE_KERNEL_PROFILING)
    list(APPEND PPL_KERNEL_ARM_SERVER_COMPILE_DEFINITIONS PPLNN_ENABLE_KERNEL_PROFILING)
endif()

#### #### #### #### ####
### compile options ####
#### #### #### #### ####
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND PPL_KERNEL_ARM_SERVER_COMPILE_OPTIONS "-fno-rtti;-fno-exceptions;-fno-builtin;-Wall;-Wextra;-Wno-unused-parameter;-Wno-unknown-pragmas")
endif()

if(PPLNN_USE_ARMV8_2_FP16)
    list(APPEND PPL_KERNEL_ARM_SERVER_COMPILE_OPTIONS "-march=armv8.2-a+fp16")
endif()

#### #### #### #### ####
### targets ############
#### #### #### #### ####
add_library(pplkernelarmserver_static STATIC ${PPL_KERNEL_ARM_SERVER_SOURCES})
target_link_libraries(pplkernelarmserver_static PRIVATE ${PPL_KERNEL_ARM_SERVER_LINK_LIBRARIES})
target_include_directories(pplkernelarmserver_static
    PUBLIC ${PPL_KERNEL_ARM_SERVER_INCLUDE_DIRECTORIES}
    PRIVATE ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES}
    PRIVATE ${PPLNN_ARM_ENGINE_INCLUDE_DIRECTORIES})
target_compile_options(pplkernelarmserver_static PRIVATE ${PPL_KERNEL_ARM_SERVER_COMPILE_OPTIONS})
target_compile_definitions(pplkernelarmserver_static PRIVATE ${PPL_KERNEL_ARM_SERVER_COMPILE_DEFINITIONS})

if(CMAKE_COMPILER_IS_GNUCC)
    # requires a virtual destructor when having virtual functions, though it is not necessary for all cases
    target_compile_options(pplkernelarmserver_static PRIVATE -Werror=non-virtual-dtor)
endif()

if(PPLNN_INSTALL)
    install(TARGETS pplkernelarmserver_static DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/pplkernelarmserver-config.cmake DESTINATION lib/cmake/ppl)
endif()
