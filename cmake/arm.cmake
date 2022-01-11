option(PPL_USE_ARMV8_2 "Build arm server kernel with armv8.2-a support." ON)
option(PPLNN_USE_NUMA "build with libnuma" OFF)

set(PPLCOMMON_USE_ARMV8_2 ${PPL_USE_ARMV8_2})

if (PPL_USE_ARMV8_2)
    set(PPL_USE_ARM_SERVER_FP16 ON)
endif()

file(GLOB_RECURSE PPLNN_ARM_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.S)
list(APPEND PPLNN_SOURCES ${PPLNN_ARM_SRC})

add_subdirectory(src/ppl/nn/engines/arm/impls)

list(APPEND PPLNN_LINK_LIBRARIES PPLKernelArmServer)
list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_AARCH64)
if (PPL_USE_ARM_SERVER_FP16)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPL_USE_ARM_SERVER_FP16)
endif()

if (PPLNN_USE_NUMA)
    list(APPEND PPLNN_LINK_LIBRARIES numa)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_NUMA)
endif()
