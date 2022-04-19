option(PPLNN_USE_ARMV8_2 "Build arm server kernel with armv8.2-a support." ON)
option(PPLNN_USE_NUMA "build with libnuma" OFF)
option(PPLNN_USE_ANDROID_NDK "build with android ndk" OFF)

set(PPLNN_USE_ARM ON)
set(PPLCOMMON_USE_ARMV8_2 ${PPLNN_USE_ARMV8_2})

if (PPLNN_USE_ARMV8_2)
    set(PPLNN_USE_ARMV8_2_FP16 ON)
endif()

file(GLOB_RECURSE PPLNN_ARM_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.S)
list(APPEND PPLNN_SOURCES ${PPLNN_ARM_SRC})

if ((DEFINED ANDROID_PLATFORM) AND (DEFINED ANDROID_ABI) AND (EXISTS ${CMAKE_TOOLCHAIN_FILE}))
    set(PPLNN_USE_ANDROID_NDK ON)
endif()

if(PPLNN_USE_ANDROID_NDK)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-asm-operand-widths")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        add_link_options("-llog")
        add_link_options("-static-openmp")
    endif()

    set(PPLNN_USE_NUMA OFF)
endif()

add_subdirectory(src/ppl/nn/engines/arm/impls)

list(APPEND PPLNN_LINK_LIBRARIES pplkernelarmserver_static)
list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_ARM PPLNN_USE_AARCH64)
if (PPLNN_USE_ARMV8_2_FP16)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_ARMV8_2_FP16)
endif()

if (PPLNN_USE_NUMA)
    list(APPEND PPLNN_LINK_LIBRARIES numa)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_NUMA)
endif()

if(PPLNN_ENABLE_SANITIZE_OPTIONS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(__ASAN_FLAGS__ "-fsanitize=undefined -fsanitize=address -fsanitize=leak -fno-omit-frame-pointer")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${__ASAN_FLAGS__}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${__ASAN_FLAGS__}")
        unset(__ASAN_FLAGS__)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_link_options("-static-libasan")
        endif()
    else()
        message(FATAL_ERROR "UNSUPPORTED: `PPLNN_USE_SANITIZE` is ON when using compiler `${CMAKE_CXX_COMPILER_ID}`.")
    endif()
endif()
