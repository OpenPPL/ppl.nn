option(PPLNN_USE_ARMV8_2 "Build arm kernel with armv8.2-a support." ON)
option(PPLNN_USE_ARMV8_2_BF16 "Build arm kernel with armv8.2-a bf16 support. must enable PPLNN_USE_ARMV8_2 first." OFF)
option(PPLNN_USE_ARMV8_2_I8MM "Build arm kernel with armv8.2-a i8mm support. must enable PPLNN_USE_ARMV8_2 first." OFF)
option(PPLNN_USE_NUMA "build with libnuma" OFF)
option(PPLNN_USE_ANDROID_NDK "build with android ndk" OFF)

set(PPLNN_USE_ARM ON)
if(PPLNN_USE_AARCH64)
    set(PPLNN_USE_ARMV7   OFF)
elseif(PPLNN_USE_ARMV7)
    set(PPLNN_USE_AARCH64 OFF)
    set(PPLNN_USE_ARMV8_2 OFF)
    set(PPLNN_USE_NUMA OFF)
endif()

set(PPLCOMMON_USE_AARCH64 ${PPLNN_USE_AARCH64})
set(PPLCOMMON_USE_ARMV8_2 ${PPLNN_USE_ARMV8_2})
set(PPLCOMMON_USE_ARMV7   ${PPLNN_USE_ARMV7})
set(PPLCOMMON_USE_ARMV8_2_BF16 ${PPLNN_USE_ARMV8_2_BF16})
set(PPLCOMMON_USE_ARMV8_2_I8MM ${PPLNN_USE_ARMV8_2_I8MM})

if (PPLNN_USE_ARMV8_2)
    set(PPLNN_USE_ARMV8_2_FP16 ON)  # default enable fp16
else()
    set(PPLNN_USE_ARMV8_2_BF16 OFF) # bf16 must enable armv8.2 first
    set(PPLNN_USE_ARMV8_2_I8MM OFF) # i8mm must enable armv8.2 first
endif()

file(GLOB __PPLNN_ARM_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.cc)
if(PPLNN_SOURCE_EXTERNAL_ARM_ENGINE_SOURCES)
    list(REMOVE_ITEM __PPLNN_ARM_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/default_register_resources.cc)
endif()
file(GLOB_RECURSE __PPLNN_ARM_SRC_RECURSE__
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/kernels/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/optimizer/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/params/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/pmx/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/utils/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.S)
add_library(pplnn_arm_static STATIC
    ${__PPLNN_ARM_SRC__}
    ${__PPLNN_ARM_SRC_RECURSE__}
    ${PPLNN_SOURCE_EXTERNAL_ARM_ENGINE_SOURCES})
unset(__PPLNN_ARM_SRC_RECURSE__)
unset(__PPLNN_ARM_SRC__)

if ((DEFINED ANDROID_ABI) AND (EXISTS ${CMAKE_TOOLCHAIN_FILE}))
    set(PPLNN_USE_ANDROID_NDK ON)
endif()

if(PPLNN_USE_ANDROID_NDK)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-format")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-asm-operand-widths")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-asm-operand-widths")
    endif()
    add_link_options("-llog")

    set(PPLNN_USE_NUMA OFF)
endif()

add_subdirectory(src/ppl/nn/engines/arm/impls)
target_link_libraries(pplnn_arm_static PUBLIC pplnn_basic_static pplkernelarm_static)
target_include_directories(pplnn_arm_static PUBLIC ${PPLNN_SOURCE_EXTERNAL_ARM_INCLUDE_DIRECTORIES})
target_compile_definitions(pplnn_arm_static PUBLIC PPLNN_USE_ARM)

if (PPLNN_USE_AARCH64)
    target_compile_definitions(pplnn_arm_static PUBLIC PPLNN_USE_AARCH64)
endif()
if (PPLNN_USE_ARMV7)
    target_compile_definitions(pplnn_arm_static PUBLIC PPLNN_USE_ARMV7)
endif()
if (PPLNN_USE_ARMV8_2_FP16)
    target_compile_definitions(pplnn_arm_static PUBLIC PPLNN_USE_ARMV8_2_FP16)
endif()
if (PPLNN_USE_ARMV8_2_BF16)
    target_compile_definitions(pplnn_arm_static PUBLIC PPLNN_USE_ARMV8_2_BF16)
endif()
if (PPLNN_USE_ARMV8_2_I8MM)
    target_compile_definitions(pplnn_arm_static PUBLIC PPLNN_USE_ARMV8_2_I8MM)
endif()

if (PPLNN_USE_NUMA)
    target_link_libraries(pplnn_arm_static PUBLIC numa)
    target_compile_definitions(pplnn_arm_static PUBLIC PPLNN_USE_NUMA)
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

target_link_libraries(pplnn_static INTERFACE pplnn_arm_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/arm DESTINATION include/ppl/nn/engines)
    install(TARGETS pplnn_arm_static DESTINATION lib)
endif()
