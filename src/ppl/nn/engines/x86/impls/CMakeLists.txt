cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(PPLKernelX86)

option(PPL_USE_X86_OMP "Build x86 kernel with openmp support." OFF)
option(PPL_USE_X86_AVX512 "Build x86 kernel with avx512 support." ON)

if(MSVC)
set(PPLKERNELX86_COMPILE_OPTIONS )
else()
set(PPLKERNELX86_COMPILE_OPTIONS "-Werror=return-type;-Wno-strict-aliasing")
endif()

# `PPLKERNELX86_LINK_LIBRARIES` is needed for generating pplkernex86-config.cmake
set(PPLKERNELX86_LINK_LIBRARIES )

list(APPEND PPLKERNELX86_PUBLIC_INCLUDE_DIRECTORIES include)
list(APPEND PPLKERNELX86_PRIVATE_INCLUDE_DIRECTORIES src)

if(HPCC_USE_OPENMP AND NOT PPLNN_USE_OPENMP)
    message(FATAL_ERROR "`HPCC_USE_OPENMP` is deprecated. use `PPLNN_USE_OPENMP` instead.")
endif()

if(PPLNN_USE_OPENMP)
    set(PPL_USE_X86_OMP ON)
endif()

if(PPL_USE_X86_OMP)
    FIND_PACKAGE(OpenMP REQUIRED)
    list(APPEND PPLKERNELX86_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    list(APPEND PPLKERNELX86_COMPILE_DEFINITIONS PPL_USE_X86_OMP)
endif()

if(NOT ((CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9.2) OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6.0.0) OR (MSVC_VERSION GREATER 1910)))
    if (PPL_USE_X86_AVX512)
        message(FATAL_ERROR
            "[FATAL_ERROR]:\n"
            "Compiler does not support AVX512F due to the compiler version is too old.\n"
            "We STRONGLY SUGGEST to upgrade the compliler version to gcc>4.9.2 to enable AVX512F, clang>=6.0.0, MSVC>1910.\n"
            "Another NOT RECOMENDED option is disabling AVX512F by setting cmake option `PPL_USE_X86_AVX512=OFF`\n")
    endif()
endif()

file(GLOB_RECURSE _I_PPLKERNELX86_SRC src/ppl/kernel/x86/*.cpp)
file(GLOB_RECURSE _I_PPLKERNELX86_SSE_SRC src/ppl/kernel/x86/*_sse.cpp)
file(GLOB_RECURSE _I_PPLKERNELX86_AVX_SRC src/ppl/kernel/x86/*_avx.cpp)
file(GLOB_RECURSE _I_PPLKERNELX86_FMA_SRC src/ppl/kernel/x86/*_fma.cpp)
file(GLOB_RECURSE _I_PPLKERNELX86_AVX512_SRC src/ppl/kernel/x86/*_avx512.cpp)

list(APPEND PPLKERNELX86_SRC ${_I_PPLKERNELX86_SRC})
list(APPEND PPLKERNELX86_SSE_SRC ${_I_PPLKERNELX86_SSE_SRC})
list(APPEND PPLKERNELX86_AVX_SRC ${_I_PPLKERNELX86_AVX_SRC})
list(APPEND PPLKERNELX86_FMA_SRC ${_I_PPLKERNELX86_FMA_SRC})
list(APPEND PPLKERNELX86_AVX512_SRC ${_I_PPLKERNELX86_AVX512_SRC})

set(PPLKERNELX86_SSE_FLAGS )
set(PPLKERNELX86_AVX_FLAGS )
set(PPLKERNELX86_FMA_FLAGS )
set(PPLKERNELX86_AVX512_FLAGS )
if (CMAKE_COMPILER_IS_GNUCC)
    set(PPLKERNELX86_AVX512_FLAGS "-mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
    set(PPLKERNELX86_FMA_FLAGS "-mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
    set(PPLKERNELX86_AVX_FLAGS "-mtune-ctrl=256_unaligned_load_optimal,256_unaligned_store_optimal")
endif()

set_source_files_properties(${PPLKERNELX86_SSE_SRC} PROPERTIES
    COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${PPLKERNELX86_SSE_FLAGS}")
set_source_files_properties(${PPLKERNELX86_AVX_SRC} PROPERTIES
    COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${AVX_ENABLED_FLAGS} ${PPLKERNELX86_AVX_FLAGS}")
set_source_files_properties(${PPLKERNELX86_FMA_SRC} PROPERTIES
    COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${AVX_ENABLED_FLAGS} ${FMA_ENABLED_FLAGS} ${PPLKERNELX86_FMA_FLAGS}")
if (PPL_USE_X86_AVX512)
    set_source_files_properties(${PPLKERNELX86_AVX512_SRC} PROPERTIES
        COMPILE_FLAGS "${SSE_ENABLED_FLAGS} ${AVX_ENABLED_FLAGS} ${FMA_ENABLED_FLAGS} ${AVX512_ENABLED_FLAGS} ${PPLKERNELX86_AVX512_FLAGS}")
else()
    list(REMOVE_ITEM PPLKERNELX86_SRC ${PPLKERNELX86_AVX512_SRC})
endif()

configure_file(include/ppl/kernel/x86/common/config.h.in ${PROJECT_BINARY_DIR}/include/ppl/kernel/x86/common/config.h @ONLY)
list(APPEND PPLKERNELX86_PUBLIC_INCLUDE_DIRECTORIES ${PROJECT_BINARY_DIR}/include)

list(APPEND PPLKERNELX86_LINK_LIBRARIES pplcommon_static)

get_filename_component(__PPLNN_SOURCE_DIR__ "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES ${__PPLNN_SOURCE_DIR__}/include ${__PPLNN_SOURCE_DIR__}/src)

add_library(pplkernelx86_static STATIC ${PPLKERNELX86_SRC})
target_compile_features(pplkernelx86_static PRIVATE cxx_std_11)
target_link_libraries(pplkernelx86_static ${PPLKERNELX86_LINK_LIBRARIES})
target_include_directories(pplkernelx86_static
    PUBLIC ${PPLKERNELX86_PUBLIC_INCLUDE_DIRECTORIES}
    PRIVATE ${PPLKERNELX86_PRIVATE_INCLUDE_DIRECTORIES} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
target_compile_definitions(pplkernelx86_static PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
target_compile_options(pplkernelx86_static PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})

if(CMAKE_COMPILER_IS_GNUCC)
    # requires a virtual destructor when having virtual functions, though it is not necessary for all cases
    target_compile_options(pplkernelx86_static PRIVATE -Werror=non-virtual-dtor)
endif()

if(PPLNN_INSTALL)
    install(TARGETS pplkernelx86_static DESTINATION lib)

    # `PPLKERNELX86_LINK_LIBRARIES` is needed for generating pplkernex86-config.cmake
    set(__PPLNN_CMAKE_CONFIG_FILE__ ${CMAKE_CURRENT_BINARY_DIR}/generated/pplkernelx86-config.cmake)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/pplkernelx86-config.cmake.in
        ${__PPLNN_CMAKE_CONFIG_FILE__}
        @ONLY)
    install(FILES ${__PPLNN_CMAKE_CONFIG_FILE__} DESTINATION lib/cmake/ppl)
    unset(__PPLNN_CMAKE_CONFIG_FILE__)
endif()

################### Test ###################

if(PPLNN_BUILD_TESTS)
    set(__PPLNN_TOOLS_DIR__ ${__PPLNN_SOURCE_DIR__}/tools)

    add_executable(test_conv2d test/test_conv2d.cpp ${__PPLNN_TOOLS_DIR__}/simple_flags.cc)
    target_include_directories(test_conv2d
        PUBLIC ${PPLKERNELX86_PUBLIC_INCLUDE_DIRECTORIES}
        PRIVATE ${PPLKERNELX86_PRIVATE_INCLUDE_DIRECTORIES} ${__PPLNN_TOOLS_DIR__} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
    target_compile_options(test_conv2d PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})
    target_compile_definitions(test_conv2d PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
    target_compile_features(test_conv2d PRIVATE cxx_std_11)
    target_link_libraries(test_conv2d PRIVATE pplkernelx86_static ${PPLKERNELX86_LINK_LIBRARIES})

    add_executable(test_gemm test/test_gemm.cpp ${__PPLNN_TOOLS_DIR__}/simple_flags.cc)
    target_include_directories(test_gemm
        PUBLIC ${PPLKERNELX86_PUBLIC_INCLUDE_DIRECTORIES} ${PPLKERNELX86_INCLUDE_DIRECTORIES}
        PRIVATE ${PPLKERNELX86_PRIVATE_INCLUDE_DIRECTORIES} ${__PPLNN_TOOLS_DIR__} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
    target_compile_options(test_gemm PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})
    target_compile_definitions(test_gemm PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
    target_compile_features(test_gemm PRIVATE cxx_std_11)
    target_link_libraries(test_gemm PRIVATE pplkernelx86_static ${PPLKERNELX86_LINK_LIBRARIES})

    add_executable(test_pd_conv2d test/test_pd_conv2d.cpp ${__PPLNN_TOOLS_DIR__}/simple_flags.cc)
    target_include_directories(test_pd_conv2d
        PUBLIC ${PPLKERNELX86_PUBLIC_INCLUDE_DIRECTORIES} ${PPLKERNELX86_INCLUDE_DIRECTORIES}
        PRIVATE ${PPLKERNELX86_PRIVATE_INCLUDE_DIRECTORIES} ${__PPLNN_TOOLS_DIR__} ${PPLNN_FRAMEWORK_INCLUDE_DIRECTORIES})
    target_compile_options(test_pd_conv2d PRIVATE ${PPLKERNELX86_COMPILE_OPTIONS})
    target_compile_definitions(test_pd_conv2d PRIVATE ${PPLKERNELX86_COMPILE_DEFINITIONS})
    target_compile_features(test_pd_conv2d PRIVATE cxx_std_11)
    target_link_libraries(test_pd_conv2d PRIVATE pplkernelx86_static ${PPLKERNELX86_LINK_LIBRARIES})

    unset(__PPLNN_TOOLS_DIR__)
endif()

unset(__PPLNN_SOURCE_DIR__)
