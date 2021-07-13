if(NOT IS_X86)
    return()
endif()

option(PPLNN_USE_X86_AVX512 "Build x86 kernel with avx512 support." ON)

if(NOT ((CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9.2) OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6.0.0) OR (MSVC_VERSION GREATER 1910)))
    set(PPLNN_USE_X86_AVX512 OFF) # compiler does not support avx512
endif()
if (PPLNN_USE_X86_AVX512)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_X86_AVX512)
endif()

file(GLOB_RECURSE PPLNN_X86_SRC src/ppl/nn/engines/x86/*.cc)
list(APPEND PPLNN_SOURCES ${PPLNN_X86_SRC})

add_subdirectory(src/ppl/nn/engines/x86/impls)
list(APPEND PPLNN_LINK_LIBRARIES PPLKernelX86)

list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_X86)
if (IS_X64)
    list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_X64)
endif()

# ----- x86 engine ----- #

file(GLOB PPLNN_X86_PUBLIC_HEADERS
    src/ppl/nn/engines/x86/x86_options.h
    src/ppl/nn/engines/x86/engine_factory.h)
install(FILES ${PPLNN_X86_PUBLIC_HEADERS}
    DESTINATION include/ppl/nn/engines/x86)
