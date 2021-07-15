if(NOT IS_X86)
    return()
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
