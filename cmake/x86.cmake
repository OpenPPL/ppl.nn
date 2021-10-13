file(GLOB_RECURSE PPLNN_X86_SRC src/ppl/nn/engines/x86/*.cc)
list(APPEND PPLNN_SOURCES ${PPLNN_X86_SRC})

add_subdirectory(src/ppl/nn/engines/x86/impls)
list(APPEND PPLNN_LINK_LIBRARIES pplkernelx86_static)

list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_X86)
