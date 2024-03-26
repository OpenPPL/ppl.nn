set(PPLNN_USE_X86 ON)

file(GLOB __PPLNN_X86_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/x86/*.cc)
if(PPLNN_SOURCE_EXTERNAL_X86_ENGINE_SOURCES)
    list(REMOVE_ITEM __PPLNN_X86_SRC__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/x86/default_register_resources.cc)
endif()

file(GLOB_RECURSE __PPLNN_X86_SRC_RECURSE__
    src/ppl/nn/engines/x86/kernels/*.cc
    src/ppl/nn/engines/x86/params/*.cc
    src/ppl/nn/engines/x86/optimizer/*.cc)
add_library(pplnn_x86_static STATIC
    ${__PPLNN_X86_SRC__}
    ${__PPLNN_X86_SRC_RECURSE__}
    ${PPLNN_SOURCE_EXTERNAL_X86_ENGINE_SOURCES})
unset(__PPLNN_X86_SRC_RECURSE__)
unset(__PPLNN_X86_SRC__)

hpcc_populate_dep(ppl.kernel.cpu)
target_link_libraries(pplnn_x86_static PUBLIC pplnn_basic_static pplkernelx86_static)

target_compile_definitions(pplnn_x86_static PUBLIC PPLNN_USE_X86)

target_link_libraries(pplnn_static INTERFACE pplnn_x86_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/x86 DESTINATION include/ppl/nn/engines)
    install(TARGETS pplnn_x86_static DESTINATION lib)
endif()
