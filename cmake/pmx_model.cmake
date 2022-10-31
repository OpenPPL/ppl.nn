file(GLOB_RECURSE __PPLNN_MODEL_PMX_SRC__ src/ppl/nn/models/pmx/*.cc)
add_library(pplnn_pmx_static STATIC ${PPLNN_SOURCE_EXTERNAL_PMX_MODEL_SOURCES} ${__PPLNN_MODEL_PMX_SRC__})
unset(__PPLNN_MODEL_PMX_SRC__)

target_compile_definitions(pplnn_pmx_static PUBLIC PPLNN_ENABLE_PMX_MODEL)
target_link_libraries(pplnn_pmx_static PUBLIC pplnn_basic_static)

hpcc_populate_dep(flatbuffers)
target_include_directories(pplnn_pmx_static PRIVATE ${flatbuffers_SOURCE_DIR}/include)

target_link_libraries(pplnn_static INTERFACE pplnn_pmx_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/models/pmx DESTINATION include/ppl/nn/models)
    install(TARGETS pplnn_pmx_static DESTINATION lib)
endif()
