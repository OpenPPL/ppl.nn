file(GLOB_RECURSE __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/*.cc)
# if external sources are set, remove `default_register_resources.cc`
if(PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES)
    list(REMOVE_ITEM __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/default_register_resources.cc)
endif()
add_library(pplnn_onnx_static STATIC ${__PPLNN_MODEL_ONNX_SRC__} ${PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES})
unset(__PPLNN_MODEL_ONNX_SRC__)

target_compile_definitions(pplnn_onnx_static PUBLIC PPLNN_ENABLE_ONNX_MODEL)
target_link_libraries(pplnn_onnx_static PUBLIC pplnn_basic_static)

include(cmake/protobuf.cmake)
target_link_libraries(pplnn_onnx_static PUBLIC libprotobuf)
target_include_directories(pplnn_onnx_static PRIVATE ${protobuf_SOURCE_DIR}/src)

target_link_libraries(pplnn_static INTERFACE pplnn_onnx_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/models/onnx DESTINATION include/ppl/nn/models)
    install(TARGETS pplnn_onnx_static DESTINATION lib)
endif()
