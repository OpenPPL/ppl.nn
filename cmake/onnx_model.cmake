file(GLOB_RECURSE __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/*.cc)

if(PPLNN_PROTOBUF_VERSION AND NOT PPLNN_ONNX_GENERATED_DIR)
   message(FATAL_ERROR "`PPLNN_PROTOBUF_VERSION` is set to be [${PPLNN_PROTOBUF_VERSION}], but `PPLNN_ONNX_GENERATED_DIR` is not set.")
endif()

# replace default *.pb.* files
if(PPLNN_ONNX_GENERATED_DIR)
    list(REMOVE_ITEM __PPLNN_MODEL_ONNX_SRC__ ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated/onnx.pb.cc)
    list(APPEND __PPLNN_MODEL_ONNX_SRC__ ${PPLNN_ONNX_GENERATED_DIR}/onnx.pb.cc)
endif()

# if external sources are set, remove `default_register_resources.cc`
if(PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES)
    list(REMOVE_ITEM __PPLNN_MODEL_ONNX_SRC__ ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/default_register_resources.cc)
endif()

add_library(pplnn_onnx_static STATIC ${__PPLNN_MODEL_ONNX_SRC__} ${PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES})
unset(__PPLNN_MODEL_ONNX_SRC__)

if(PPLNN_ONNX_GENERATED_DIR)
    target_include_directories(pplnn_onnx_static PUBLIC ${PPLNN_ONNX_GENERATED_DIR})
else()
    target_include_directories(pplnn_onnx_static PUBLIC src/ppl/nn/models/onnx/generated)
endif()

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
