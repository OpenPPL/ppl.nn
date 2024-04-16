file(GLOB_RECURSE __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/*.cc)
list(REMOVE_ITEM __PPLNN_MODEL_ONNX_SRC__ ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated/onnx.pb.cc)

if(PPLNN_PROTOBUF_VERSION AND NOT PPLNN_ONNX_GENERATED_LIBS)
   message(FATAL_ERROR "`PPLNN_PROTOBUF_VERSION` is set to be [${PPLNN_PROTOBUF_VERSION}], but `PPLNN_ONNX_GENERATED_LIBS` is not set.")
endif()

if(NOT TARGET libprotobuf)
    include(cmake/protobuf.cmake)
endif()

# use default *.pb.* files
if(NOT PPLNN_ONNX_GENERATED_LIBS)
    add_library(pplnn_onnx_pb_generated_static ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated/onnx.pb.cc)
    target_link_libraries(pplnn_onnx_pb_generated_static PUBLIC libprotobuf)
    target_include_directories(pplnn_onnx_pb_generated_static PUBLIC
        ${protobuf_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated)
    set(PPLNN_ONNX_GENERATED_LIBS pplnn_onnx_pb_generated_static)
endif()

# if external sources are set, remove `default_register_resources.cc`
if(PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES)
    list(REMOVE_ITEM __PPLNN_MODEL_ONNX_SRC__ ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/default_register_resources.cc)
endif()

add_library(pplnn_onnx_static STATIC ${__PPLNN_MODEL_ONNX_SRC__} ${PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES})

unset(__PPLNN_MODEL_ONNX_SRC__)

target_compile_definitions(pplnn_onnx_static PUBLIC PPLNN_ENABLE_ONNX_MODEL)
target_link_libraries(pplnn_onnx_static PUBLIC pplnn_basic_static)
target_link_libraries(pplnn_onnx_static PRIVATE ${PPLNN_ONNX_GENERATED_LIBS})

target_link_libraries(pplnn_static INTERFACE pplnn_onnx_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/models/onnx DESTINATION include/ppl/nn/models)
    install(TARGETS pplnn_onnx_static ${PPLNN_ONNX_GENERATED_LIBS} DESTINATION lib)
endif()
