file(GLOB_RECURSE __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/*.cc)
list(REMOVE_ITEM __PPLNN_MODEL_ONNX_SRC__ ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated/onnx.pb.cc)

if(NOT TARGET libprotobuf)
    include(cmake/protobuf.cmake)
endif()

if(NOT PPLNN_ONNX_GENERATED_LIBS)
    if(PPLNN_PROTOBUF_VERSION)
        if(CMAKE_CROSSCOMPILING)
            message(FATAL_ERROR "`PPLNN_PROTOBUF_VERSION` is set to be [${PPLNN_PROTOBUF_VERSION}], but `PPLNN_ONNX_GENERATED_LIBS` is not set.")
        else() # use protoc to generate *.pb.*
            if(NOT PPLNN_PROTOC_EXECUTABLE)
                set(PPLNN_PROTOC_EXECUTABLE ${protobuf_BINARY_DIR}/protoc)
            endif()

            set(__GENERATED_DIR__ ${CMAKE_CURRENT_BINARY_DIR}/generated)
            file(MAKE_DIRECTORY ${__GENERATED_DIR__})

            set(__PROTO_DIR__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/models/onnx/proto)
            set(__ONNX_GENERATED_FILES__ "${__GENERATED_DIR__}/onnx.pb.h;${__GENERATED_DIR__}/onnx.pb.cc")
            add_custom_command(
                OUTPUT ${__ONNX_GENERATED_FILES__}
                COMMAND ${PPLNN_PROTOC_EXECUTABLE}
                ARGS --cpp_out ${__GENERATED_DIR__} -I ${__PROTO_DIR__}
                ${__PROTO_DIR__}/onnx.proto
                DEPENDS protoc ${__PROTO_DIR__}/onnx.proto)
            add_library(pplnn_onnx_generated_static STATIC ${__ONNX_GENERATED_FILES__})
            target_link_libraries(pplnn_onnx_generated_static PUBLIC libprotobuf)
            target_include_directories(pplnn_onnx_generated_static PUBLIC ${__GENERATED_DIR__})
            set(PPLNN_ONNX_GENERATED_LIBS pplnn_onnx_generated_static)

            unset(__ONNX_GENERATED_FILES__)
            unset(__PROTO_DIR__)
            unset(__GENERATED_DIR__)
        endif()
    else() # use defualt protobuf version
        add_library(pplnn_onnx_pb_generated_static ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated/onnx.pb.cc)
        target_link_libraries(pplnn_onnx_pb_generated_static PUBLIC libprotobuf)
        target_include_directories(pplnn_onnx_pb_generated_static PUBLIC
            ${protobuf_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated)
        set(PPLNN_ONNX_GENERATED_LIBS pplnn_onnx_pb_generated_static)
    endif()
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
