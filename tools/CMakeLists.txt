file(GLOB PPLNN_TOOL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/pplnn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_flags.cc)

add_executable(pplnn ${PPLNN_TOOL_SRC})
target_link_libraries(pplnn PRIVATE pplnn_static)
target_compile_features(pplnn PRIVATE cxx_std_11)

if(PPLNN_ENABLE_ONNX_MODEL)
    add_executable(add_all_tensors_to_output
        ${CMAKE_CURRENT_SOURCE_DIR}/add_all_tensors_to_output.cc
        ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated/onnx.pb.cc)
    target_include_directories(add_all_tensors_to_output PRIVATE
        ${protobuf_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(add_all_tensors_to_output PRIVATE
        libprotobuf)
    target_compile_features(add_all_tensors_to_output PRIVATE cxx_std_11)

    add_executable(onnx_serializer onnx_serializer.cc)
    target_link_libraries(onnx_serializer PRIVATE pplnn_static)
endif()
