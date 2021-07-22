file(GLOB PPLNN_TOOL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/pplnn.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_flags.cc)

add_executable(pplnn ${PPLNN_TOOL_SRC})
target_link_libraries(pplnn PUBLIC pplnn_static)

add_executable(add_all_tensors_to_output
    ${CMAKE_CURRENT_SOURCE_DIR}/add_all_tensors_to_output.cc
    ${PROJECT_SOURCE_DIR}/src/ppl/nn/models/onnx/generated/onnx.pb.cc)
target_include_directories(add_all_tensors_to_output PRIVATE
    ${protobuf_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(add_all_tensors_to_output PRIVATE
    libprotobuf)

add_executable(onnx_model_is_supported
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx_model_is_supported.cc)
target_link_libraries(onnx_model_is_supported PRIVATE pplnn_static)
