hpcc_populate_dep(protobuf)

if(NOT PPL_LLM_PROTOC_EXECUTABLE)
    set(PPL_LLM_PROTOC_EXECUTABLE ${protobuf_BINARY_DIR}/protoc)
endif()

if(NOT PPLNN_ONNX_GENERATED_LIBS)
    # generate new onnx.pb.* for pplnn
    set(__LLM_GENERATED_DIR__ ${CMAKE_CURRENT_BINARY_DIR}/generated)
    file(MAKE_DIRECTORY ${__LLM_GENERATED_DIR__})

    set(__PROTO_DIR__ ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/models/onnx/proto)
    set(__ONNX_GENERATED_FILES__ "${__LLM_GENERATED_DIR__}/onnx.pb.h;${__LLM_GENERATED_DIR__}/onnx.pb.cc")
    add_custom_command(
        OUTPUT ${__ONNX_GENERATED_FILES__}
        COMMAND ${PPL_LLM_PROTOC_EXECUTABLE}
        ARGS --cpp_out ${__LLM_GENERATED_DIR__} -I ${__PROTO_DIR__}
        ${__PROTO_DIR__}/onnx.proto
        DEPENDS protoc ${__PROTO_DIR__}/onnx.proto)
    add_library(pplnn_onnx_generated_static STATIC ${__ONNX_GENERATED_FILES__})
    target_link_libraries(pplnn_onnx_generated_static PUBLIC libprotobuf)
    target_include_directories(pplnn_onnx_generated_static PUBLIC ${__LLM_GENERATED_DIR__})
    set(PPLNN_ONNX_GENERATED_LIBS pplnn_onnx_generated_static)

    unset(__ONNX_GENERATED_FILES__)
    unset(__PROTO_DIR__)
    unset(__LLM_GENERATED_DIR__)
endif()

include(${HPCC_DEPS_DIR}/hpcc/cmake/cuda-common.cmake)

hpcc_populate_dep(ppl.llm.kernel.cuda)

file(GLOB_RECURSE __SRC__ src/ppl/nn/engines/llm_cuda/*.cc)
add_library(ppl_llm_cuda_static ${__SRC__})
target_link_libraries(ppl_llm_cuda_static PUBLIC pplnn_basic_static pplkernelcuda_static)
target_include_directories(ppl_llm_cuda_static PUBLIC include src)
target_compile_definitions(ppl_llm_cuda_static PUBLIC PPLNN_USE_LLM_CUDA)

if(PPLNN_CUDA_ENABLE_NCCL)
    target_compile_definitions(ppl_llm_cuda_static PUBLIC PPLNN_CUDA_ENABLE_NCCL)
endif()

unset(__SRC__)

target_link_libraries(pplnn_static INTERFACE ppl_llm_cuda_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/engines/llm_cuda DESTINATION include/ppl/nn/engines)
    install(TARGETS ppl_llm_cuda_static DESTINATION lib)
endif()
