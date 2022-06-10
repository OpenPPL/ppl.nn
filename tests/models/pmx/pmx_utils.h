#include "gtest/gtest.h"

class PmxTest : public testing::Test {};

#define DEFINE_ARG(param_name, arg_name)         \
    ppl::nn::onnx::param_name arg_name##_param1; \
    ppl::nn::onnx::param_name arg_name##_param3; \
    const ppl::nn::pmx::onnx::param_name* arg_name##_param2 = nullptr

#define MAKE_BUFFER(param_name, arg_name)                                                            \
    flatbuffers::FlatBufferBuilder flatbuffer_builder;                                               \
    flatbuffers::Offset<ppl::nn::pmx::onnx::param_name> serlize_param =                              \
        Serialize##param_name(arg_name##_param1, &flatbuffer_builder);                               \
    flatbuffers::Offset<ppl::nn::pmx::onnx::OpParam> root_param = ppl::nn::pmx::onnx::CreateOpParam( \
        flatbuffer_builder, ppl::nn::pmx::onnx::OpParamType_##param_name, serlize_param.Union());    \
    flatbuffer_builder.Finish(root_param);                                                           \
    auto tmp_buffer = flatbuffer_builder.Release();                                                  \
    const ppl::nn::pmx::onnx::OpParam* op_param = ppl::nn::pmx::onnx::GetOpParam(tmp_buffer.data()); \
    arg_name##_param2 = op_param->value_as_##param_name();                                           \
    Deserialize##param_name(*arg_name##_param2, &arg_name##_param3);
