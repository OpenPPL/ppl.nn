#include "ppl/nn/models/onnx/parsers/parse_ppl_channel_shuffle_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseChannelShuffleParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto channel_shuffle_param = static_cast<ppl::nn::common::ChannelShuffleParam*>(arg);

    int32_t group = utils::GetNodeAttrByKey<int32_t>(pb_node, "group", 1);
    channel_shuffle_param->group = utils::ConvertOnnxDataTypeToPplDataType(group);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
