#include "ppl/nn/models/onnx/parsers/parse_depth_to_space_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseDepthToSpaceParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::DepthToSpaceParam*>(arg);

    std::string mode = utils::GetNodeAttrByKey<std::string>(pb_node, "mode", "DCR");
    if (mode != "DCR" && mode != "CRD") {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (mode == "DCR") {
        param->mode = ppl::nn::common::DepthToSpaceParam::DCR;
    } else {
        param->mode = ppl::nn::common::DepthToSpaceParam::CRD;
    }

    int32_t blocksize = utils::GetNodeAttrByKey<int32_t>(pb_node, "blocksize", 0);
    if (blocksize <= 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    param->blocksize = blocksize;

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
