#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_PARSERS_PARSE_MMCV_ROIALIGN_PARAM_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_PARSERS_PARSE_MMCV_ROIALIGN_PARAM_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/params/mmcv/mmcv_roialign_param.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/models/onnx/generated/onnx.pb.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseMMCVROIAlignParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*);

}}} // namespace ppl::nn::onnx

#endif
