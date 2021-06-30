#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_PARAMS_IF_PARAM_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_PARAMS_IF_PARAM_H_

#include "ppl/nn/ir/graph.h"

namespace ppl { namespace nn { namespace onnx {

struct IfParam {
    ir::Graph then_branch;
    ir::Graph else_branch;
    // indices in ir::Node::GetExtraInput()
    std::vector<uint32_t> then_extra_input_indices_in_parent_node;
    std::vector<uint32_t> else_extra_input_indices_in_parent_node;

    bool operator==(const IfParam& p) const {
        return false; // has subgraph
    }
};

}}} // namespace ppl::nn::onnx

#endif
