#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_PARAMS_LOOP_PARAM_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_PARAMS_LOOP_PARAM_H_

#include "ppl/nn/ir/graph.h"

namespace ppl { namespace nn { namespace onnx {

struct LoopParam {
    ir::Graph graph;

    bool operator==(const LoopParam& p) const {
        return false; // has subgraph
    }
};

}}} // namespace ppl::nn::onnx

#endif
