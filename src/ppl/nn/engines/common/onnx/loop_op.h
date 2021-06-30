#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_LOOP_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_LOOP_OP_H_

#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/engines/common/onnx/loop_kernel.h"
#include "ppl/nn/models/onnx/params/loop_param.h"

namespace ppl { namespace nn { namespace utils {
struct SharedResource;
}}} // namespace ppl::nn::utils

namespace ppl { namespace nn { namespace common {

class LoopOp final {
public:
    LoopOp(const ir::Node* node) : node_(node), resource_(nullptr) {}
    ppl::common::RetCode Init(utils::SharedResource*, ppl::nn::onnx::LoopParam*, LoopConcatOutputFunc);
    KernelImpl* CreateKernelImpl() const;

private:
    const ir::Node* node_;
    utils::SharedResource* resource_;
    ir::Graph graph_;
    RuntimeGraphInfo graph_info_;
    RuntimeAuxInfo aux_info_;
    LoopConcatOutputFunc concat_output_func_;
};

}}} // namespace ppl::nn::common

#endif
