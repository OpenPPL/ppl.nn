#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_IF_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_IF_OP_H_

#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/models/onnx/params/if_param.h"

namespace ppl { namespace nn { namespace utils {
struct SharedResource;
}}} // namespace ppl::nn::utils

namespace ppl { namespace nn { namespace common {

class IfOp final {
public:
    IfOp(const ir::Node* node) : node_(node), resource_(nullptr) {}
    ppl::common::RetCode Init(utils::SharedResource*, ppl::nn::onnx::IfParam*);
    KernelImpl* CreateKernelImpl() const;

private:
    const ir::Node* node_;
    utils::SharedResource* resource_;

    ir::Graph then_graph_;
    RuntimeGraphInfo then_info_;
    RuntimeAuxInfo then_aux_info_;
    std::vector<uint32_t> extra_inputs_of_then_graph_; // indices in ir::Node::GetExtraInput()

    ir::Graph else_graph_;
    RuntimeGraphInfo else_info_;
    RuntimeAuxInfo else_aux_info_;
    std::vector<uint32_t> extra_inputs_of_else_graph_; // indices in ir::Node::GetExtraInput()
};

}}} // namespace ppl::nn::common

#endif
