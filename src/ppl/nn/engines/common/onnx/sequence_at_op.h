#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SEQUENCE_AT_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SEQUENCE_AT_OP_H_

#include "ppl/nn/runtime/kernel_impl.h"

namespace ppl { namespace nn { namespace common {

class SequenceAtOp final {
public:
    SequenceAtOp(const ir::Node* node) : node_(node) {}
    KernelImpl* CreateKernelImpl() const;

private:
    const ir::Node* node_;
};

}}} // namespace ppl::nn::common

#endif
