#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SPLIT_TO_SEQUENCE_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SPLIT_TO_SEQUENCE_OP_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/engines/common/onnx/split_to_sequence_kernel.h"

namespace ppl { namespace nn { namespace common {

class SplitToSequenceOp final {
public:
    SplitToSequenceOp(const ir::Node* node) : node_(node), axis_(0), keepdims_(1) {}
    void Init(uint64_t axis, uint64_t keepdims, const SplitToSequenceKernel::SplitFunc& f);
    KernelImpl* CreateKernelImpl() const;

    static ppl::common::RetCode GenericSplitFunc(uint64_t dims_before_axis, uint64_t dims_from_axis,
                                                 uint64_t dims_after_axis, uint32_t dims_of_chunk,
                                                 uint32_t element_size, Device* device, BufferDesc* src_cursor,
                                                 BufferDesc* dst);

private:
    const ir::Node* node_;
    uint64_t axis_, keepdims_;
    SplitToSequenceKernel::SplitFunc split_func_;
};

}}} // namespace ppl::nn::common

#endif
