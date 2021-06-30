#ifndef _ST_HPC_PPL_NN_RUNTIME_TENSOR_SEQUENCE_H_
#define _ST_HPC_PPL_NN_RUNTIME_TENSOR_SEQUENCE_H_

#include "ppl/nn/runtime/sequence.h"
#include "ppl/nn/common/tensor_buffer_info.h"

namespace ppl { namespace nn {

typedef Sequence<TensorBufferInfo> TensorSequence;

template <>
struct EdgeObjectType<TensorSequence> final {
    static const uint32_t value = EdgeObject::T_TENSOR_SEQUENCE;
};

}} // namespace ppl::nn

#endif
