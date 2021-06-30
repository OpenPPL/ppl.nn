#ifndef _ST_HPC_PPL_NN_TESTS_COMMON_TENSOR_BUFFER_INFO_TOOLS_H_
#define _ST_HPC_PPL_NN_TESTS_COMMON_TENSOR_BUFFER_INFO_TOOLS_H_

#include "ppl/nn/common/tensor_buffer_info.h"
#include <random>
using namespace ppl::common;

namespace ppl { namespace nn { namespace test {

static inline int64_t GenRandDim() {
    static const uint32_t max_dim = 640;
    return rand() % max_dim + 1;
}

static inline TensorBufferInfo GenRandomTensorBufferInfo(Device* device) {
    TensorBufferInfo info;

    TensorShape shape;
    shape.Reshape({1, 3, GenRandDim(), GenRandDim()});
    shape.SetDataType(DATATYPE_FLOAT32);
    shape.SetDataFormat(DATAFORMAT_NDARRAY);
    info.Reshape(shape);

    info.SetDevice(device);
    info.ReallocBuffer();
    return info;
}

}}} // namespace ppl::nn::test

#endif
