#include "cudakernel/memory/constant_of_shape.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/common/types.h"
#include <cuda_runtime.h>

template <typename T>
__global__ void ppl_cukernel_constant_of_shape(
    int64_t num_elems,
    const T *pre_set_value,
    T *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    output[index] = pre_set_value[0];
}

ppl::common::RetCode PPLCUDAConstantOfShapeForwardImp(
    cudaStream_t stream,
    const void* pre_set_value,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    int64_t num_elems = output_shape->GetElementsIncludingPadding();
    int block_size = 256;
    int grid_size  = (num_elems + block_size - 1) / block_size;

#define SWITCH_CASE(TYPE)                                                      \
    case sizeof(TYPE): {                                                       \
        ppl_cukernel_constant_of_shape<<<grid_size, block_size, 0, stream>>>( \
            num_elems, (const TYPE *)pre_set_value, (TYPE *)output);           \
        return ppl::common::RC_SUCCESS;                                        \
    }

    switch (ppl::common::GetSizeOfDataType(output_shape->GetDataType())) {
        SWITCH_CASE(int8_t);
        SWITCH_CASE(int16_t);
        SWITCH_CASE(int32_t);
        SWITCH_CASE(int64_t);
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}