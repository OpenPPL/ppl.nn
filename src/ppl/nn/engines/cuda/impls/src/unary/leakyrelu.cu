#include "cudakernel/unary/leakyrelu.h"
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
template <typename DataT>
__device__ __inline__ DataT ppl_scalar_leakyrelu(const DataT& in_val, float alpha);

template <>
__device__ __inline__ float ppl_scalar_leakyrelu<float>(const float& in_val, float alpha)
{
    float res;
    res = (in_val > 0) ? in_val : alpha * in_val;
    return res;
}

template <>
__device__ __inline__ half ppl_scalar_leakyrelu<half>(const half& in_val, float alpha)
{
    half res;
    res = __hgt(in_val, 0) ? in_val : __hmul((half)alpha, in_val);
    return res;
}
#endif

template <typename DataT>
__global__ void ppl_cukernel_unary_leakyrelu(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output,
    float alpha)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = ppl_scalar_leakyrelu<DataT>(in_val, alpha);
#endif
}

ppl::common::RetCode PPLCUDAUnaryLeakyReluForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    float alpha)
{
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int block_size     = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ppl_cukernel_unary_leakyrelu<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output, alpha);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ppl_cukernel_unary_leakyrelu<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output, alpha);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}
