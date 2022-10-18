#include "cudakernel/unary/neg.h"
#include "ppl/nn/engines/cuda/impls/src/reformat/cvt_int8_float.cuh"
#include <cuda_fp16.h>



template <typename DataT>
__global__ void ppl_cukernel_unary_any(
    const uint64_t num_elems,
    const DataT* input,
    DataT* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    DataT in_val  = input[index];
    output[index] = -in_val;
#endif
}


ppl::common::RetCode PPLCUDANegForwardImp(                                                                                       
    cudaStream_t stream,                                                                                                                   
    const ppl::nn::TensorShape* input_shape,                                                                                               
    const void* input,                                                                                                                     
    const ppl::nn::TensorShape* output_shape,                                                                                              
    void* output,                                                                                                                          
    const ppl::nn::cuda::QuantParamCuda* qparam)                                                                                           
{                                                                                                                                          
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();                                                                     
    int block_size     = 256;                                                                                                              
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;                                                                        
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {                                                                    
        ppl_cukernel_unary_any<float><<<grid_size, block_size, 0, stream>>>(num_elems, (const float*)input, (float*)output); 
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {                                                             
        ppl_cukernel_unary_any<half><<<grid_size, block_size, 0, stream>>>(num_elems, (const half*)input, (half*)output);    
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {                                                                
        ppl_cukernel_unary_any<int8_t><<<grid_size, block_size, 0, stream>>>(num_elems, (const int8_t*)input, (int8_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT16) {
        ppl_cukernel_unary_any<int16_t><<<grid_size, block_size, 0, stream>>>(num_elems, (const int16_t*)input, (int16_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT32) {
        ppl_cukernel_unary_any<int32_t><<<grid_size, block_size, 0, stream>>>(num_elems, (const int32_t*)input, (int32_t*)output);
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT64) {
        ppl_cukernel_unary_any<int64_t><<<grid_size, block_size, 0, stream>>>(num_elems, (const int64_t*)input, (int64_t*)output);
    } else {                                                                                                                               
        return ppl::common::RC_UNSUPPORTED;                                                                                                
    }                                                                                                                                      
    return ppl::common::RC_SUCCESS;                                                                                                 
}