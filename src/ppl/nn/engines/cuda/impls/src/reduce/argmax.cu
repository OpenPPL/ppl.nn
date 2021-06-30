#include "cudakernel/reduce/argmax.h"
#include "cudakernel/reduce/reduce_kernel.h"

template <typename T>
__global__ void ppl_argmax(
    PPLReduceDimDes des,
     const T* input,
     int64_t* output)
{
    int64_t n_outer    = des.n_outer;
    int64_t n_reduce   = des.n_reduce;
    int64_t n_inner    = des.n_inner;

    int64_t outer_stride       = n_reduce * n_inner;
    int64_t non_reduce         = n_outer * n_inner;
    int64_t block_size         = blockDim.x * blockDim.y;
    int64_t grid_stride        = block_size * gridDim.x;
    int64_t tid                = blockIdx.x * block_size + threadIdx.y * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < non_reduce; idx += grid_stride) {
        int64_t out_idx = idx / n_inner;
        int64_t in_idx  = idx % n_inner;
        int64_t offset  = out_idx * outer_stride + in_idx;
        int64_t val     = 0;
        for (int i = 1; i < n_reduce; i++) {
            float temp1 = input[offset + val * n_inner];
            float temp2 = input[offset + i * n_inner];
            if (temp1 <= temp2)
                val = i;
        }
        output[idx] = val;
    }
    return;
}

ppl::common::RetCode PPLCUDAArgMaxForwardImp(
    cudaStream_t stream,
    PPLReduceDimDes des,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    dim3 block_dim(32, BLOCKSIZE / 32);
    dim3 grid_dim(DivUp(BLOCKSIZE, des.n_outer * des.n_inner), 1);

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        ppl_argmax<half><<<grid_dim, block_dim, 0, stream>>>(des, (const half*)input, (int64_t*)output);
    }
    else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        ppl_argmax<float><<<grid_dim, block_dim, 0, stream>>>(des, (const float*)input, (int64_t*)output);
    }
    else {
        return ppl::common::RC_UNSUPPORTED;
    }
    
    return ppl::common::RC_SUCCESS;
#undef CASE
#undef CASEPROMOTION
}