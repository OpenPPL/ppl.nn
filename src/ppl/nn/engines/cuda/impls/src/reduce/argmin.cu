// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "cudakernel/reduce/argmin.h"
#include "cudakernel/common/common.h"
#include "cudakernel/reduce/reduce_kernel.h"

template <typename T>
__device__ bool __greater_or_equal(T& lhs, T& rhs) {
    return lhs >= rhs;
}

template<>
__device__ bool __greater_or_equal<half>(half& lhs, half& rhs) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hge(lhs, rhs);
#else
    return false;
#endif
}
template <typename T>
__device__ bool __greater(T& lhs, T& rhs) {
    return lhs > rhs;
}

template<>
__device__ bool __greater<half>(half& lhs, half& rhs) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hgt(lhs, rhs);
#else
    return false;
#endif
}

template <typename T>
__global__ void ppl_argmin_select_last(
    PPLReduceDimDes des,
    const T* input,
    int64_t* output)
{
    int64_t n_outer  = des.n_outer;
    int64_t n_reduce = des.n_reduce;
    int64_t n_inner  = des.n_inner;

    int64_t outer_stride = n_reduce * n_inner;
    int64_t non_reduce   = n_outer * n_inner;
    int64_t block_size   = blockDim.x * blockDim.y;
    int64_t grid_stride  = block_size * gridDim.x;
    int64_t tid          = blockIdx.x * block_size + threadIdx.y * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < non_reduce; idx += grid_stride) {
        int64_t out_idx = idx / n_inner;
        int64_t in_idx  = idx % n_inner;
        int64_t offset  = out_idx * outer_stride + in_idx;
        int64_t val     = 0;
        for (int i = 1; i < n_reduce; i++) {
            T temp1 = input[offset + val * n_inner];
            T temp2 = input[offset + i * n_inner];
            if (__greater_or_equal<T>(temp1, temp2))
                val = i;
        }
        output[idx] = val;
    }
    return;
}
template <typename T>
__global__ void ppl_argmin_select_first(
    PPLReduceDimDes des,
    const T* input,
    int64_t* output)
{
    int64_t n_outer  = des.n_outer;
    int64_t n_reduce = des.n_reduce;
    int64_t n_inner  = des.n_inner;

    int64_t outer_stride = n_reduce * n_inner;
    int64_t non_reduce   = n_outer * n_inner;
    int64_t block_size   = blockDim.x * blockDim.y;
    int64_t grid_stride  = block_size * gridDim.x;
    int64_t tid          = blockIdx.x * block_size + threadIdx.y * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < non_reduce; idx += grid_stride) {
        int64_t out_idx = idx / n_inner;
        int64_t in_idx  = idx % n_inner;
        int64_t offset  = out_idx * outer_stride + in_idx;
        int64_t val     = 0;
        for (int i = 1; i < n_reduce; i++) {
            T temp1 = input[offset + val * n_inner];
            T temp2 = input[offset + i * n_inner];
            if (__greater<T>(temp1, temp2))
                val = i;
        }
        output[idx] = val;
    }
    return;
}

ppl::common::RetCode PPLCUDAArgMinForwardImp(
    cudaStream_t stream,
    PPLReduceDimDes des,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    int32_t select_last_index)
{
    dim3 block_dim(32, BLOCKSIZE / 32);
    dim3 grid_dim(DivUp(BLOCKSIZE, des.n_outer * des.n_inner), 1);

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        if(select_last_index==1)
            ppl_argmin_select_last<half><<<grid_dim, block_dim, 0, stream>>>(des, (const half*)input, (int64_t*)output);
        else
            ppl_argmin_select_first<half><<<grid_dim, block_dim, 0, stream>>>(des, (const half*)input, (int64_t*)output);

    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        if(select_last_index==1)
            ppl_argmin_select_last<float><<<grid_dim, block_dim, 0, stream>>>(des, (const float*)input, (int64_t*)output);
        else
            ppl_argmin_select_first<float><<<grid_dim, block_dim, 0, stream>>>(des, (const float*)input, (int64_t*)output);
    }
    else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        if(select_last_index==1)
            ppl_argmin_select_last<int8_t><<<grid_dim, block_dim, 0, stream>>>(des, (const int8_t*)input, (int64_t*)output);
        else
            ppl_argmin_select_first<int8_t><<<grid_dim, block_dim, 0, stream>>>(des, (const int8_t*)input, (int64_t*)output);
    }
    else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
#undef CASE
#undef CASEPROMOTION
}