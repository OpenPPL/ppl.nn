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

#include "cudakernel/memory/pad.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/common/memory_utils.h"
#include "cudakernel/math/math.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

template <typename T>
__global__ void ppl_cukernel_range(
    int64_t num_elems,
    const T* start,
    const T* delta,
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    output[index] = start[0] + index * delta[0];
}

template <>
__global__ void ppl_cukernel_range<half>(
    int64_t num_elems,
    const half* start,
    const half* delta,
    half* output)
{
    typedef Math<half, half, half> OpMath;
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    output[index] = OpMath::add(start[0], OpMath::mul(delta[0], __ll2half_rn(index)));
}

ppl::common::RetCode PPLCUDARangeForwardImp(
    cudaStream_t stream,
    const void* start,
    const void* delta,
    ppl::nn::TensorShape* output_shape,
    void* output)
{
    int block_size     = 256;
    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int grid_size      = (num_elems + block_size - 1) / block_size;
    switch (output_shape->GetDataType()) {
        case ppl::common::DATATYPE_FLOAT32:
            ppl_cukernel_range<float><<<grid_size, block_size, 0, stream>>>(num_elems, (float*)start, (float*)delta, (float*)output);
            break;
        case ppl::common::DATATYPE_FLOAT16:
            ppl_cukernel_range<half><<<grid_size, block_size, 0, stream>>>(num_elems, (half*)start, (half*)delta, (half*)output);
            break;
        case ppl::common::DATATYPE_INT64:
            ppl_cukernel_range<int64_t><<<grid_size, block_size, 0, stream>>>(num_elems, (int64_t*)start, (int64_t*)delta, (int64_t*)output);
            break;
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
    // ppl_cukernel_range<<<grid_size, block_size, 0, stream>>>(num_elems, start, delta, (T*)output);
    return ppl::common::RC_SUCCESS;
}
