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

#include "cudakernel/nn/one_hot.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>

template <typename T>
__global__ void ppl_cukernel_one_hot(const int64_t* incides, T on_value, T* output, int64_t outer, int64_t depth, int64_t inner){

        int tid = threadIdx.x;
        int outer_id = blockIdx.x;
        int inner_id = blockIdx.y;

        for(int id = tid; id < depth; id += blockDim.x){
            if(id < depth){
                uint64_t output_offset = outer_id * depth * inner + id * inner + inner_id;
                uint64_t incides_offset = outer_id * inner + inner_id;
                int64_t on_index = incides[incides_offset];
                if(id == on_index)
                    output[output_offset] = on_value;
            }
        }
}

template <typename T>
T  get_mask_value(const T* values, T* output, uint64_t nelem){
    T* host_value = (T*)malloc(sizeof(T)*2);
    cudaMemcpy(host_value, (T*)values, sizeof(T) * 2, cudaMemcpyDeviceToHost);
    T off_value = host_value[0];
    cudaMemset((void*)output, off_value, sizeof(T)*nelem); // set off value

    return host_value[1];
}

ppl::common::RetCode PPLCUDAOneHotForwardImp(
    cudaStream_t stream,
    const void* indices,
    ppl::common::TensorShape* values_shape,
    const void* values,
    ppl::common::TensorShape* output_shape,
    void* output,
    uint32_t real_axis)
{
    uint64_t num_elems = output_shape->CalcElementsExcludingPadding();

    auto outer = output_shape->CalcElementsToDimensionExcludingPadding(real_axis);
    auto depth_val = output_shape->GetDim(real_axis);
    auto inner = output_shape->CalcElementsFromDimensionExcludingPadding(real_axis+1);

    int block_size     = 256;
    dim3 block(block_size);
    dim3 grid(outer, inner);

    auto datatype = values_shape->GetDataType();
    auto dataformat = output_shape->GetDataFormat();

    switch(datatype){
        case ppl::common::DATATYPE_FLOAT32:{
            float on_value = get_mask_value<float>((const float*)values, (float*)output, num_elems);
            ppl_cukernel_one_hot<float><<<grid, block, 0, stream>>>((const int64_t*)indices, on_value, (float*)output, outer, depth_val, inner);
            break;
        }
        case ppl::common::DATATYPE_FLOAT16:{
            half on_value = get_mask_value<half>((const half*)values, (half*)output, num_elems);
            ppl_cukernel_one_hot<half><<<grid, block, 0, stream>>>((const int64_t*)indices, on_value, (half*)output, outer, depth_val, inner);
            break;
        }
        case ppl::common::DATATYPE_INT64:{
            int64_t on_value = get_mask_value<int64_t>((const int64_t*)values, (int64_t*)output, num_elems);
            ppl_cukernel_one_hot<int64_t><<<grid, block, 0, stream>>>((const int64_t*)indices, on_value, (int64_t*)output, outer, depth_val, inner);
            break;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }


    return ppl::common::RC_SUCCESS;
}
