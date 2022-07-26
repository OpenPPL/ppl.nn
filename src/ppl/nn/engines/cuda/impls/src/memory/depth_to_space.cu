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

#include "cudakernel/memory/depth_to_space.h"
#include "cudakernel/memory/transpose.h"
#include "cudakernel/memory/subpixel.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDADepthToSpaceForwardImp(
    int device_id,
    cudaStream_t stream,
    ppl::nn::onnx::DepthToSpaceParam param,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    // transpose case
    if (param.mode == ppl::nn::onnx::DepthToSpaceParam::DCR) {
        int num_transpose_dim = 6; // from depth_to_space definition
        ppl::nn::onnx::TransposeParam trans_param;
        trans_param.perm[0] = 0;
        trans_param.perm[1] = 3;
        trans_param.perm[2] = 4;
        trans_param.perm[3] = 1;
        trans_param.perm[4] = 5;
        trans_param.perm[5] = 2;
        ppl::nn::TensorShape input_shape_trans(*input_shape);
        ppl::nn::TensorShape output_shape_trans(*output_shape);
        input_shape_trans.SetDimCount(num_transpose_dim);
        output_shape_trans.SetDimCount(num_transpose_dim);
        input_shape_trans.SetDim(1, param.blocksize);
        input_shape_trans.SetDim(2, param.blocksize);
        int trans_channel = input_shape->GetDim(1) / param.blocksize / param.blocksize;
        input_shape_trans.SetDim(3, trans_channel);
        input_shape_trans.SetDim(4, input_shape->GetDim(2));
        input_shape_trans.SetDim(5, input_shape->GetDim(3));
        for (int it = 0; it < num_transpose_dim; ++it) {
            output_shape_trans.SetDim(it, input_shape_trans.GetDim(trans_param.perm[it]));
        }
        return PPLCUDATransposeForwardImp(device_id,
                                          stream,
                                          trans_param,
                                          &input_shape_trans,
                                          input,
                                          &output_shape_trans,
                                          output);
    } else { // subpixel up case
        return PPLCUDASubpixelUpForwardImp(stream,
                                           param.blocksize,
                                           input_shape,
                                           input,
                                           output_shape,
                                           output);
    }
}
