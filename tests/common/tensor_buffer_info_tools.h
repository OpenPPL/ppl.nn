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
