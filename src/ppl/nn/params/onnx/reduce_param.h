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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_REDUCE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_REDUCE_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace onnx {

struct ReduceParam final : public ir::TypedAttr<ReduceParam> {
    typedef enum {
        ReduceSum = 0,
        ReduceMax = 1,
        ReduceMin = 2,
        ReduceProd = 3,
        ReduceMean = 4,
        ReduceL2 = 5,
        ReduceUnknown = 6,
    } reduce_type_t;

    reduce_type_t type;
    int32_t keepdims;
    std::vector<int32_t> axes;

    bool operator==(const ReduceParam& p) const {
        return (type == p.type && keepdims == p.keepdims && axes == p.axes);
    }
};

}}} // namespace ppl::nn::onnx

#endif
