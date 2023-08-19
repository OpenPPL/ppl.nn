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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CONSTANT_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CONSTANT_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include "ppl/common/types.h"
#include "ppl/common/mmap.h"
#include <cstring>
#include <vector>

namespace ppl { namespace nn { namespace onnx {

struct ConstantParam final : public ir::TypedAttr<ConstantParam> {
    ppl::common::datatype_t data_type;
    std::vector<int64_t> dims;
    ppl::common::Mmap data;

    bool operator==(const ConstantParam& p) const {
        return (data_type == p.data_type && dims == p.dims &&
                (data.GetSize() == p.data.GetSize() && memcmp(data.GetData(), p.data.GetData(), data.GetSize())));
    }
};

}}} // namespace ppl::nn::onnx

#endif
