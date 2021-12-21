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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_UTILS_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_UTILS_H_

#include "ppl/nn/common/tensor_shape.h"

namespace ppl { namespace nn { namespace x86 {

inline bool TensorShapeEqual(const TensorShape &a, const TensorShape &b) {
    if (a.GetDataType() != b.GetDataType()) {
        return false;
    }
    if (a.GetDataFormat() != b.GetDataFormat()) {
        return false;
    }
    if (a.IsEmpty() && b.IsEmpty()) {
        return true;
    }
    if (a.IsScalar() && b.IsScalar()) {
        return true;
    }
    if (a.GetDimCount() != b.GetDimCount()) {
        return false;
    }
    for (uint32_t i = 0; i < a.GetDimCount(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

}}}; // namespace

#endif
