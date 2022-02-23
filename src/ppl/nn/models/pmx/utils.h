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

#ifndef _ST_HPC_PPL_NN_MODELS_PMX_UTILS_H_
#define _ST_HPC_PPL_NN_MODELS_PMX_UTILS_H_

#include "flatbuffers/flatbuffers.h"

namespace ppl { namespace nn { namespace pmx { namespace utils {

template <typename T1, typename T2>
static void Fbvec2Stdvec(const flatbuffers::Vector<T1>* src, std::vector<T2>* dst) {
    dst->resize(src->size());
    for (uint32_t i = 0; i < src->size(); ++i) {
        dst->at(i) = src->Get(i);
    }
}

}}}} // namespace ppl::nn::pmx::utils

#endif
