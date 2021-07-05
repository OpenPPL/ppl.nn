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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_PARAMS_CONVOLUTION_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_PARAMS_CONVOLUTION_PARAM_H_

#include <functional>

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

namespace ppl { namespace nn { namespace x86 {

struct Convolution2DParam {
    ppl::kernel::x86::conv2d_fp32_param param;
    ppl::kernel::x86::conv2d_fp32_algo_info algo_info;
    ppl::kernel::x86::conv2d_fp32_manager* mgr = nullptr;
    ppl::kernel::x86::conv2d_fp32_manager* fallback_mgr = nullptr;
    std::function<bool(const TensorImpl*, const TensorImpl*, const ppl::kernel::x86::conv2d_fp32_param*)>
        infer_fallback_func;
};

}}}; // namespace ppl::nn::x86

#endif