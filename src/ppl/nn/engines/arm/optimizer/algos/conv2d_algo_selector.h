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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_ALGOS_CONV2D_ALGO_SELECTOR_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_ALGOS_CONV2D_ALGO_SELECTOR_H_

#include "ppl/kernel/arm_server/conv2d/neon/conv2d.h"

namespace ppl { namespace nn { namespace arm {

class conv2d_algo_selector {
public:
    // algo selection is simple but fast. return the final algo manager.
    static ppl::kernel::arm_server::neon::conv2d_offline_manager *fast_gen_algo(
        const ppl::common::TensorShape &shape,
        const ppl::common::dataformat_t forward_precision,
        const int32_t sp_tuning_level,
        const int32_t winograd_level,
        const ppl::common::isa_t isa_flags,
        const ppl::kernel::arm_server::neon::conv2d_param &param,
        ppl::common::Allocator *allocator);
    // run all possible algo & block size, select the best one and return the final algo manager.
    static ppl::kernel::arm_server::neon::conv2d_offline_manager *gen_fast_algo(
        const ppl::common::TensorShape &shape,
        const ppl::common::dataformat_t forward_precision,
        const int32_t sp_tuning_level,
        const int32_t winograd_level,
        const ppl::common::isa_t isa_flags,
        const ppl::kernel::arm_server::neon::conv2d_param &param,
        ppl::common::Allocator *allocator);
};

}}} // namespace ppl::nn::arm

#endif
