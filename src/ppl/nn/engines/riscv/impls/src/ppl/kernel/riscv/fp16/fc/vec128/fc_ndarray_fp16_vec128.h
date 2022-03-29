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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_FC_VEC128_FC_NDARRAY_FP16_VEC128_H_
#define __ST_PPL_KERNEL_RISCV_FP16_FC_VEC128_FC_NDARRAY_FP16_VEC128_H_

#include "ppl/kernel/riscv/fp16/fc.h"
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

// forward declare;
class fc_ndarray_fp16_vec128_manager;

class fc_ndarray_fp16_vec128_executor final : public fc_executor<__fp16> {
public:
    fc_ndarray_fp16_vec128_executor() {}
    fc_ndarray_fp16_vec128_executor(const fc_common_param* fc_param, const __fp16* cvt_filter, const __fp16* bias)
        : fc_executor<__fp16>(fc_param, cvt_filter, bias) {}
    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

private:
    fc_tunning_param tunning_param_;
    void cal_kernel_tunning_param();
    friend fc_ndarray_fp16_vec128_manager;
};

class fc_ndarray_fp16_vec128_manager final : public fc_manager<__fp16> {
public:
    fc_ndarray_fp16_vec128_manager() {}
    fc_ndarray_fp16_vec128_manager(const fc_common_param& param, ppl::common::Allocator* allocator)
        : fc_manager<__fp16>(param, allocator) {}
    ppl::common::RetCode gen_cvt_weights(const __fp16* filter, const __fp16* bias) override;
    fc_executor<__fp16>* gen_executor() override;

private:
    fc_tunning_param tunning_param_;
};

}}}; // namespace ppl::kernel::riscv

#endif
