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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_CONV2D_DEPTHWISE_VEC128_CONV2D_N8CX_DW_FP16_H_
#define __ST_PPL_KERNEL_RISCV_FP16_CONV2D_DEPTHWISE_VEC128_CONV2D_N8CX_DW_FP16_H_

#include "ppl/kernel/riscv/fp16/conv2d.h"

namespace ppl { namespace kernel { namespace riscv {

class conv2d_n8cx_dw_fp16_offline_manager;

class conv2d_n8cx_dw_fp16_runtime_executor final : public conv2d_runtime_executor<__fp16> {
public:
    conv2d_n8cx_dw_fp16_runtime_executor() {}
    conv2d_n8cx_dw_fp16_runtime_executor(const conv2d_common_param* conv_param, const __fp16* cvt_filter, const __fp16* bias)
        : conv2d_runtime_executor<__fp16>(conv_param, cvt_filter, bias) {}

    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

private:
    void adjust_tunning_param();

    friend conv2d_n8cx_dw_fp16_offline_manager;
};

class conv2d_n8cx_dw_fp16_offline_manager final : public conv2d_offline_manager<__fp16> {
public:
    conv2d_n8cx_dw_fp16_offline_manager() {}
    conv2d_n8cx_dw_fp16_offline_manager(const conv2d_common_param& param, const conv2d_common_algo_info& algo_info, ppl::common::Allocator* allocator)
        : conv2d_offline_manager<__fp16>(param, algo_info, allocator) {}
    bool is_supported() override;
    ppl::common::RetCode gen_cvt_weights(const __fp16* filter, const __fp16* bias) override;
    ppl::common::RetCode fast_init_tunning_param() override;
    ppl::common::RetCode pick_best_tunning_param(const __fp16* src, const __fp16* filter, __fp16* dst, ppl::common::TensorShape& src_shape, ppl::common::TensorShape& dst_shape) override;

    conv2d_base_runtime_executor* gen_executor() override
    {
        return new conv2d_n8cx_dw_fp16_runtime_executor(&param_, cvt_filter_, cvt_bias_);
    }
};

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP16_CONV2D_DEPTHWISE_VEC128_CONV2D_N8CX_DW_FP16_H_
