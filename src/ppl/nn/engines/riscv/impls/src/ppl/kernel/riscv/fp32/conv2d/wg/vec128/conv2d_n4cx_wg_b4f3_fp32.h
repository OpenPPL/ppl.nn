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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_CONV2D_WG_VEC128_CONV2D_N4CX_WG_B4F3_FP32_H_
#define __ST_PPL_KERNEL_RISCV_FP32_CONV2D_WG_VEC128_CONV2D_N4CX_WG_B4F3_FP32_H_

#include "ppl/kernel/riscv/fp32/conv2d.h"
#include "ppl/kernel/riscv/fp32/conv2d/wg/vec128/common/wg_pure.h"

namespace ppl { namespace kernel { namespace riscv {

class conv2d_n4cx_wg_b4f3_fp32_offline_manager;

class conv2d_n4cx_wg_b4f3_fp32_runtime_executor final : public conv2d_runtime_executor<float> {
public:
    conv2d_n4cx_wg_b4f3_fp32_runtime_executor() {}
    conv2d_n4cx_wg_b4f3_fp32_runtime_executor(const conv2d_common_param* conv_param, const float* cvt_filter, const float* bias, conv2d_n4cx_wg_bxfxs1_fp32_vec128_tunning_param tunning_param)
        : conv2d_runtime_executor<float>(conv_param, cvt_filter, bias)
        , tunning_param_(tunning_param) {}

    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

private:
    conv2d_n4cx_wg_bxfxs1_fp32_vec128_tunning_param tunning_param_;
    void adjust_tunning_param();

    friend conv2d_n4cx_wg_b4f3_fp32_offline_manager;
};

class conv2d_n4cx_wg_b4f3_fp32_offline_manager final : public conv2d_offline_manager<float> {
public:
    conv2d_n4cx_wg_b4f3_fp32_offline_manager() {}
    conv2d_n4cx_wg_b4f3_fp32_offline_manager(const conv2d_common_param& param, const conv2d_common_algo_info& algo_info, ppl::common::Allocator* allocator)
        : conv2d_offline_manager<float>(param, algo_info, allocator) {}
    bool is_supported() override;
    ppl::common::RetCode gen_cvt_weights(const float* filter, const float* bias) override;
    ppl::common::RetCode fast_init_tunning_param() override;
    ppl::common::RetCode pick_best_tunning_param(const float* src, const float* filter, float* dst, ppl::common::TensorShape& src_shape, ppl::common::TensorShape& dst_shape) override;

    conv2d_base_runtime_executor* gen_executor() override
    {
        return new conv2d_n4cx_wg_b4f3_fp32_runtime_executor(&param_, cvt_filter_, cvt_bias_, tunning_param_);
    }

private:
    conv2d_n4cx_wg_bxfxs1_fp32_vec128_tunning_param tunning_param_;
};

}}}; // namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_FP32_CONV2D_WG_VEC128_CONV2D_N4CX_WG_B4F3_FP32_H_
