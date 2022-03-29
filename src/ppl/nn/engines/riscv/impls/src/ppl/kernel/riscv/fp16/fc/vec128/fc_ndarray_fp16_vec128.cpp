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

#include <new>
#include <cstring>

#include "ppl/kernel/riscv/common/fc/fc_ndarray_common.h"
#include "ppl/kernel/riscv/fp16/fc/vec128/fc_ndarray_fp16_vec128.h"
#include "ppl/kernel/riscv/fp16/fc/vec128/kernel/fc_ndarray_kernel_fp16_vec128.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)8)


void fc_ndarray_fp16_vec128_executor::cal_kernel_tunning_param() {
    tunning_param_.m_blk = 7;
    tunning_param_.n_blk = 32;
    tunning_param_.k_blk = 128;
}

uint64_t fc_ndarray_fp16_vec128_executor::cal_temp_buffer_size()
{
    LOG(DEBUG) << "FC cal_temp_buffer_size";
    constexpr int64_t atom_oc = 8;
    constexpr int64_t atom_ic = 4;
    constexpr int64_t flt_atom_oc = 32;

    tunning_param_.m_blk = min(tunning_param_.m_blk, src_shape_->GetDim(0));
    tunning_param_.n_blk = min(tunning_param_.n_blk, fc_param_->num_output);
    tunning_param_.k_blk = min(tunning_param_.k_blk, fc_param_->channels);

    return fc_ndarray_common_cal_temp_buffer_size<__fp16, atom_oc, atom_ic, flt_atom_oc>(
        src_shape_->GetDim(0),  // m
        fc_param_->num_output,  // n
        fc_param_->channels,    // k
        tunning_param_
    );
}

ppl::common::RetCode fc_ndarray_fp16_vec128_executor::prepare()
{
    if (!fc_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    cal_kernel_tunning_param();
    LOG(DEBUG) << "FC prepare";

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode fc_ndarray_fp16_vec128_executor::execute()
{
    if (!fc_param_ || !cvt_filter_ || !cvt_bias_ || !src_ || !dst_ || !temp_buffer_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    LOG(DEBUG) << "FC ndarray execute";

    tunning_param_.m_blk = min(tunning_param_.m_blk, src_shape_->GetDim(0));
    tunning_param_.n_blk = min(tunning_param_.n_blk, fc_param_->num_output);
    tunning_param_.k_blk = min(tunning_param_.k_blk, fc_param_->channels);

    constexpr int64_t atom_oc = 8;
    constexpr int64_t atom_ic = 4;
    constexpr int64_t flt_atom_oc = 32;
    fc_ndarray_common_blocking_execute<__fp16, atom_oc, atom_ic, flt_atom_oc>(
        src_,
        cvt_filter_,
        cvt_bias_,
        dst_,
        temp_buffer_,
        src_shape_->GetDim(0),
        fc_param_->channels,
        fc_param_->num_output,
        tunning_param_,
        fc_ndarray_select_gemm_kernel_fp16_vec128<true>,
        fc_ndarray_select_gemm_kernel_fp16_vec128<false>
    );
    return common::RC_SUCCESS;
}

ppl::common::RetCode fc_ndarray_fp16_vec128_manager::gen_cvt_weights(const __fp16* filter, const __fp16* bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int32_t padded_oc = round_up(param_.num_output, C_BLK());
    {
        cvt_bias_size_ = padded_oc;
        cvt_bias_      = (__fp16*)allocator_->Alloc(cvt_bias_size_ * sizeof(__fp16));
        if (cvt_bias_ == nullptr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }
        memcpy(cvt_bias_, bias, param_.num_output * sizeof(__fp16));
        memset(cvt_bias_ + param_.num_output, 0, (padded_oc - param_.num_output) * sizeof(__fp16));
    }

    {
        constexpr int32_t flt_atom_oc = 32;
        constexpr int32_t atom_ic = 4;

        const int32_t padded_ic = round_up(param_.channels, atom_ic);
        const int32_t flt_padded_oc = round_up(param_.num_output, flt_atom_oc);
        cvt_filter_size_        = padded_ic * flt_padded_oc * sizeof(__fp16);
        cvt_filter_             = (__fp16*)allocator_->Alloc(cvt_filter_size_);
        if (cvt_filter_ == nullptr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }
        fc_ndarray_common_cvt_flt_to_nxcx<__fp16, atom_ic, flt_atom_oc>(filter, cvt_filter_, param_.num_output, param_.channels);
    }
    return ppl::common::RC_SUCCESS;
}

fc_executor<__fp16>* fc_ndarray_fp16_vec128_manager::gen_executor()
{
    return new fc_ndarray_fp16_vec128_executor(&param_, cvt_filter_, cvt_bias_);
}

}}}; // namespace ppl::kernel::riscv
