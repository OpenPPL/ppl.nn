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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_B4F3_AVX512_CONV2D_N16CX_WINOGRAD_B4F3_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_B4F3_AVX512_CONV2D_N16CX_WINOGRAD_B4F3_FP32_AVX512_H_

#include "ppl/kernel/x86/fp32/conv2d.h"
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/timer.h"

namespace ppl { namespace kernel { namespace x86 {

// forward declare;
class conv2d_n16cx_winograd_b4f3_fp32_avx512_manager;

class conv2d_n16cx_winograd_b4f3_fp32_avx512_executor final : public conv2d_fp32_executor {
public:
    conv2d_n16cx_winograd_b4f3_fp32_avx512_executor() {}
    conv2d_n16cx_winograd_b4f3_fp32_avx512_executor(const conv2d_fp32_param *conv_param, const float *cvt_filter, const float *bias)
        : conv2d_fp32_executor(conv_param, cvt_filter, bias) {}
    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

    bool init_profiler() override;
    void clear_profiler() override;
    std::string export_profiler() override;

private:
    struct kernel_schedule_param {
        // Preprocessed param
        int64_t ic_per_gp;
        int64_t oc_per_gp;
        int64_t padded_ic;
        int64_t padded_oc;

        int64_t num_tiles_h;
        int64_t num_tiles_w;
        int64_t num_tiles_b;
        int64_t num_tiles;

        // Multithread mode
        int32_t parallel_mode;
        int32_t use_nt_store;
        int32_t override_only;

        // Blocking
        int64_t ic_l2_blk;
        int64_t oc_l2_blk;
        int64_t tiles_l2_blk;

        // Array length
        int64_t thread_tile_in_len;
        int64_t thread_matmul_in_len;
        int64_t thread_src_trans_len;
        int64_t thread_gemm_out_len;
        int64_t thread_matmul_out_len;
        int64_t thread_postprocess_len;
        int64_t thread_src_dst_trans_len;
        int64_t thread_workspace_len;
        int64_t src_trans_len;
        int64_t gemm_out_len;

    } schedule_param_;

    struct tile_corr {
        int64_t b;
        int64_t th;
        int64_t tw;
    };
    static inline tile_corr cal_tile_corr(const kernel_schedule_param& sp, const int64_t& tid) {
        tile_corr tc;
        tc.b = tid / sp.num_tiles_b;
        const int64_t hw = tid % sp.num_tiles_b;
        tc.th = hw / sp.num_tiles_w;
        tc.tw = hw % sp.num_tiles_w;
        return tc;
    }

#ifdef PPL_X86_KERNEL_TIMING
    thread_timer_t profiler_;
#endif

    void init_preproc_param();

    friend conv2d_n16cx_winograd_b4f3_fp32_avx512_manager;
};

class conv2d_n16cx_winograd_b4f3_fp32_avx512_manager final : public conv2d_fp32_manager {
public:
    conv2d_n16cx_winograd_b4f3_fp32_avx512_manager() {}
    conv2d_n16cx_winograd_b4f3_fp32_avx512_manager(const conv2d_fp32_param &param, ppl::common::Allocator *allocator)
        : conv2d_fp32_manager(param, allocator) {}
    bool is_supported() override;
    ppl::common::RetCode gen_cvt_weights(const float *filter, const float *bias) override;
    conv2d_fp32_executor *gen_executor() override;
};

}}}; // namespace ppl::kernel::x86

#endif
