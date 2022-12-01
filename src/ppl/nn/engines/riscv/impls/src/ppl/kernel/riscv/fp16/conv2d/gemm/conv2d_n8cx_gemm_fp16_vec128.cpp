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

#include "ppl/common/log.h"
#include "ppl/kernel/riscv/common/math.h"
#include "ppl/kernel/riscv/fp16/conv2d/gemm/conv2d_n8cx_gemm_fp16_vec128.h"
#include "ppl/kernel/riscv/fp16/conv2d/common/conv_shell.h"
#include "ppl/kernel/riscv/fp16/conv2d/common/gemm_common_kernel.h"
#include "ppl/kernel/riscv/fp16/conv2d/common/gemm_common_mem.h"
#include <cstring>

namespace ppl { namespace kernel { namespace riscv {

struct conv2d_n8cx_gemm_tunning_info {
    int64_t gemm_m_blk;
    int64_t gemm_n_blk;
    int64_t gemm_k_blk;
};

size_t conv2d_n8cx_gemm_get_cvt_filter_size_fp16_vec128(
    int64_t num_outs,
    int64_t channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t group)
{
    int64_t channels_per_group = channels / group;
    int64_t num_outs_per_group = num_outs / group;

    int64_t padded_channels_per_group = round_up(channels_per_group, 8);
    int64_t padded_num_outs_per_group = round_up(num_outs_per_group, 8);

    int64_t cvt_filter_size_per_group = padded_num_outs_per_group * padded_channels_per_group * kernel_h * kernel_w;

    return size_t(group * cvt_filter_size_per_group) * sizeof(__fp16);
}

size_t conv2d_n8cx_gemm_get_temp_buffer_size_fp16_vec128(
    int64_t src_h,
    int64_t src_w,
    int64_t channels,
    int64_t num_outs,
    int64_t group,
    int64_t pad_h,
    int64_t pad_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t hole_h,
    int64_t hole_w,
    int64_t stride_h,
    int64_t stride_w,

    int64_t tile_m,
    int64_t tile_n,
    int64_t tile_k)
{
    const int64_t atom_ic             = 8;
    const int64_t atom_oc             = 8;
    int64_t channels_per_group        = channels / group;
    int64_t num_outs_per_group        = num_outs / group;
    int64_t padded_channels_per_group = round_up(channels_per_group, atom_ic);
    int64_t padded_num_outs_per_group = round_up(num_outs_per_group, atom_oc);
    int64_t flt_h_with_hole           = hole_h * (flt_h - 1) + 1;
    int64_t flt_w_with_hole           = hole_w * (flt_w - 1) + 1;
    int64_t dst_h                     = (src_h + pad_h * 2 - flt_h_with_hole) / stride_h + 1;
    int64_t dst_w                     = (src_w + pad_w * 2 - flt_w_with_hole) / stride_w + 1;

    size_t gemm_b_size = size_t(tile_k / atom_oc / atom_ic * tile_n) * sizeof(__fp16);
    size_t gemm_c_size = size_t(tile_m * dst_h * dst_w * atom_oc) * sizeof(__fp16);

    size_t src_cto8c_size = 0;
    if (channels_per_group % atom_ic != 0 && group != 1) {
        src_cto8c_size = size_t(group * padded_channels_per_group * src_h * src_w) * sizeof(__fp16);
    }

    size_t im2col_per_group_size = 0;
    if (flt_h != 1 || flt_w != 1 || pad_h != 0 || pad_w != 0 ||
        stride_h != 1 || stride_w != 1 || hole_h != 1 || hole_w != 1) {
        im2col_per_group_size = size_t(padded_channels_per_group * flt_h * flt_w * dst_h * dst_w) * sizeof(__fp16);
    }

    size_t dst_cto8c_size = 0;
    if (num_outs_per_group % atom_oc != 0 && group != 1) {
        dst_cto8c_size = size_t(group * padded_num_outs_per_group * dst_h * dst_w) * sizeof(__fp16);
    }

    return src_cto8c_size + dst_cto8c_size + im2col_per_group_size + gemm_b_size + gemm_c_size;
}

uint64_t conv2d_n8cx_gemm_fp16_runtime_executor::cal_temp_buffer_size()
{
    size_t temp_buffer_size = conv2d_n8cx_gemm_get_temp_buffer_size_fp16_vec128(
        src_shape_->GetDim(2),
        src_shape_->GetDim(3),
        conv_param_->channels,
        conv_param_->num_output,
        conv_param_->group,
        conv_param_->pad_h,
        conv_param_->pad_w,
        conv_param_->kernel_h,
        conv_param_->kernel_w,
        conv_param_->dilation_h,
        conv_param_->dilation_w,
        conv_param_->stride_h,
        conv_param_->stride_w,

        tunning_param_.m_blk,
        tunning_param_.n_blk,
        tunning_param_.k_blk);

    return temp_buffer_size;
}

void conv2d_n8cx_gemm_fp16_runtime_executor::adjust_tunning_param()
{
    auto dst_h = dst_shape_->GetDim(2);
    auto dst_w = dst_shape_->GetDim(3);

    int64_t N            = dst_h * dst_w * 8;
    tunning_param_.n_blk = min(tunning_param_.n_blk, N);
}

ppl::common::RetCode conv2d_n8cx_gemm_fp16_runtime_executor::prepare()
{
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }
    adjust_tunning_param();

    return ppl::common::RC_SUCCESS;
}

static int64_t conv2d_gemm_get_real_filter_size(const int64_t flt)
{
    return flt;
}

void sgemm_riscv_n8cx_cvt_src(
    const __fp16* src,
    __fp16* dst,

    int64_t K,
    int64_t N,
    int64_t k_blk,
    int64_t n_blk)
{
    int64_t atom_ic    = 8;
    int64_t atom_oc    = 8;
    int64_t real_K     = K / atom_oc / atom_ic;
    int64_t real_k_blk = k_blk / atom_oc / atom_ic;
    for (int64_t k = 0; k < real_k_blk; k++) {
        for (int64_t n = 0; n < n_blk; n++) {
            dst[k * n_blk + n] = src[k * N + n];
        }
    }
}

void sgemm_riscv_n8cx_cvt_dst(
    const __fp16* src,
    const __fp16* bias,
    __fp16* dst,

    int64_t M,
    int64_t N,
    int64_t m_blk,
    int64_t n_blk,
    int64_t m_krnl,
    int64_t n_krnl,
    int64_t m_loop,
    int64_t n_loop)
{
    int64_t atom_oc     = 8;
    int64_t real_n_blk  = n_blk / atom_oc;
    int64_t real_n_krnl = n_krnl / atom_oc;
    for (int64_t ml = 0; ml < m_loop; ml++) {
        for (int64_t nl = 0; nl < n_loop; nl++) {
            for (int64_t m = 0; m < m_krnl; m++) {
                for (int64_t n = 0; n < n_krnl; n++) {
                    int64_t src_idx = 0;
                    src_idx += m * n_krnl + n;
                    src_idx += ml * m_krnl * n_blk + nl * m_krnl * n_krnl;
                    int64_t dst_idx = 0;
                    int64_t ohw_idx = n / atom_oc;
                    int64_t oc8_idx = n % atom_oc;
                    dst_idx += oc8_idx;
                    dst_idx += (ohw_idx + nl * real_n_krnl) * atom_oc;
                    dst_idx += (ml * m_krnl + m) * N;
                    int64_t oc_idx = 0;
                    oc_idx += ml * m_krnl * atom_oc + m * atom_oc + oc8_idx;
                    dst[dst_idx] = src[src_idx] + bias[oc_idx];
                }
            }
        }
    }
}

void im2col_riscv_n8cx_per_group(
    const __fp16* src,
    __fp16* dst,

    int64_t padded_channels,
    int64_t src_h,
    int64_t src_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t hole_h,
    int64_t hole_w,
    int64_t dst_h,
    int64_t dst_w)
{
    int64_t num_8c = padded_channels / 8;
    int64_t outHW  = dst_h * dst_w;
    for (int64_t c = 0; c < num_8c; c++) {
        for (int64_t hk = 0; hk < flt_h; hk++) {
            for (int64_t wk = 0; wk < flt_w; wk++) {
                for (int64_t i = 0; i < outHW; i++) {
                    int64_t out_h   = i / dst_w;
                    int64_t out_w   = i % dst_w;
                    int64_t iw_loc  = out_w * stride_w + wk * hole_w - pad_w;
                    int64_t ih_loc  = out_h * stride_h + hk * hole_h - pad_h;
                    int64_t src_idx = (c * src_h * src_w + ih_loc * src_w + iw_loc) * 8;
                    int64_t dst_idx = c * flt_h * flt_w * outHW * 8 + (hk * flt_w + wk) * outHW * 8 + i * 8;
                    if (ih_loc < 0 || ih_loc >= src_h || iw_loc < 0 || iw_loc >= src_w) {
                        for (int64_t j = 0; j < 8; j++) {
                            dst[dst_idx + j] = 0;
                        }
                    } else {
                        for (int64_t j = 0; j < 8; j++) {
                            dst[dst_idx + j] = src[src_idx + j];
                        }
                    }
                }
            }
        }
    }
}

void sgemm_riscv_n8cx_per_group(
    const __fp16* src,
    const __fp16* filter,
    const __fp16* bias,
    __fp16* gemm_buffer,
    __fp16* dst,

    int64_t M,
    int64_t N,
    int64_t K,
    int64_t m_blk,
    int64_t n_blk,
    int64_t k_blk)
{
    int64_t atom_ic      = 8;
    int64_t atom_oc      = 8;
    int64_t gemm_src_num = (k_blk / atom_ic / atom_oc) * n_blk;
    __fp16* gemm_src_loc = gemm_buffer;
    __fp16* gemm_dst_loc = gemm_src_loc + gemm_src_num;

    int64_t real_blk_m;
    int64_t real_blk_n;
    int64_t real_blk_k;
    for (int64_t m = 0; m < M; m += m_blk) {
        real_blk_m = min(m_blk, M - m);
        bool first = true;
        for (int64_t k = 0; k < K; k += k_blk) {
            real_blk_k              = min(k_blk, K - k);
            int64_t gemm_dst_stride = 0;
            for (int64_t n = 0; n < N; n += n_blk) {
                real_blk_n = min(n_blk, N - n);

                const __fp16* src_  = src + k / atom_ic / atom_oc * N + n;
                int64_t bias_offset = m * atom_oc;
                sgemm_riscv_n8cx_cvt_src(src_, gemm_src_loc, K, N, real_blk_k, real_blk_n);

                if (first) {
                    auto sgemm_n8cx_tile_kernel = conv_gemm_select_xcto8c_kernel_fp16<8, true>(real_blk_m * atom_oc, real_blk_n / atom_oc);
                    sgemm_n8cx_tile_kernel(
                        filter,
                        gemm_src_loc,
                        gemm_dst_loc + gemm_dst_stride,
                        real_blk_m * atom_oc,
                        real_blk_n / atom_oc,
                        real_blk_k / atom_oc);
                } else {
                    auto sgemm_n8cx_tile_kernel = conv_gemm_select_xcto8c_kernel_fp16<8, false>(real_blk_m * atom_oc, real_blk_n / atom_oc);
                    sgemm_n8cx_tile_kernel(
                        filter,
                        gemm_src_loc,
                        gemm_dst_loc + gemm_dst_stride,
                        real_blk_m * atom_oc,
                        real_blk_n / atom_oc,
                        real_blk_k / atom_oc);
                }

                if (k + real_blk_k == K) {
                    __fp16* dst_offset = dst + m * N + n / atom_oc * atom_ic;
                    int64_t m_loop     = real_blk_m / 1;
                    int64_t n_loop     = real_blk_n / atom_oc;
                    sgemm_riscv_n8cx_cvt_dst(
                        gemm_dst_loc + gemm_dst_stride,
                        bias + bias_offset,
                        dst_offset,
                        M,
                        N,
                        real_blk_m,
                        real_blk_n,
                        1,
                        atom_oc,
                        m_loop,
                        n_loop);
                }
                gemm_dst_stride += real_blk_m * real_blk_n;
            }
            first = false;
            filter += real_blk_m * real_blk_k;
        }
    }
}

void conv2d_n8cx_gemm_per_group_fp16_vec128(
    const __fp16* src,
    const __fp16* filter,
    const __fp16* bias,
    __fp16* temp_buffer,
    __fp16* dst,

    int64_t src_h,
    int64_t src_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t flt_h,
    int64_t flt_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t hole_h,
    int64_t hole_w,
    int64_t dst_h,
    int64_t dst_w,
    int64_t ic,
    int64_t oc,

    conv2d_n8cx_gemm_tunning_info tunning_info)
{
    const int64_t atom_ic = 8;
    const int64_t atom_oc = 8;

    int64_t gemm_m_blk = tunning_info.gemm_m_blk;
    int64_t gemm_n_blk = tunning_info.gemm_n_blk;
    int64_t gemm_k_blk = tunning_info.gemm_k_blk;

    int64_t padded_ic = round_up(ic, atom_ic);
    int64_t padded_oc = round_up(oc, atom_oc);

    int64_t im2col_size = 0;
    if (flt_h != 1 || flt_w != 1 || pad_h != 0 || pad_w != 0 ||
        stride_h != 1 || stride_w != 1 || hole_h != 1 || hole_w != 1) {
        im2col_size = padded_ic * flt_h * flt_w * dst_h * dst_w;
    }

    __fp16* im2col_buf  = temp_buffer;
    __fp16* gemm_buffer = im2col_buf + im2col_size;

    int64_t M = padded_oc / atom_oc;
    int64_t K = padded_ic * flt_h * flt_w * atom_oc;
    int64_t N = dst_h * dst_w * atom_oc;
    if (flt_h == 1 && flt_w == 1 && pad_h == 0 && pad_w == 0 &&
        stride_h == 1 && stride_w == 1 && hole_h == 1 && hole_w == 1) {
        sgemm_riscv_n8cx_per_group(src, filter, bias, gemm_buffer, dst, M, N, K, gemm_m_blk, gemm_n_blk, gemm_k_blk);
    } else {
        im2col_riscv_n8cx_per_group(
            src,
            im2col_buf,
            padded_ic,
            src_h,
            src_w,
            flt_h,
            flt_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            hole_h,
            hole_w,
            dst_h,
            dst_w);
        sgemm_riscv_n8cx_per_group(im2col_buf, filter, bias, gemm_buffer, dst, M, N, K, gemm_m_blk, gemm_n_blk, gemm_k_blk);
    }
}

ppl::common::RetCode conv2d_n8cx_gemm_fp16_runtime_executor::execute()
{
    const conv2d_common_param& cp = *conv_param_;

    if (src_ == nullptr || cvt_bias_ == nullptr || cvt_filter_ == nullptr || temp_buffer_ == nullptr ||
        dst_ == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    conv_shell_riscv_fp16<conv2d_n8cx_gemm_tunning_info,
                          8,
                          conv2d_gemm_get_real_filter_size,
                          conv2d_n8cx_gemm_per_group_fp16_vec128>(
        src_,
        cvt_filter_,
        cvt_bias_,
        (__fp16*)temp_buffer_,
        dst_,

        src_shape_->GetDim(2),
        src_shape_->GetDim(3),
        conv_param_->pad_h,
        conv_param_->pad_w,
        conv_param_->kernel_h,
        conv_param_->kernel_w,
        conv_param_->stride_h,
        conv_param_->stride_w,
        conv_param_->dilation_h,
        conv_param_->dilation_w,
        conv_param_->channels,
        conv_param_->num_output,
        conv_param_->group,
        src_shape_->GetDim(0),

        {tunning_param_.m_blk, tunning_param_.n_blk, tunning_param_.k_blk});

    return ppl::common::RC_SUCCESS;
}

bool conv2d_n8cx_gemm_fp16_offline_manager::is_supported()
{
    return true;
}

ppl::common::RetCode conv2d_n8cx_gemm_fp16_offline_manager::fast_init_tunning_param()
{
    const int64_t channels_per_group = param_.channels / param_.group;
    const int64_t num_outs_per_group = param_.num_output / param_.group;
    int64_t M                        = round_up(num_outs_per_group, 8) / 8;
    int64_t K                        = round_up(channels_per_group, 8) * param_.kernel_h * param_.kernel_w * 8;

    tunning_param_.m_blk = min(int64_t(16), M);
    tunning_param_.k_blk = min(int64_t(512), K);
    tunning_param_.n_blk = 1152;

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_gemm_fp16_offline_manager::pick_best_tunning_param(
    const __fp16* src,
    const __fp16* filter,
    __fp16* dst,
    ppl::common::TensorShape& src_shape,
    ppl::common::TensorShape& dst_shape)
{
    return ppl::common::RC_SUCCESS;
}

void sgemm_riscv_n8cx_cvt_filter_per_group(
    const __fp16* flt,
    __fp16* flt_cvt,

    int64_t num_outs,
    int64_t channels,
    int64_t flt_h,
    int64_t flt_w,
    int64_t M,
    int64_t K,
    int64_t M_pad,
    int64_t K_pad,
    int64_t m_blk,
    int64_t k_blk,
    int64_t m_krnl,
    int64_t k_krnl)
{
    for (int64_t mt = 0; mt < M_pad; mt += m_blk) {
        int64_t real_blk_m = min(m_blk, M_pad - mt);
        for (int64_t kt = 0; kt < K_pad; kt += k_blk) {
            int64_t real_blk_k = min(k_blk, K_pad - kt);
            for (int64_t mk = 0; mk < real_blk_m; mk += m_krnl) {
                for (int64_t kk = 0; kk < real_blk_k; kk += k_krnl) {
                    for (int64_t i = 0; i < m_krnl; i++) {
                        for (int64_t j = 0; j < k_krnl; j++) {
                            int64_t acc_left_krnl = k_krnl / 8 / 8;
                            int64_t acc_left      = j / (8 * 8);
                            int64_t ic8oc8_idx    = j % (8 * 8);
                            int64_t ic8_idx       = ic8oc8_idx / 8;
                            int64_t oc8_idx       = ic8oc8_idx % 8;

                            int64_t flt_cvt_idx = 0;
                            flt_cvt_idx += mt * K_pad + kt * real_blk_m;
                            flt_cvt_idx += mk * real_blk_k + kk * m_krnl;
                            flt_cvt_idx += i * k_krnl + j;

                            int64_t oc_idx  = (mt + mk + i) * 8 + oc8_idx;
                            int64_t flt_idx = 0;
                            flt_idx += oc_idx * (K / 8);
                            int64_t acc_left_idx = acc_left_krnl * ((kk / 8 / 8) + (kt / 8 / 8));
                            int64_t ic_left_idx  = acc_left_idx / (flt_h * flt_w);
                            int64_t hwk_idx      = acc_left_idx % (flt_h * flt_w);
                            int64_t acc_idx      = (ic_left_idx * 8 + ic8_idx) * flt_h * flt_w + hwk_idx;
                            flt_idx += acc_idx;
                            if (oc_idx >= num_outs || acc_idx >= (channels * flt_h * flt_w)) {
                                flt_cvt[flt_cvt_idx] = 0.0f;
                            } else {
                                flt_cvt[flt_cvt_idx] = flt[flt_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv2d_n8cx_gemm_cvt_filter_fp16_vec128(
    const __fp16* filter,
    __fp16* filter_cvt,

    int64_t flt_h,
    int64_t flt_w,
    int64_t channels,
    int64_t num_outs,
    int64_t group,
    int64_t m_blk,
    int64_t k_blk)
{
    int64_t channels_per_group        = channels / group;
    int64_t num_outs_per_group        = num_outs / group;
    int64_t padded_channels_per_group = round_up(channels_per_group, 8);
    int64_t padded_num_outs_per_group = round_up(num_outs_per_group, 8);

    int64_t flt_size_per_group     = num_outs_per_group * channels_per_group * flt_h * flt_w;
    int64_t flt_cvt_size_per_group = padded_num_outs_per_group * padded_channels_per_group * flt_h * flt_w;

    for (int64_t g = 0; g < group; g++) {
        sgemm_riscv_n8cx_cvt_filter_per_group(
            filter + g * flt_size_per_group,
            filter_cvt + g * flt_cvt_size_per_group,
            num_outs_per_group,
            channels_per_group,
            flt_h,
            flt_w,
            num_outs_per_group / 8,
            channels_per_group * flt_h * flt_w * 8,
            padded_num_outs_per_group / 8,
            padded_channels_per_group * flt_h * flt_w * 8,
            m_blk,
            k_blk,
            1,
            8 * 8);
    }
}

ppl::common::RetCode conv2d_n8cx_gemm_fp16_offline_manager::gen_cvt_weights(
    const __fp16* filter,
    const __fp16* bias)
{
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_output = param_.num_output;
    const int64_t channels   = param_.channels;
    const int64_t kernel_h   = param_.kernel_h;
    const int64_t kernel_w   = param_.kernel_w;
    const int64_t num_group  = param_.group;

    // cvt bias
    {
        cvt_bias_size_ = round_up(num_output, 8);
        cvt_bias_      = (__fp16*)allocator_->Alloc(cvt_bias_size_ * sizeof(__fp16));
        memcpy(cvt_bias_, bias, num_output * sizeof(__fp16));
        memset(cvt_bias_ + num_output, 0.0f, (cvt_bias_size_ - num_output) * sizeof(__fp16));
    }
    // cvt filter
    {
        cvt_filter_size_ = conv2d_n8cx_gemm_get_cvt_filter_size_fp16_vec128(
            num_output,
            channels,
            kernel_h,
            kernel_w,
            num_group);
        cvt_filter_ = (__fp16*)allocator_->Alloc(cvt_filter_size_);
        conv2d_n8cx_gemm_cvt_filter_fp16_vec128(
            filter,
            cvt_filter_,
            kernel_h,
            kernel_w,
            channels,
            num_output,
            num_group,
            tunning_param_.m_blk,
            tunning_param_.k_blk);
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::riscv
