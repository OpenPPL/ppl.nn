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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_CONV_TRANSPOSE_CONV_TRANSPOSE_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_CONV_TRANSPOSE_CONV_TRANSPOSE_COMMON_H_

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <memory>

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/fp16/conv2d/common/gemm_common_kernel.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename eT, int32_t c_blk>
void conv_transpose_nxcx_col2im_common(
    const eT *col,
    const int32_t col_h,
    const int32_t col_w,
    const int32_t num_output,
    const int32_t img_h,
    const int32_t img_w,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t hole_h,
    const int32_t hole_w,
    const eT beta,
    eT *image)
{
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t c_img = 0; c_img < num_output; c_img += c_blk) {
        if (beta == 0.0f) {
            memset(image + c_img * img_h * img_w, 0.0f, c_blk * img_h * img_w * sizeof(eT));
        } else {
            for (int64_t hw = 0; hw < img_h * img_w; ++hw) {
                for (int64_t k = 0; k < c_blk; ++k) {
                    image[c_img * img_h * img_w + hw * c_blk + k] *= beta;
                }
            }
        }
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t c_col    = c_img * kernel_h * kernel_w + kh * kernel_w * c_blk + kw * c_blk;
                int64_t w_offset = kw * hole_w;
                int64_t h_offset = kh * hole_h;
                for (int64_t h = 0; h < col_h; ++h) {
                    for (int64_t w = 0; w < col_w; ++w) {
                        int64_t h_pad = h * stride_h - pad_h + h_offset;
                        int64_t w_pad = w * stride_w - pad_w + w_offset;
                        if (h_pad >= 0 && h_pad < img_h && w_pad >= 0 && w_pad < img_w) {
                            const int64_t image_idx = c_img * img_h * img_w + h_pad * img_w * c_blk + w_pad * c_blk;
                            const int64_t col_idx   = c_col * col_h * col_w + h * col_w * c_blk + w * c_blk;
                            for (int64_t k = 0; k < c_blk; ++k) {
                                image[image_idx + k] += col[col_idx + k];
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename eT, int32_t ic_blk, int32_t oc_blk>
void conv_transpose_nxcx_cvt_filter_common(
    const eT *filter,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    eT *cvt_filter)
{
    const int64_t kernel_len     = kernel_h * kernel_w;
    const int64_t pad_num_output = round_up(num_output, oc_blk);
    const int64_t pad_channels   = round_up(channels, ic_blk);
    const int64_t M              = num_output * kernel_len;
    const int64_t K              = channels;
    const int64_t pad_M          = pad_num_output * kernel_len;
    const int64_t pad_K          = pad_channels;

    // ci * (co * k * k) -> pad(co) / oc_blk * k * k * pad(ci) * oc_blk
    memset(cvt_filter, 0.f, pad_M * pad_K);
    for (int64_t i = 0; i < K; i += 1) {
        for (int64_t j = 0; j < num_output; j += 1) {
            for (int64_t k = 0; k < kernel_len; k += 1) {
                int64_t cvt_filter_idx =
                    j / oc_blk * kernel_len * pad_K * oc_blk +
                    k * pad_K * oc_blk +
                    i * oc_blk + j % oc_blk;
                cvt_filter[cvt_filter_idx] = filter[i * M + j * kernel_len + k];
            }
        }
    }
}

template <typename eT, int32_t ic_blk, int32_t oc_blk>
int64_t conv_transpose_nxcx_get_buffer_bytes_common(
    const int32_t batch,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w)
{
    const int64_t pad_channels   = round_up(channels, ic_blk);
    const int64_t pad_num_output = round_up(num_output, oc_blk);

    const int64_t M = num_output * kernel_h * kernel_w;
    const int64_t N = src_h * src_w;
    const int64_t K = channels;

    const int64_t pad_M = pad_num_output * kernel_h * kernel_w;
    const int64_t pad_K = pad_channels;

    const bool do_col2im         = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);
    const int64_t col2im_len     = !do_col2im ? 0 : pad_M * N;
    const int64_t cvt_filter_len = pad_M * pad_K;

    return (col2im_len + cvt_filter_len) * sizeof(eT);
}

template <typename eT, int32_t c_blk>
void conv_transpose_nxcx_add_bias_common(
    eT *dst,
    const eT *bias,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t num_output)
{
    // TODO:
    const int32_t pad_num_output = round_up(num_output, c_blk);
    const int32_t dst_img_len    = dst_h * dst_w;
    int32_t i;

    for (i = 0; i <= pad_num_output - c_blk; i += c_blk) {
        for (int32_t j = 0; j < dst_img_len; j += 1) {
            for (int32_t k = 0; k < c_blk; k += 1) {
                dst[i * dst_img_len + j * c_blk + k] += bias[i + k];
            }
        }
    }
    if (i < pad_num_output) {
        for (int32_t j = 0; j < dst_img_len; j += 1) {
            for (int32_t k = 0; k < pad_num_output - i; k += 1) {
                dst[i * dst_img_len + j * c_blk + k] += bias[i + k];
            }
        }
    }
}

template <typename eT>
using conv_transpose_nxcx_gemm_func_type_t = void (*)(const eT *A, const eT *B, eT *C, const int32_t M, const int32_t N, const int32_t K);

template <typename eT, int32_t c_blk, conv_transpose_nxcx_gemm_func_type_t<eT> gemm_func>
ppl::common::RetCode conv_transpose_nxcx_common(
    const eT *input,
    const eT *filter,
    const eT *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    eT *tmp_buffer,
    eT *output)
{
    constexpr int32_t ic_blk = c_blk;
    constexpr int32_t oc_blk = c_blk;

    const int64_t pad_channels   = round_up(channels, ic_blk);
    const int64_t pad_num_output = round_up(num_output, oc_blk);

    const int64_t M = num_output * kernel_h * kernel_w;
    const int64_t N = src_h * src_w;
    const int64_t K = channels;

    const int64_t pad_M = pad_num_output * kernel_h * kernel_w;
    const int64_t pad_K = pad_channels;

    eT *cvt_filter    = tmp_buffer;
    eT *col2im_buffer = tmp_buffer + pad_M * pad_K;

    const bool do_col2im = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1);

    conv_transpose_nxcx_cvt_filter_common<eT, ic_blk, oc_blk>(
        filter,
        num_output,
        channels,
        kernel_h,
        kernel_w,
        cvt_filter);

    for (int64_t b = 0; b < batch; b += 1) {
        const eT *src_d = input + b * channels * src_h * src_w;
        eT *dst_d       = output + b * num_output * dst_h * dst_w;

        {
            eT *gemm_out;
            if (do_col2im) {
                gemm_out = col2im_buffer;
            } else {
                gemm_out = dst_d;
            }

            gemm_func(
                cvt_filter,
                src_d,
                gemm_out,
                pad_M,
                N,
                pad_K);

            if (do_col2im) {
                conv_transpose_nxcx_col2im_common<eT, c_blk>(
                    col2im_buffer,
                    src_h,
                    src_w,
                    num_output,
                    dst_h,
                    dst_w,
                    kernel_h,
                    kernel_w,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    hole_h,
                    hole_w,
                    0.0f,
                    dst_d);
            }
        }

        if (bias) {
            conv_transpose_nxcx_add_bias_common<eT, c_blk>(
                dst_d,
                bias,
                dst_h,
                dst_w,
                num_output);
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::riscv

#endif // __ST_PPL_KERNEL_RISCV_COMMON_CONV_TRANSPOSE_CONV_TRANSPOSE_COMMON_H_