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

#include <riscv-vector.h>
#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/common/averagepool2d/averagepool2d_common.h"
#include "ppl/nn/params/onnx/pooling_param.h"

namespace ppl { namespace kernel { namespace riscv {

#define C_BLK() ((int64_t)4)

template <int32_t pooling_mode, int64_t w_len>
static void averagepool2d_n4cx_1x16_kernel_fp32(
    const float* src,
    float* dst,

    const averagepool2d_param* param,
    const int64_t oh,
    const int64_t ow,
    const int64_t ih_start_valid,
    const int64_t ih_end_valid)
{
    const int32_t& kernel_h = param->kernel_h;
    const int32_t& kernel_w = param->kernel_w;
    const int32_t& pad_w    = param->pad_w;
    const int32_t& src_w    = param->src_w;
    const int32_t& dst_w    = param->dst_w;
    const int32_t& stride_w = param->stride_w;

    const int32_t iw_start_valid = -pad_w + ow * stride_w;
    const int32_t iw_end_valid   = iw_start_valid + kernel_w;

    int64_t win_size = 0;
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        win_size = (ih_end_valid - ih_start_valid) * (iw_end_valid - iw_start_valid);
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        win_size = kernel_h * kernel_w;
    }
    float ave_divider = (float)(1.0f / win_size);

    const auto vl = vsetvli(C_BLK(), RVV_E32, RVV_M1);
    float32xm1_t vfave0, vfave1, vfave2, vfave3;
    float32xm1_t vfave4, vfave5, vfave6, vfave7;
    float32xm1_t vfave8, vfave9, vfave10, vfave11;
    float32xm1_t vfave12, vfave13, vfave14, vfave15;

    float32xm1_t vfzero = vfmvvf_float32xm1(0.0f, vl);

    int64_t ih = ih_start_valid;
    int64_t iw = iw_start_valid;
    {
        const float* src_p = src + (ih * src_w + iw) * C_BLK();
        if (w_len >= 1)
            vfave0 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 0 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 2)
            vfave1 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 1 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 3)
            vfave2 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 2 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 4)
            vfave3 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 3 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 5)
            vfave4 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 4 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 6)
            vfave5 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 5 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 7)
            vfave6 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 6 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 8)
            vfave7 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 7 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 9)
            vfave8 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 8 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 10)
            vfave9 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 9 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 11)
            vfave10 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 10 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 12)
            vfave11 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 11 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 13)
            vfave12 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 12 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 14)
            vfave13 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 13 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 15)
            vfave14 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 14 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 16)
            vfave15 = vfaddvv_float32xm1(vfzero, vlev_float32xm1(src_p + 15 * stride_w * C_BLK(), vl), vl);

        iw += 1;
    }
    for (; iw < iw_end_valid; iw++) {
        const float* src_p = src + (ih * src_w + iw) * C_BLK();
        if (w_len >= 1)
            vfave0 = vfaddvv_float32xm1(vfave0, vlev_float32xm1(src_p + 0 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 2)
            vfave1 = vfaddvv_float32xm1(vfave1, vlev_float32xm1(src_p + 1 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 3)
            vfave2 = vfaddvv_float32xm1(vfave2, vlev_float32xm1(src_p + 2 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 4)
            vfave3 = vfaddvv_float32xm1(vfave3, vlev_float32xm1(src_p + 3 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 5)
            vfave4 = vfaddvv_float32xm1(vfave4, vlev_float32xm1(src_p + 4 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 6)
            vfave5 = vfaddvv_float32xm1(vfave5, vlev_float32xm1(src_p + 5 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 7)
            vfave6 = vfaddvv_float32xm1(vfave6, vlev_float32xm1(src_p + 6 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 8)
            vfave7 = vfaddvv_float32xm1(vfave7, vlev_float32xm1(src_p + 7 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 9)
            vfave8 = vfaddvv_float32xm1(vfave8, vlev_float32xm1(src_p + 8 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 10)
            vfave9 = vfaddvv_float32xm1(vfave9, vlev_float32xm1(src_p + 9 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 11)
            vfave10 = vfaddvv_float32xm1(vfave10, vlev_float32xm1(src_p + 10 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 12)
            vfave11 = vfaddvv_float32xm1(vfave11, vlev_float32xm1(src_p + 11 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 13)
            vfave12 = vfaddvv_float32xm1(vfave12, vlev_float32xm1(src_p + 12 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 14)
            vfave13 = vfaddvv_float32xm1(vfave13, vlev_float32xm1(src_p + 13 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 15)
            vfave14 = vfaddvv_float32xm1(vfave14, vlev_float32xm1(src_p + 14 * stride_w * C_BLK(), vl), vl);
        if (w_len >= 16)
            vfave15 = vfaddvv_float32xm1(vfave15, vlev_float32xm1(src_p + 15 * stride_w * C_BLK(), vl), vl);
    }
    ih += 1;
    for (; ih < ih_end_valid; ih++) {
        for (iw = iw_start_valid; iw < iw_end_valid; iw++) {
            const float* src_p = src + (ih * src_w + iw) * C_BLK();
            if (w_len >= 1)
                vfave0 = vfaddvv_float32xm1(vfave0, vlev_float32xm1(src_p + 0 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 2)
                vfave1 = vfaddvv_float32xm1(vfave1, vlev_float32xm1(src_p + 1 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 3)
                vfave2 = vfaddvv_float32xm1(vfave2, vlev_float32xm1(src_p + 2 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 4)
                vfave3 = vfaddvv_float32xm1(vfave3, vlev_float32xm1(src_p + 3 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 5)
                vfave4 = vfaddvv_float32xm1(vfave4, vlev_float32xm1(src_p + 4 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 6)
                vfave5 = vfaddvv_float32xm1(vfave5, vlev_float32xm1(src_p + 5 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 7)
                vfave6 = vfaddvv_float32xm1(vfave6, vlev_float32xm1(src_p + 6 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 8)
                vfave7 = vfaddvv_float32xm1(vfave7, vlev_float32xm1(src_p + 7 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 9)
                vfave8 = vfaddvv_float32xm1(vfave8, vlev_float32xm1(src_p + 8 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 10)
                vfave9 = vfaddvv_float32xm1(vfave9, vlev_float32xm1(src_p + 9 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 11)
                vfave10 = vfaddvv_float32xm1(vfave10, vlev_float32xm1(src_p + 10 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 12)
                vfave11 = vfaddvv_float32xm1(vfave11, vlev_float32xm1(src_p + 11 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 13)
                vfave12 = vfaddvv_float32xm1(vfave12, vlev_float32xm1(src_p + 12 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 14)
                vfave13 = vfaddvv_float32xm1(vfave13, vlev_float32xm1(src_p + 13 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 15)
                vfave14 = vfaddvv_float32xm1(vfave14, vlev_float32xm1(src_p + 14 * stride_w * C_BLK(), vl), vl);
            if (w_len >= 16)
                vfave15 = vfaddvv_float32xm1(vfave15, vlev_float32xm1(src_p + 15 * stride_w * C_BLK(), vl), vl);
        }
    }

    float* dst_p = dst + (oh * dst_w + ow) * C_BLK();
    if (w_len >= 1)
        vsev_float32xm1(dst_p + 0 * C_BLK(), vfmulvf_float32xm1(vfave0, ave_divider, vl), vl);
    if (w_len >= 2)
        vsev_float32xm1(dst_p + 1 * C_BLK(), vfmulvf_float32xm1(vfave1, ave_divider, vl), vl);
    if (w_len >= 3)
        vsev_float32xm1(dst_p + 2 * C_BLK(), vfmulvf_float32xm1(vfave2, ave_divider, vl), vl);
    if (w_len >= 4)
        vsev_float32xm1(dst_p + 3 * C_BLK(), vfmulvf_float32xm1(vfave3, ave_divider, vl), vl);
    if (w_len >= 5)
        vsev_float32xm1(dst_p + 4 * C_BLK(), vfmulvf_float32xm1(vfave4, ave_divider, vl), vl);
    if (w_len >= 6)
        vsev_float32xm1(dst_p + 5 * C_BLK(), vfmulvf_float32xm1(vfave5, ave_divider, vl), vl);
    if (w_len >= 7)
        vsev_float32xm1(dst_p + 6 * C_BLK(), vfmulvf_float32xm1(vfave6, ave_divider, vl), vl);
    if (w_len >= 8)
        vsev_float32xm1(dst_p + 7 * C_BLK(), vfmulvf_float32xm1(vfave7, ave_divider, vl), vl);
    if (w_len >= 9)
        vsev_float32xm1(dst_p + 8 * C_BLK(), vfmulvf_float32xm1(vfave8, ave_divider, vl), vl);
    if (w_len >= 10)
        vsev_float32xm1(dst_p + 9 * C_BLK(), vfmulvf_float32xm1(vfave9, ave_divider, vl), vl);
    if (w_len >= 11)
        vsev_float32xm1(dst_p + 10 * C_BLK(), vfmulvf_float32xm1(vfave10, ave_divider, vl), vl);
    if (w_len >= 12)
        vsev_float32xm1(dst_p + 11 * C_BLK(), vfmulvf_float32xm1(vfave11, ave_divider, vl), vl);
    if (w_len >= 13)
        vsev_float32xm1(dst_p + 12 * C_BLK(), vfmulvf_float32xm1(vfave12, ave_divider, vl), vl);
    if (w_len >= 14)
        vsev_float32xm1(dst_p + 13 * C_BLK(), vfmulvf_float32xm1(vfave13, ave_divider, vl), vl);
    if (w_len >= 15)
        vsev_float32xm1(dst_p + 14 * C_BLK(), vfmulvf_float32xm1(vfave14, ave_divider, vl), vl);
    if (w_len >= 16)
        vsev_float32xm1(dst_p + 15 * C_BLK(), vfmulvf_float32xm1(vfave15, ave_divider, vl), vl);
}

typedef void (*averagepool2d_n4cx_kernel_fp32_func)(const float*, float*, const averagepool2d_param*, const int64_t, const int64_t, const int64_t, const int64_t);
static const averagepool2d_n4cx_kernel_fp32_func averagepool2d_n4cx_1x16_kernel_select[2][16]{
    {averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 1>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 2>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 3>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 4>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 5>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 6>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 7>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 8>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 9>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 10>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 11>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 12>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 13>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 14>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 15>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE, 16>},
    {averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 1>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 2>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 3>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 4>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 5>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 6>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 7>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 8>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 9>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 10>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 11>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 12>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 13>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 14>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 15>,
     averagepool2d_n4cx_1x16_kernel_fp32<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE, 16>}};

template <int32_t pooling_mode>
static inline void averagepool2d_n4cx_border_fp32(
    const float* src,
    float* dst,

    const averagepool2d_param* param,
    const int64_t oh,
    const int64_t ow,
    const int64_t ih_start_valid,
    const int64_t ih_end_valid)
{
    const int32_t& kernel_h = param->kernel_h;
    const int32_t& kernel_w = param->kernel_w;
    const int32_t& stride_h = param->stride_h;
    const int32_t& stride_w = param->stride_w;
    const int32_t& pad_h    = param->pad_h;
    const int32_t& pad_w    = param->pad_w;

    const int32_t& src_h = param->src_h;
    const int32_t& src_w = param->src_w;
    const int32_t& dst_h = param->dst_h;
    const int32_t& dst_w = param->dst_w;

    int64_t iw_start       = -pad_w + ow * stride_w;
    int64_t iw_end         = iw_start + kernel_w;
    int64_t iw_start_valid = max(iw_start, int64_t(0));
    int64_t iw_end_valid   = min(iw_end, int64_t(src_w));

    int64_t win_size = 0;
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        win_size = (ih_end_valid - ih_start_valid) * (iw_end_valid - iw_start_valid);
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        win_size = kernel_h * kernel_w;
    }
    float ave_divider = (float)(1.0f / win_size);

    const auto vl      = vsetvli(C_BLK(), RVV_E32, RVV_M1);
    float32xm1_t vfave = vfmvvf_float32xm1(0.0f, vl);
    for (int64_t ih = ih_start_valid; ih < ih_end_valid; ih++) {
        for (int64_t iw = iw_start_valid; iw < iw_end_valid; iw++) {
            const float* src_p = src + (ih * src_w + iw) * C_BLK();
            vfave              = vfaddvv_float32xm1(vfave, vlev_float32xm1(src_p, vl), vl);
        }
    }

    float* dst_p = dst + (oh * dst_w + ow) * C_BLK();
    vsev_float32xm1(dst_p, vfmulvf_float32xm1(vfave, ave_divider, vl), vl);
}

template <int32_t pooling_mode>
ppl::common::RetCode averagepool2d_n4cx_1x16_fp32_impl(
    const ppl::nn::TensorShape* src_shape,
    const ppl::nn::TensorShape* dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,

    const float* src,
    float* dst)
{
    const int32_t batch    = src_shape->GetDim(0);
    const int32_t channels = src_shape->GetDim(1);
    const int32_t src_h    = src_shape->GetDim(2);
    const int32_t src_w    = src_shape->GetDim(3);
    const int32_t dst_h    = dst_shape->GetDim(2);
    const int32_t dst_w    = dst_shape->GetDim(3);

    const averagepool2d_param param = {kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, batch, channels, src_h, src_w, dst_h, dst_w};

    const int32_t padded_channels = (channels + 8 - 1) / 8 * 8;
    int32_t dst_1x16_start_w      = max((pad_w + stride_w - 1) / stride_w, 0);
    int32_t dst_1x16_end_w        = min((src_w + pad_w - kernel_w) / stride_w + 1, dst_w);

    for (int64_t nc = 0; nc < batch * padded_channels; nc += C_BLK()) {
        const float* src_ = src + nc * src_h * src_w;
        float* dst_       = dst + nc * dst_h * dst_w;
        for (int64_t oh = 0; oh < dst_h; oh++) {
            int64_t ih_start       = -pad_h + oh * stride_h;
            int64_t ih_end         = ih_start + kernel_h;
            int64_t ih_start_valid = max(ih_start, int64_t(0));
            int64_t ih_end_valid   = min(ih_end, int64_t(src_h));

            int64_t ow = 0;
            for (; ow < dst_1x16_start_w; ++ow) {
                averagepool2d_n4cx_border_fp32<pooling_mode>(src_, dst_, &param, oh, ow, ih_start_valid, ih_end_valid);
            }
            for (; ow + 16 <= dst_1x16_end_w; ow += 16) {
                if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE)
                    averagepool2d_n4cx_1x16_kernel_select[1][15](src_, dst_, &param, oh, ow, ih_start_valid, ih_end_valid);
                else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE)
                    averagepool2d_n4cx_1x16_kernel_select[0][15](src_, dst_, &param, oh, ow, ih_start_valid, ih_end_valid);
            }
            if (ow < dst_1x16_end_w) {
                if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE)
                    averagepool2d_n4cx_1x16_kernel_select[1][dst_1x16_end_w - ow - 1](src_, dst_, &param, oh, ow, ih_start_valid, ih_end_valid);
                else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE)
                    averagepool2d_n4cx_1x16_kernel_select[0][dst_1x16_end_w - ow - 1](src_, dst_, &param, oh, ow, ih_start_valid, ih_end_valid);
                ow = dst_1x16_end_w;
            }
            for (; ow < dst_w; ++ow) {
                averagepool2d_n4cx_border_fp32<pooling_mode>(src_, dst_, &param, oh, ow, ih_start_valid, ih_end_valid);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode averagepool2d_n4cx_1x16_fp32(
    const ppl::nn::TensorShape* src_shape,
    const ppl::nn::TensorShape* dst_shape,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t pooling_mode,

    const float* src,
    float* dst)
{
    if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE) {
        return averagepool2d_n4cx_1x16_fp32_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_EXCLUDE>(
            src_shape, dst_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, src, dst);
    } else if (pooling_mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
        return averagepool2d_n4cx_1x16_fp32_impl<ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE>(
            src_shape, dst_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, src, dst);
    }

    return ppl::common::RC_INVALID_VALUE;
}

}}}; // namespace ppl::kernel::riscv
