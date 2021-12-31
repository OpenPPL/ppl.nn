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

#include <cstring>
#include "ppl/common/log.h"
#include "ppl/nn/common/logger.h"
#include "ppl/kernel/riscv/common/math.h"
#include "ppl/kernel/riscv/fp16/conv2d/depthwise/vec128/conv2d_n8cx_dw_fp16.h"
#include "ppl/kernel/riscv/fp16/conv2d/depthwise/vec128/conv2d_n8cx_dw_f3s1_kernel_fp16.cpp"
#include "ppl/kernel/riscv/fp16/conv2d/depthwise/vec128/conv2d_n8cx_dw_f3s2_kernel_fp16.cpp"

namespace ppl { namespace kernel { namespace riscv {

void conv_dw_src_padding_fp16(const __fp16* src, __fp16* src_padded, int64_t src_h, int64_t src_w, int64_t pad_l,
                              int64_t pad_r, int64_t pad_t, int64_t pad_b) {
    int64_t src_w_padded = src_w + pad_l + pad_r;
    int64_t src_padded_h_stride = src_w_padded * 8;

    // top pad
    for (int64_t i = 0; i < pad_t; i++) {
        memset(src_padded, 0.0f, src_w_padded * 8 * sizeof(__fp16));
        src_padded += src_padded_h_stride;
    }
    for (int64_t i = 0; i < src_h; i++) {
        int64_t src_w_idx = 0;
        // left pad
        if (src_w_idx < pad_l) {
            memset(src_padded, 0.0f, pad_l * 8 * sizeof(__fp16));
            src_w_idx += pad_l;
            src_padded += pad_l * 8;
        }
        if (src_w_idx < pad_l + src_w) {
            memcpy(src_padded, src, src_w * 8 * sizeof(__fp16));
            src_w_idx += src_w;
            src_padded += src_w * 8;
            src += src_w * 8;
        }
        // right pad
        if (src_w_idx < src_w_padded) {
            memset(src_padded, 0.0f, pad_r * 8 * sizeof(__fp16));
            src_padded += pad_r * 8;
        }
    }
    // bottom pad
    for (int64_t i = 0; i < pad_b; i++) {
        memset(src_padded, 0.0f, src_w_padded * 8 * sizeof(__fp16));
        src_padded += src_padded_h_stride;
    }
}

size_t conv_dw_get_cvt_flt_size_fp16(int64_t flt_h, int64_t flt_w, int64_t channels) {
    int64_t padded_channels = (channels + 8 - 1) / 8 * 8;
    return size_t(flt_h * flt_w * padded_channels) * sizeof(__fp16);
}

void conv_dw_cvt_flt_fp16(const __fp16* flt, __fp16* cvt_flt, int64_t flt_h, int64_t flt_w, int64_t channels) {
    int64_t i;
    int64_t flt_size = flt_h * flt_w;
    for (i = 0; i + 8 < channels; i += 8) {
        for (int64_t j = 0; j < flt_size; j++) {
            for (int64_t k = 0; k < 8; k++) {
                cvt_flt[j * 8 + k] = flt[k * flt_size + j];
            }
        }
        flt += flt_size * 8;
        cvt_flt += flt_size * 8;
    }
    if (i < channels) {
        for (int64_t j = 0; j < flt_size; j++) {
            int64_t k;
            for (k = 0; k < channels - i; k++) {
                cvt_flt[j * 8 + k] = flt[k * flt_size + j];
            }
            for (; k < 8; k++) {
                cvt_flt[j * 8 + k] = 0.0f;
            }
        }
    }
}

size_t conv_dw_get_temp_buffer_size_fp16(int64_t src_h, int64_t src_w, int64_t pad_h, int64_t pad_w, int64_t flt_h,
                                         int64_t flt_w, int64_t stride_h, int64_t stride_w, int64_t channels) {
    int64_t padded_channels = round_up(channels, 8);
    int64_t src_padded_size = (src_h + pad_h * 2) * (src_w + pad_w * 2) * padded_channels;
    size_t temp_buffer_size = size_t(src_padded_size);

    return temp_buffer_size * sizeof(__fp16);
}

void conv_dw_kernel_riscv_fp16(const __fp16* src, const __fp16* flt, const __fp16* bias, __fp16* dst,

                               int64_t src_pad_w, int64_t dst_h, int64_t dst_w, int64_t flt_h, int64_t flt_w,
                               int64_t stride_h, int64_t stride_w) {
    int64_t dst_w_div8 = dst_w >> 3;
    int64_t dst_w_left = dst_w - (dst_w_div8 << 3);

    for (int64_t i = 0; i < dst_h; i++) {
        for (int64_t j = 0; j < dst_w_div8; j++) {
            int64_t dst_base_idx = 0;
            dst_base_idx += i * dst_w * 8;
            for (int64_t k = 0; k < 8; k++) {
                int64_t dst_idx = dst_base_idx + k;
                dst[dst_idx + (j * 8 + 0) * 8] = bias[k];
                dst[dst_idx + (j * 8 + 1) * 8] = bias[k];
                dst[dst_idx + (j * 8 + 2) * 8] = bias[k];
                dst[dst_idx + (j * 8 + 3) * 8] = bias[k];
                dst[dst_idx + (j * 8 + 4) * 8] = bias[k];
                dst[dst_idx + (j * 8 + 5) * 8] = bias[k];
                dst[dst_idx + (j * 8 + 6) * 8] = bias[k];
                dst[dst_idx + (j * 8 + 7) * 8] = bias[k];
            }
            for (int64_t hk = 0; hk < flt_h; hk++) {
                for (int64_t wk = 0; wk < flt_w; wk++) {
                    for (int64_t k = 0; k < 8; k++) {
                        int64_t flt_idx = 0;
                        flt_idx += k + wk * 8 + hk * flt_w * 8;
                        int64_t dst_idx = dst_base_idx + k;
                        int64_t src_idx0 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 0) * stride_w + wk) * 8 + k;
                        int64_t src_idx1 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 1) * stride_w + wk) * 8 + k;
                        int64_t src_idx2 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 2) * stride_w + wk) * 8 + k;
                        int64_t src_idx3 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 3) * stride_w + wk) * 8 + k;
                        int64_t src_idx4 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 4) * stride_w + wk) * 8 + k;
                        int64_t src_idx5 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 5) * stride_w + wk) * 8 + k;
                        int64_t src_idx6 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 6) * stride_w + wk) * 8 + k;
                        int64_t src_idx7 = (i * stride_h + hk) * src_pad_w * 8 + ((j * 8 + 7) * stride_w + wk) * 8 + k;
                        dst[dst_idx + (j * 8 + 0) * 8] += src[src_idx0] * flt[flt_idx];
                        dst[dst_idx + (j * 8 + 1) * 8] += src[src_idx1] * flt[flt_idx];
                        dst[dst_idx + (j * 8 + 2) * 8] += src[src_idx2] * flt[flt_idx];
                        dst[dst_idx + (j * 8 + 3) * 8] += src[src_idx3] * flt[flt_idx];
                        dst[dst_idx + (j * 8 + 4) * 8] += src[src_idx4] * flt[flt_idx];
                        dst[dst_idx + (j * 8 + 5) * 8] += src[src_idx5] * flt[flt_idx];
                        dst[dst_idx + (j * 8 + 6) * 8] += src[src_idx6] * flt[flt_idx];
                        dst[dst_idx + (j * 8 + 7) * 8] += src[src_idx7] * flt[flt_idx];
                    }
                }
            }
        }
        for (int64_t j = 0; j < dst_w_left; j++) {
            for (int64_t k = 0; k < 8; k++) {
                int64_t dst_idx = 0;
                dst_idx += k + i * dst_w * 8 + (dst_w_div8 * 8 + j) * 8;
                dst[dst_idx] = bias[k];
            }
            for (int64_t hk = 0; hk < flt_h; hk++) {
                for (int64_t wk = 0; wk < flt_w; wk++) {
                    for (int64_t k = 0; k < 8; k++) {
                        int64_t flt_idx = k + wk * 8 + hk * flt_w * 8;
                        int64_t src_idx =
                            (i * stride_h + hk) * src_pad_w * 8 + ((dst_w_div8 * 8 + j) * stride_w + wk) * 8 + k;
                        int64_t dst_idx = k + (dst_w_div8 * 8 + j) * 8 + i * dst_w * 8;
                        dst[dst_idx] += src[src_idx] * flt[flt_idx];
                    }
                }
            }
        }
    }
}

uint64_t conv2d_n8cx_dw_fp16_runtime_executor::cal_temp_buffer_size() {
    size_t temp_buffer_size = conv_dw_get_temp_buffer_size_fp16(
        src_shape_->GetDim(2), src_shape_->GetDim(3), conv_param_->pad_h, conv_param_->pad_w, conv_param_->kernel_h,
        conv_param_->kernel_w, conv_param_->stride_h, conv_param_->stride_w, conv_param_->channels);
    return temp_buffer_size;
}

void conv2d_n8cx_dw_fp16_runtime_executor::adjust_tunning_param() {}

ppl::common::RetCode conv2d_n8cx_dw_fp16_runtime_executor::prepare() {
    if (!conv_param_ || !src_shape_ || !dst_shape_) {
        return ppl::common::RC_INVALID_VALUE;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_dw_fp16_runtime_executor::execute() {
    const conv2d_common_param& cp = *conv_param_;

    if (src_ == nullptr || cvt_bias_ == nullptr || cvt_filter_ == nullptr || temp_buffer_ == nullptr ||
        dst_ == nullptr) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t channels = conv_param_->channels;
    const int64_t kernel_h = conv_param_->kernel_h;
    const int64_t kernel_w = conv_param_->kernel_w;
    const int64_t stride_h = conv_param_->stride_h;
    const int64_t stride_w = conv_param_->stride_w;
    const int64_t pad_h = conv_param_->pad_h;
    const int64_t pad_w = conv_param_->pad_w;

    const int64_t src_h = src_shape_->GetDim(2);
    const int64_t src_w = src_shape_->GetDim(3);
    const int64_t dst_h = dst_shape_->GetDim(2);
    const int64_t dst_w = dst_shape_->GetDim(3);

    int64_t padded_channels = round_up(channels, 8);
    int64_t src_h_padded = src_h + pad_h * 2;
    int64_t src_w_padded = src_w + pad_w * 2;

    typedef void (*depthwise_riscv_kernel_fp16)(const __fp16*, const __fp16*, const __fp16*, __fp16*, int64_t, int64_t,
                                                int64_t);
    depthwise_riscv_kernel_fp16 dw_conv_kernel;
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) {
        switch (dst_w % 4) {
            case 0: {
                dw_conv_kernel = conv_dw_f3s1_h1w4_kernel_riscv_fp16<0>;
                break;
            }
            case 1: {
                dw_conv_kernel = conv_dw_f3s1_h1w4_kernel_riscv_fp16<1>;
                break;
            }
            case 2: {
                dw_conv_kernel = conv_dw_f3s1_h1w4_kernel_riscv_fp16<2>;
                break;
            }
            case 3: {
                dw_conv_kernel = conv_dw_f3s1_h1w4_kernel_riscv_fp16<3>;
                break;
            }
            default:
                break;
        }
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2) {
        switch (dst_h % 3) {
            case 0: {
                switch (dst_w % 4) {
                    case 0: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<0, 0>;
                        break;
                    }
                    case 1: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<0, 1>;
                        break;
                    }
                    case 2: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<0, 2>;
                        break;
                    }
                    case 3: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<0, 3>;
                        break;
                    }
                    default:
                        break;
                }
            }; break;
            case 1: {
                switch (dst_w % 4) {
                    case 0: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<1, 0>;
                        break;
                    }
                    case 1: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<1, 1>;
                        break;
                    }
                    case 2: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<1, 2>;
                        break;
                    }
                    case 3: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<1, 3>;
                        break;
                    }
                    default:
                        break;
                }
            }; break;
            case 2: {
                switch (dst_w % 4) {
                    case 0: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<2, 0>;
                        break;
                    }
                    case 1: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<2, 1>;
                        break;
                    }
                    case 2: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<2, 2>;
                        break;
                    }
                    case 3: {
                        dw_conv_kernel = conv_dw_f3s2_h3w4_kernel_riscv_fp16<2, 3>;
                        break;
                    }
                    default:
                        break;
                }
            }; break;
            default:
                break;
        }
    }

    for (int64_t i = 0; i < padded_channels; i += 8) {
        conv_dw_src_padding_fp16(src_ + i * src_h * src_w, (__fp16*)temp_buffer_ + i * src_h_padded * src_w_padded,
                                 src_h, src_w, pad_h, pad_w, pad_h, pad_w);
        if ((kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1) ||
            (kernel_h == 3 && kernel_w == 3 && stride_h == 2 && stride_w == 2)) {
            dw_conv_kernel((__fp16*)temp_buffer_ + i * src_h_padded * src_w_padded,
                           cvt_filter_ + i * kernel_h * kernel_w, cvt_bias_ + i, dst_ + i * dst_h * dst_w,

                           src_w_padded, dst_h, dst_w);
        } else {
            conv_dw_kernel_riscv_fp16((__fp16*)temp_buffer_ + i * src_h_padded * src_w_padded,
                                      cvt_filter_ + i * kernel_h * kernel_w, cvt_bias_ + i, dst_ + i * dst_h * dst_w,

                                      src_w_padded, dst_h, dst_w, kernel_h, kernel_w, stride_h, stride_w);
        }
    }

    return ppl::common::RC_SUCCESS;
}

bool conv2d_n8cx_dw_fp16_offline_manager::is_supported() {
    return true;
}

ppl::common::RetCode conv2d_n8cx_dw_fp16_offline_manager::fast_init_tunning_param() {
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_dw_fp16_offline_manager::pick_best_tunning_param(const __fp16* src,
                                                                                  const __fp16* filter, __fp16* dst,
                                                                                  ppl::nn::TensorShape& src_shape,
                                                                                  ppl::nn::TensorShape& dst_shape) {
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode conv2d_n8cx_dw_fp16_offline_manager::gen_cvt_weights(const __fp16* filter, const __fp16* bias) {
    if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
        return ppl::common::RC_PERMISSION_DENIED;
    }

    const int64_t num_output = param_.num_output;
    const int64_t channels = param_.channels;
    const int64_t kernel_h = param_.kernel_h;
    const int64_t kernel_w = param_.kernel_w;

    {
        cvt_bias_size_ = round_up(num_output, 8);
        cvt_bias_ = (__fp16*)allocator_->Alloc(cvt_bias_size_ * sizeof(__fp16));
        memcpy(cvt_bias_, bias, num_output * sizeof(__fp16));
        memset(cvt_bias_ + num_output, 0.f, (cvt_bias_size_ - num_output) * sizeof(__fp16));
    }
    {
        cvt_filter_size_ = conv_dw_get_cvt_flt_size_fp16(kernel_h, kernel_w, channels);

        cvt_filter_ = (__fp16*)allocator_->Alloc(cvt_filter_size_);
        conv_dw_cvt_flt_fp16(filter, cvt_filter_, kernel_h, kernel_w, channels);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv