#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_AVX512_CONV2D_N16CX_DEPTHWISE_BLK1X1_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_AVX512_CONV2D_N16CX_DEPTHWISE_BLK1X1_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/depthwise/avx512/conv2d_n16cx_depthwise_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store>
void conv2d_n16cx_depthwise_fp32_avx512_blk1x1_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
    __m512 zmm0, zmm1;

    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t kernel_flags  = shar_param[FLAGS_IDX()];
    const int64_t kernel_w      = shar_param[KW_IDX()];

    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end   = priv_param[KH_END_IDX()];
    const int64_t kw_start = priv_param[KW_START_IDX()];
    const int64_t kw_end   = priv_param[KW_END_IDX()];

    zmm0 = _mm512_loadu_ps(PICK_PARAM(const float*, priv_param, BIAS_IDX()));

    const float *k_src = PICK_PARAM(const float*, priv_param, SRC_IDX()) + kh_start * src_dh_stride;
    const float *k_flt = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK();
    if (src_dw_stride == CH_DT_BLK()) {
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                zmm1 = _mm512_loadu_ps(k_flt + kw * CH_DT_BLK());
                zmm0 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + kw * CH_DT_BLK()), zmm1, zmm0);
            }
            k_flt += kernel_w * CH_DT_BLK();
            k_src += src_dh_stride;
        }
    } else {
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = kw_start; kw < kw_end; ++kw) {
                zmm1 = _mm512_loadu_ps(k_flt + kw * CH_DT_BLK());
                zmm0 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + kw * src_dw_stride), zmm1, zmm0);
            }
            k_flt += kernel_w * CH_DT_BLK();
            k_src += src_dh_stride;
        }
    }
        
    if (kernel_flags & KERNEL_FLAG_SUM()) {
        zmm0 = _mm512_add_ps(_mm512_loadu_ps(PICK_PARAM(const float*, priv_param, SUM_SRC_IDX())), zmm0);
    }
    if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
        zmm0 = _mm512_max_ps(_mm512_setzero_ps(), zmm0);
    }
    if (kernel_flags & KERNEL_FLAG_RELU6()) {
        zmm0 = _mm512_min_ps(_mm512_set1_ps(6.0f), zmm0);
    }
    if (nt_store) {
        _mm512_stream_ps(PICK_PARAM(float*, priv_param, DST_IDX()), zmm0);
    } else {
        _mm512_storeu_ps(PICK_PARAM(float*, priv_param, DST_IDX()), zmm0);
    }
}

}}};

#endif
