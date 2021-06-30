#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_AVX512_CONV2D_N16CX_DEPTHWISE_BLK1X31_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_AVX512_CONV2D_N16CX_DEPTHWISE_BLK1X31_KERNEL_FP32_AVX512_H_

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/depthwise/avx512/conv2d_n16cx_depthwise_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

template <bool nt_store, int32_t spec_stride_w, int32_t w_len>
void conv2d_n16cx_depthwise_fp32_avx512_blk1x31_kernel(
    const int64_t *priv_param,
    const int64_t *shar_param)
{
#define KW_COMPUTE_STEP() do {\
    zmm31 = _mm512_loadu_ps(k_flt);\
    if (w_len > 0) zmm0 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 0 * src_sw_stride), zmm31, zmm0);\
    if (w_len > 1) zmm1 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 1 * src_sw_stride), zmm31, zmm1);\
    if (w_len > 2) zmm2 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 2 * src_sw_stride), zmm31, zmm2);\
    if (w_len > 3) zmm3 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 3 * src_sw_stride), zmm31, zmm3);\
    if (w_len > 4) zmm4 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 4 * src_sw_stride), zmm31, zmm4);\
    if (w_len > 5) zmm5 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 5 * src_sw_stride), zmm31, zmm5);\
    if (w_len > 6) zmm6 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 6 * src_sw_stride), zmm31, zmm6);\
    if (w_len > 7) zmm7 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 7 * src_sw_stride), zmm31, zmm7);\
    if (w_len > 8) zmm8 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 8 * src_sw_stride), zmm31, zmm8);\
    if (w_len > 9) zmm9 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 9 * src_sw_stride), zmm31, zmm9);\
    if (w_len > 10) zmm10 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 10 * src_sw_stride), zmm31, zmm10);\
    if (w_len > 11) zmm11 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 11 * src_sw_stride), zmm31, zmm11);\
    if (w_len > 12) zmm12 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 12 * src_sw_stride), zmm31, zmm12);\
    if (w_len > 13) zmm13 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 13 * src_sw_stride), zmm31, zmm13);\
    if (w_len > 14) zmm14 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 14 * src_sw_stride), zmm31, zmm14);\
    if (w_len > 15) zmm15 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 15 * src_sw_stride), zmm31, zmm15);\
    if (w_len > 16) zmm16 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 16 * src_sw_stride), zmm31, zmm16);\
    if (w_len > 17) zmm17 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 17 * src_sw_stride), zmm31, zmm17);\
    if (w_len > 18) zmm18 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 18 * src_sw_stride), zmm31, zmm18);\
    if (w_len > 19) zmm19 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 19 * src_sw_stride), zmm31, zmm19);\
    if (w_len > 20) zmm20 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 20 * src_sw_stride), zmm31, zmm20);\
    if (w_len > 21) zmm21 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 21 * src_sw_stride), zmm31, zmm21);\
    if (w_len > 22) zmm22 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 22 * src_sw_stride), zmm31, zmm22);\
    if (w_len > 23) zmm23 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 23 * src_sw_stride), zmm31, zmm23);\
    if (w_len > 24) zmm24 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 24 * src_sw_stride), zmm31, zmm24);\
    if (w_len > 25) zmm25 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 25 * src_sw_stride), zmm31, zmm25);\
    if (w_len > 26) zmm26 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 26 * src_sw_stride), zmm31, zmm26);\
    if (w_len > 27) zmm27 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 27 * src_sw_stride), zmm31, zmm27);\
    if (w_len > 28) zmm28 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 28 * src_sw_stride), zmm31, zmm28);\
    if (w_len > 29) zmm29 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 29 * src_sw_stride), zmm31, zmm29);\
    if (w_len > 30) zmm30 = _mm512_fmadd_ps(_mm512_loadu_ps(k_src + 30 * src_sw_stride), zmm31, zmm30);\
    k_flt += CH_DT_BLK();\
    k_src += src_dw_stride;\
} while (0)

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    const int64_t src_sw_stride = spec_stride_w ? spec_stride_w * CH_DT_BLK() : shar_param[SRC_SW_STRIDE_IDX()];
    const int64_t src_dh_stride = shar_param[SRC_DH_STRIDE_IDX()];
    const int64_t src_dw_stride = shar_param[SRC_DW_STRIDE_IDX()];
    const int64_t kernel_flags  = shar_param[FLAGS_IDX()];
    const int64_t kernel_w      = shar_param[KW_IDX()];
    const int64_t src_kh_stride = src_dh_stride - kernel_w * src_dw_stride;

    const int64_t kh_start = priv_param[KH_START_IDX()];
    const int64_t kh_end   = priv_param[KH_END_IDX()];

    const float *src = PICK_PARAM(const float*, priv_param, SRC_IDX());
    const float *sum = PICK_PARAM(const float*, priv_param, SUM_SRC_IDX());
    float *dst       = PICK_PARAM(float*, priv_param, DST_IDX());
    int64_t ow       = priv_param[OW_IDX()];
    do {
        const float* bias = PICK_PARAM(const float*, priv_param, BIAS_IDX());
        if (w_len > 0) zmm0 = _mm512_loadu_ps(bias + 0 * CH_DT_BLK());
        if (w_len > 1) zmm1 = zmm0;
        if (w_len > 2) zmm2 = zmm0;
        if (w_len > 3) zmm3 = zmm0;
        if (w_len > 4) zmm4 = zmm0;
        if (w_len > 5) zmm5 = zmm0;
        if (w_len > 6) zmm6 = zmm0;
        if (w_len > 7) zmm7 = zmm0;
        if (w_len > 8) zmm8 = zmm0;
        if (w_len > 9) zmm9 = zmm0;
        if (w_len > 10) zmm10 = zmm0;
        if (w_len > 11) zmm11 = zmm0;
        if (w_len > 12) zmm12 = zmm0;
        if (w_len > 13) zmm13 = zmm0;
        if (w_len > 14) zmm14 = zmm0;
        if (w_len > 15) zmm15 = zmm0;
        if (w_len > 16) zmm16 = zmm0;
        if (w_len > 17) zmm17 = zmm0;
        if (w_len > 18) zmm18 = zmm0;
        if (w_len > 19) zmm19 = zmm0;
        if (w_len > 20) zmm20 = zmm0;
        if (w_len > 21) zmm21 = zmm0;
        if (w_len > 22) zmm22 = zmm0;
        if (w_len > 23) zmm23 = zmm0;
        if (w_len > 24) zmm24 = zmm0;
        if (w_len > 25) zmm25 = zmm0;
        if (w_len > 26) zmm26 = zmm0;
        if (w_len > 27) zmm27 = zmm0;
        if (w_len > 28) zmm28 = zmm0;
        if (w_len > 29) zmm29 = zmm0;
        if (w_len > 30) zmm30 = zmm0;

        const float *k_src = src + kh_start * src_dh_stride;
        const float *k_flt  = PICK_PARAM(const float*, priv_param, FLT_IDX()) + kh_start * kernel_w * CH_DT_BLK();
        for (int32_t kh = kh_start; kh < kh_end; ++kh) {
            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                KW_COMPUTE_STEP();
            }
            k_src += src_kh_stride;
        }
        
        if (kernel_flags & KERNEL_FLAG_SUM()) {
            if (w_len > 0) zmm0 = _mm512_add_ps(_mm512_loadu_ps(sum + 0 * CH_DT_BLK()), zmm0);
            if (w_len > 1) zmm1 = _mm512_add_ps(_mm512_loadu_ps(sum + 1 * CH_DT_BLK()), zmm1);
            if (w_len > 2) zmm2 = _mm512_add_ps(_mm512_loadu_ps(sum + 2 * CH_DT_BLK()), zmm2);
            if (w_len > 3) zmm3 = _mm512_add_ps(_mm512_loadu_ps(sum + 3 * CH_DT_BLK()), zmm3);
            if (w_len > 4) zmm4 = _mm512_add_ps(_mm512_loadu_ps(sum + 4 * CH_DT_BLK()), zmm4);
            if (w_len > 5) zmm5 = _mm512_add_ps(_mm512_loadu_ps(sum + 5 * CH_DT_BLK()), zmm5);
            if (w_len > 6) zmm6 = _mm512_add_ps(_mm512_loadu_ps(sum + 6 * CH_DT_BLK()), zmm6);
            if (w_len > 7) zmm7 = _mm512_add_ps(_mm512_loadu_ps(sum + 7 * CH_DT_BLK()), zmm7);
            if (w_len > 8) zmm8 = _mm512_add_ps(_mm512_loadu_ps(sum + 8 * CH_DT_BLK()), zmm8);
            if (w_len > 9) zmm9 = _mm512_add_ps(_mm512_loadu_ps(sum + 9 * CH_DT_BLK()), zmm9);
            if (w_len > 10) zmm10 = _mm512_add_ps(_mm512_loadu_ps(sum + 10 * CH_DT_BLK()), zmm10);
            if (w_len > 11) zmm11 = _mm512_add_ps(_mm512_loadu_ps(sum + 11 * CH_DT_BLK()), zmm11);
            if (w_len > 12) zmm12 = _mm512_add_ps(_mm512_loadu_ps(sum + 12 * CH_DT_BLK()), zmm12);
            if (w_len > 13) zmm13 = _mm512_add_ps(_mm512_loadu_ps(sum + 13 * CH_DT_BLK()), zmm13);
            if (w_len > 14) zmm14 = _mm512_add_ps(_mm512_loadu_ps(sum + 14 * CH_DT_BLK()), zmm14);
            if (w_len > 15) zmm15 = _mm512_add_ps(_mm512_loadu_ps(sum + 15 * CH_DT_BLK()), zmm15);
            if (w_len > 16) zmm16 = _mm512_add_ps(_mm512_loadu_ps(sum + 16 * CH_DT_BLK()), zmm16);
            if (w_len > 17) zmm17 = _mm512_add_ps(_mm512_loadu_ps(sum + 17 * CH_DT_BLK()), zmm17);
            if (w_len > 18) zmm18 = _mm512_add_ps(_mm512_loadu_ps(sum + 18 * CH_DT_BLK()), zmm18);
            if (w_len > 19) zmm19 = _mm512_add_ps(_mm512_loadu_ps(sum + 19 * CH_DT_BLK()), zmm19);
            if (w_len > 20) zmm20 = _mm512_add_ps(_mm512_loadu_ps(sum + 20 * CH_DT_BLK()), zmm20);
            if (w_len > 21) zmm21 = _mm512_add_ps(_mm512_loadu_ps(sum + 21 * CH_DT_BLK()), zmm21);
            if (w_len > 22) zmm22 = _mm512_add_ps(_mm512_loadu_ps(sum + 22 * CH_DT_BLK()), zmm22);
            if (w_len > 23) zmm23 = _mm512_add_ps(_mm512_loadu_ps(sum + 23 * CH_DT_BLK()), zmm23);
            if (w_len > 24) zmm24 = _mm512_add_ps(_mm512_loadu_ps(sum + 24 * CH_DT_BLK()), zmm24);
            if (w_len > 25) zmm25 = _mm512_add_ps(_mm512_loadu_ps(sum + 25 * CH_DT_BLK()), zmm25);
            if (w_len > 26) zmm26 = _mm512_add_ps(_mm512_loadu_ps(sum + 26 * CH_DT_BLK()), zmm26);
            if (w_len > 27) zmm27 = _mm512_add_ps(_mm512_loadu_ps(sum + 27 * CH_DT_BLK()), zmm27);
            if (w_len > 28) zmm28 = _mm512_add_ps(_mm512_loadu_ps(sum + 28 * CH_DT_BLK()), zmm28);
            if (w_len > 29) zmm29 = _mm512_add_ps(_mm512_loadu_ps(sum + 29 * CH_DT_BLK()), zmm29);
            if (w_len > 20) zmm30 = _mm512_add_ps(_mm512_loadu_ps(sum + 30 * CH_DT_BLK()), zmm30);
        }
        if (kernel_flags & (KERNEL_FLAG_RELU() | KERNEL_FLAG_RELU6())) {
            zmm31 = _mm512_setzero_ps();
            if (w_len > 0) zmm0 = _mm512_max_ps(zmm0, zmm31);
            if (w_len > 1) zmm1 = _mm512_max_ps(zmm1, zmm31);
            if (w_len > 2) zmm2 = _mm512_max_ps(zmm2, zmm31);
            if (w_len > 3) zmm3 = _mm512_max_ps(zmm3, zmm31);
            if (w_len > 4) zmm4 = _mm512_max_ps(zmm4, zmm31);
            if (w_len > 5) zmm5 = _mm512_max_ps(zmm5, zmm31);
            if (w_len > 6) zmm6 = _mm512_max_ps(zmm6, zmm31);
            if (w_len > 7) zmm7 = _mm512_max_ps(zmm7, zmm31);
            if (w_len > 8) zmm8 = _mm512_max_ps(zmm8, zmm31);
            if (w_len > 9) zmm9 = _mm512_max_ps(zmm9, zmm31);
            if (w_len > 10) zmm10 = _mm512_max_ps(zmm10, zmm31);
            if (w_len > 11) zmm11 = _mm512_max_ps(zmm11, zmm31);
            if (w_len > 12) zmm12 = _mm512_max_ps(zmm12, zmm31);
            if (w_len > 13) zmm13 = _mm512_max_ps(zmm13, zmm31);
            if (w_len > 14) zmm14 = _mm512_max_ps(zmm14, zmm31);
            if (w_len > 15) zmm15 = _mm512_max_ps(zmm15, zmm31);
            if (w_len > 16) zmm16 = _mm512_max_ps(zmm16, zmm31);
            if (w_len > 17) zmm17 = _mm512_max_ps(zmm17, zmm31);
            if (w_len > 18) zmm18 = _mm512_max_ps(zmm18, zmm31);
            if (w_len > 19) zmm19 = _mm512_max_ps(zmm19, zmm31);
            if (w_len > 20) zmm20 = _mm512_max_ps(zmm20, zmm31);
            if (w_len > 21) zmm21 = _mm512_max_ps(zmm21, zmm31);
            if (w_len > 22) zmm22 = _mm512_max_ps(zmm22, zmm31);
            if (w_len > 23) zmm23 = _mm512_max_ps(zmm23, zmm31);
            if (w_len > 24) zmm24 = _mm512_max_ps(zmm24, zmm31);
            if (w_len > 25) zmm25 = _mm512_max_ps(zmm25, zmm31);
            if (w_len > 26) zmm26 = _mm512_max_ps(zmm26, zmm31);
            if (w_len > 27) zmm27 = _mm512_max_ps(zmm27, zmm31);
            if (w_len > 28) zmm28 = _mm512_max_ps(zmm28, zmm31);
            if (w_len > 29) zmm29 = _mm512_max_ps(zmm29, zmm31);
            if (w_len > 30) zmm30 = _mm512_max_ps(zmm30, zmm31);
        }
        if (kernel_flags & KERNEL_FLAG_RELU6()) {
            zmm31 = _mm512_set1_ps(6.0f);
            if (w_len > 0) zmm0 = _mm512_min_ps(zmm0, zmm31);
            if (w_len > 1) zmm1 = _mm512_min_ps(zmm1, zmm31);
            if (w_len > 2) zmm2 = _mm512_min_ps(zmm2, zmm31);
            if (w_len > 3) zmm3 = _mm512_min_ps(zmm3, zmm31);
            if (w_len > 4) zmm4 = _mm512_min_ps(zmm4, zmm31);
            if (w_len > 5) zmm5 = _mm512_min_ps(zmm5, zmm31);
            if (w_len > 6) zmm6 = _mm512_min_ps(zmm6, zmm31);
            if (w_len > 7) zmm7 = _mm512_min_ps(zmm7, zmm31);
            if (w_len > 8) zmm8 = _mm512_min_ps(zmm8, zmm31);
            if (w_len > 9) zmm9 = _mm512_min_ps(zmm9, zmm31);
            if (w_len > 10) zmm10 = _mm512_min_ps(zmm10, zmm31);
            if (w_len > 11) zmm11 = _mm512_min_ps(zmm11, zmm31);
            if (w_len > 12) zmm12 = _mm512_min_ps(zmm12, zmm31);
            if (w_len > 13) zmm13 = _mm512_min_ps(zmm13, zmm31);
            if (w_len > 14) zmm14 = _mm512_min_ps(zmm14, zmm31);
            if (w_len > 15) zmm15 = _mm512_min_ps(zmm15, zmm31);
            if (w_len > 16) zmm16 = _mm512_min_ps(zmm16, zmm31);
            if (w_len > 17) zmm17 = _mm512_min_ps(zmm17, zmm31);
            if (w_len > 18) zmm18 = _mm512_min_ps(zmm18, zmm31);
            if (w_len > 19) zmm19 = _mm512_min_ps(zmm19, zmm31);
            if (w_len > 20) zmm20 = _mm512_min_ps(zmm20, zmm31);
            if (w_len > 21) zmm21 = _mm512_min_ps(zmm21, zmm31);
            if (w_len > 22) zmm22 = _mm512_min_ps(zmm22, zmm31);
            if (w_len > 23) zmm23 = _mm512_min_ps(zmm23, zmm31);
            if (w_len > 24) zmm24 = _mm512_min_ps(zmm24, zmm31);
            if (w_len > 25) zmm25 = _mm512_min_ps(zmm25, zmm31);
            if (w_len > 26) zmm26 = _mm512_min_ps(zmm26, zmm31);
            if (w_len > 27) zmm27 = _mm512_min_ps(zmm27, zmm31);
            if (w_len > 28) zmm28 = _mm512_min_ps(zmm28, zmm31);
            if (w_len > 29) zmm29 = _mm512_min_ps(zmm29, zmm31);
            if (w_len > 30) zmm30 = _mm512_min_ps(zmm30, zmm31);
        }
        if (nt_store) {
            if (w_len > 0) _mm512_stream_ps(dst + 0 * CH_DT_BLK(), zmm0);
            if (w_len > 1) _mm512_stream_ps(dst + 1 * CH_DT_BLK(), zmm1);
            if (w_len > 2) _mm512_stream_ps(dst + 2 * CH_DT_BLK(), zmm2);
            if (w_len > 3) _mm512_stream_ps(dst + 3 * CH_DT_BLK(), zmm3);
            if (w_len > 4) _mm512_stream_ps(dst + 4 * CH_DT_BLK(), zmm4);
            if (w_len > 5) _mm512_stream_ps(dst + 5 * CH_DT_BLK(), zmm5);
            if (w_len > 6) _mm512_stream_ps(dst + 6 * CH_DT_BLK(), zmm6);
            if (w_len > 7) _mm512_stream_ps(dst + 7 * CH_DT_BLK(), zmm7);
            if (w_len > 8) _mm512_stream_ps(dst + 8 * CH_DT_BLK(), zmm8);
            if (w_len > 9) _mm512_stream_ps(dst + 9 * CH_DT_BLK(), zmm9);
            if (w_len > 10) _mm512_stream_ps(dst + 10 * CH_DT_BLK(), zmm10);
            if (w_len > 11) _mm512_stream_ps(dst + 11 * CH_DT_BLK(), zmm11);
            if (w_len > 12) _mm512_stream_ps(dst + 12 * CH_DT_BLK(), zmm12);
            if (w_len > 13) _mm512_stream_ps(dst + 13 * CH_DT_BLK(), zmm13);
            if (w_len > 14) _mm512_stream_ps(dst + 14 * CH_DT_BLK(), zmm14);
            if (w_len > 15) _mm512_stream_ps(dst + 15 * CH_DT_BLK(), zmm15);
            if (w_len > 16) _mm512_stream_ps(dst + 16 * CH_DT_BLK(), zmm16);
            if (w_len > 17) _mm512_stream_ps(dst + 17 * CH_DT_BLK(), zmm17);
            if (w_len > 18) _mm512_stream_ps(dst + 18 * CH_DT_BLK(), zmm18);
            if (w_len > 19) _mm512_stream_ps(dst + 19 * CH_DT_BLK(), zmm19);
            if (w_len > 20) _mm512_stream_ps(dst + 20 * CH_DT_BLK(), zmm20);
            if (w_len > 21) _mm512_stream_ps(dst + 21 * CH_DT_BLK(), zmm21);
            if (w_len > 22) _mm512_stream_ps(dst + 22 * CH_DT_BLK(), zmm22);
            if (w_len > 23) _mm512_stream_ps(dst + 23 * CH_DT_BLK(), zmm23);
            if (w_len > 24) _mm512_stream_ps(dst + 24 * CH_DT_BLK(), zmm24);
            if (w_len > 25) _mm512_stream_ps(dst + 25 * CH_DT_BLK(), zmm25);
            if (w_len > 26) _mm512_stream_ps(dst + 26 * CH_DT_BLK(), zmm26);
            if (w_len > 27) _mm512_stream_ps(dst + 27 * CH_DT_BLK(), zmm27);
            if (w_len > 28) _mm512_stream_ps(dst + 28 * CH_DT_BLK(), zmm28);
            if (w_len > 29) _mm512_stream_ps(dst + 29 * CH_DT_BLK(), zmm29);
            if (w_len > 30) _mm512_stream_ps(dst + 30 * CH_DT_BLK(), zmm30);
        } else {
            if (w_len > 0) _mm512_storeu_ps(dst + 0 * CH_DT_BLK(), zmm0);
            if (w_len > 1) _mm512_storeu_ps(dst + 1 * CH_DT_BLK(), zmm1);
            if (w_len > 2) _mm512_storeu_ps(dst + 2 * CH_DT_BLK(), zmm2);
            if (w_len > 3) _mm512_storeu_ps(dst + 3 * CH_DT_BLK(), zmm3);
            if (w_len > 4) _mm512_storeu_ps(dst + 4 * CH_DT_BLK(), zmm4);
            if (w_len > 5) _mm512_storeu_ps(dst + 5 * CH_DT_BLK(), zmm5);
            if (w_len > 6) _mm512_storeu_ps(dst + 6 * CH_DT_BLK(), zmm6);
            if (w_len > 7) _mm512_storeu_ps(dst + 7 * CH_DT_BLK(), zmm7);
            if (w_len > 8) _mm512_storeu_ps(dst + 8 * CH_DT_BLK(), zmm8);
            if (w_len > 9) _mm512_storeu_ps(dst + 9 * CH_DT_BLK(), zmm9);
            if (w_len > 10) _mm512_storeu_ps(dst + 10 * CH_DT_BLK(), zmm10);
            if (w_len > 11) _mm512_storeu_ps(dst + 11 * CH_DT_BLK(), zmm11);
            if (w_len > 12) _mm512_storeu_ps(dst + 12 * CH_DT_BLK(), zmm12);
            if (w_len > 13) _mm512_storeu_ps(dst + 13 * CH_DT_BLK(), zmm13);
            if (w_len > 14) _mm512_storeu_ps(dst + 14 * CH_DT_BLK(), zmm14);
            if (w_len > 15) _mm512_storeu_ps(dst + 15 * CH_DT_BLK(), zmm15);
            if (w_len > 16) _mm512_storeu_ps(dst + 16 * CH_DT_BLK(), zmm16);
            if (w_len > 17) _mm512_storeu_ps(dst + 17 * CH_DT_BLK(), zmm17);
            if (w_len > 18) _mm512_storeu_ps(dst + 18 * CH_DT_BLK(), zmm18);
            if (w_len > 19) _mm512_storeu_ps(dst + 19 * CH_DT_BLK(), zmm19);
            if (w_len > 20) _mm512_storeu_ps(dst + 20 * CH_DT_BLK(), zmm20);
            if (w_len > 21) _mm512_storeu_ps(dst + 21 * CH_DT_BLK(), zmm21);
            if (w_len > 22) _mm512_storeu_ps(dst + 22 * CH_DT_BLK(), zmm22);
            if (w_len > 23) _mm512_storeu_ps(dst + 23 * CH_DT_BLK(), zmm23);
            if (w_len > 24) _mm512_storeu_ps(dst + 24 * CH_DT_BLK(), zmm24);
            if (w_len > 25) _mm512_storeu_ps(dst + 25 * CH_DT_BLK(), zmm25);
            if (w_len > 26) _mm512_storeu_ps(dst + 26 * CH_DT_BLK(), zmm26);
            if (w_len > 27) _mm512_storeu_ps(dst + 27 * CH_DT_BLK(), zmm27);
            if (w_len > 28) _mm512_storeu_ps(dst + 28 * CH_DT_BLK(), zmm28);
            if (w_len > 29) _mm512_storeu_ps(dst + 29 * CH_DT_BLK(), zmm29);
            if (w_len > 30) _mm512_storeu_ps(dst + 30 * CH_DT_BLK(), zmm30);
        }
        src += w_len * src_sw_stride;
        sum += w_len * CH_DT_BLK();
        dst += w_len * CH_DT_BLK();
        ow -= w_len;
    } while (ow > 0);
#undef KW_COMPUTE_STEP
}

}}};

#endif
