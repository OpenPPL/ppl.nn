#include <immintrin.h>
#include <math.h>
#include <memory>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm_v2.h"

namespace ppl { namespace kernel { namespace x86 {

#define MAX_PARALLEL_IMG_NUM() 8

uint64_t conv2d_dynamic_ndarray_fp32_avx512_get_buffer_bytes(
    const int32_t batch,
    const int32_t num_output,
    const int32_t group,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w)
{
#ifdef PPL_USE_X86_OMP
    int64_t lbatch = min(min(batch * group, PPL_OMP_MAX_THREADS()), MAX_PARALLEL_IMG_NUM());
#else
    int64_t lbatch = 1;
#endif
    const bool do_im2col     = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);
    const int64_t im2col_len = !do_im2col ? 0 : int64_t(channels) * kernel_h * kernel_w * dst_h * dst_w;
    while (lbatch * im2col_len * sizeof(float) > 1024 * 1024 * 1024) {
        if (lbatch > 1) {
            --lbatch;
        } else {
            break;
        }
    }

    gemm_v2_param_fp32 param;
    param.M                   = num_output;
    param.N                   = dst_h * dst_w;
    param.K                   = channels * kernel_h * kernel_w;
    param.lda                 = param.K;
    param.ldb                 = param.N;
    param.ldy                 = param.N;
    param.isa_flag            = ppl::common::ISA_X86_AVX512;
    param.trans_A              = 0;
    param.trans_B              = 0;
    auto executor             = std::unique_ptr<gemm_v2_executor_fp32>(create_gemm_v2_executor_fp32(param));
    const uint64_t gemm_bytes = executor ? executor->get_buffer_bytes() : 0;

    return lbatch * im2col_len * sizeof(float) + gemm_bytes;
}

static void batch_im2col_ndarray_fp32_avx512(
    const float **image_list,
    float **expanded_image_list,
    const int32_t batch,
    const int32_t channels,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t hole_h,
    const int32_t hole_w)
{
    const int64_t simd_w = 16;
    __m512 mmzero        = _mm512_setzero_ps();
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t ic = 0; ic < channels; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    const int64_t expanded_id = ic * kernel_h * kernel_w + kh * kernel_w + kw;
                    const int64_t ih          = kh * hole_h;
                    const int64_t iw          = kw * hole_w;

                    const int64_t oh_beg = max<int64_t>((int64_t)ceilf((pad_h - ih) / (float)(stride_h)), 0);
                    const int64_t oh_end = max<int64_t>(oh_beg, min<int64_t>((int64_t)ceilf((src_h + pad_h - ih) / (float)(stride_h)), dst_h));
                    const int64_t ow_beg = max<int64_t>((int64_t)ceilf((pad_w - iw) / (float)(stride_w)), 0);
                    const int64_t ow_end = max<int64_t>(ow_beg, min<int64_t>((int64_t)ceilf((src_w + pad_w - iw) / (float)(stride_w)), dst_w));

                    const float *in_d = image_list[b] + ic * src_h * src_w;
                    float *out_d      = expanded_image_list[b] + expanded_id * dst_w * dst_h;

                    for (int64_t oh = 0; oh < oh_beg; ++oh) {
                        int64_t ow = 0;
                        for (; ow <= dst_w - simd_w; ow += simd_w) {
                            _mm512_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                        }
                        for (; ow < dst_w; ++ow) {
                            out_d[oh * dst_w + ow] = 0.0f;
                        }
                    }

                    int64_t ih_paded = oh_beg * stride_h - pad_h + ih;
                    int64_t iw_paded = ow_beg * stride_w - pad_w + iw;
                    for (int64_t oh = oh_beg; oh < oh_end; ++oh, ih_paded += stride_h) {
                        int64_t ow = 0;
                        for (; ow <= ow_beg - simd_w; ow += simd_w) {
                            _mm512_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                        }
                        for (; ow < ow_beg; ++ow) {
                            out_d[oh * dst_w + ow] = 0.0f;
                        }

                        int64_t out_id = oh * dst_w + ow;
                        int64_t in_id  = ih_paded * src_w + iw_paded;
                        for (; ow < ow_end; ++ow, ++out_id, in_id += stride_w) {
                            out_d[out_id] = in_d[in_id];
                        }

                        for (; ow <= dst_w - simd_w; ow += simd_w) {
                            _mm512_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                        }
                        for (; ow < dst_w; ++ow) {
                            out_d[oh * dst_w + ow] = 0.0f;
                        }
                    }

                    for (int64_t oh = oh_end; oh < dst_h; ++oh) {
                        int64_t ow = 0;
                        for (; ow <= dst_w - simd_w; ow += simd_w) {
                            _mm512_storeu_ps(out_d + oh * dst_w + ow, mmzero);
                        }
                        for (; ow < dst_w; ++ow) {
                            out_d[oh * dst_w + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

ppl::common::RetCode conv2d_dynamic_ndarray_fp32_avx512(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t group,
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
    float *tmp_buffer,
    float *output)
{
    const int64_t M      = num_output;
    const int64_t N      = dst_h * dst_w;
    const int64_t K      = channels * kernel_h * kernel_w;
    const bool do_im2col = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);

    gemm_v2_param_fp32 param;
    param.M        = M;
    param.N        = N;
    param.K        = K;
    param.lda      = param.K;
    param.ldb      = param.N;
    param.ldy      = param.N;
    param.isa_flag = common::ISA_X86_AVX512;
    param.trans_A   = 0;
    param.trans_B   = 0;
    auto executor  = std::unique_ptr<gemm_v2_executor_fp32>(create_gemm_v2_executor_fp32(param));
    if (!executor) {
        return common::RC_UNSUPPORTED;
    }
    float *tmp_gemm = tmp_buffer;
    executor->set_temp_buffer(tmp_gemm);

    float *tmp_im2col = tmp_buffer + (executor->get_buffer_bytes() / sizeof(float));

    const int64_t src_off    = channels * src_h * src_w;
    const int64_t dst_off    = num_output * dst_h * dst_w;
    const int64_t flt_off    = num_output * channels * kernel_h * kernel_w;
    const int64_t bia_off    = num_output;
    const int64_t im2col_off = !do_im2col ? 0 : int64_t(channels) * kernel_h * kernel_w * dst_h * dst_w;

    const int64_t simd_w = 16;
#ifdef PPL_USE_X86_OMP
    int64_t lbatch = min(min(batch * group, PPL_OMP_MAX_THREADS()), MAX_PARALLEL_IMG_NUM());
#else
    int64_t lbatch = 1;
#endif
    while (lbatch * im2col_off * sizeof(float) > 1024 * 1024 * 1024) {
        if (lbatch > 1) {
            --lbatch;
        } else {
            break;
        }
    }

    const float **src_list    = new const float *[batch * group];
    float **im2col_list       = new float *[lbatch];
    float **dst_list          = new float *[batch * group];
    const float **filter_list = new const float *[batch * group];
    const float **bias_list   = new const float *[batch * group];

    for (int64_t lb = 0; lb < lbatch; ++lb) {
        im2col_list[lb] = tmp_im2col + lb * im2col_off;
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < group; ++g) {
            src_list[(b * group + g)]    = input + (b * group + g) * src_off;
            dst_list[(b * group + g)]    = output + (b * group + g) * dst_off;
            filter_list[(b * group + g)] = filter + g * flt_off;
            bias_list[(b * group + g)]   = bias + g * bia_off;
        }
    }

    for (int64_t bg = 0; bg < batch * group; bg += lbatch) {
        const int64_t bg_len_eff = min<int64_t>(batch * group - bg, lbatch);

        if (do_im2col) {
            batch_im2col_ndarray_fp32_avx512(
                src_list + bg,
                im2col_list,
                bg_len_eff,
                channels,
                src_h,
                src_w,
                dst_h,
                dst_w,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                hole_h,
                hole_w);
        } else {
            for (int64_t ibg = 0; ibg < bg_len_eff; ++ibg) {
                im2col_list[ibg] = const_cast<float *>(src_list[bg + ibg]);
            }
        }

        for (int64_t b = 0; b < bg_len_eff; b++) {
            executor->get_param_mutable().src_A = filter_list[bg + b];
            executor->get_param_mutable().src_B = im2col_list[b];
            executor->get_param_mutable().dst_Y  = dst_list[bg + b];
            executor->execute();
        }

        if (bias != nullptr) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#else
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
            for (int64_t ibg = 0; ibg < bg_len_eff; ++ibg) {
                for (int64_t m = 0; m < M; ++m) {
                    __m512 mmbias = _mm512_set1_ps(bias_list[bg + ibg][m]);
                    for (int64_t n = 0; n < round(N, simd_w); n += simd_w) {
                        _mm512_storeu_ps(dst_list[bg + ibg] + m * N + n,
                                         _mm512_add_ps(mmbias, _mm512_loadu_ps(dst_list[bg + ibg] + m * N + n)));
                    }
                    for (int64_t n = round(N, simd_w); n < N; ++n) {
                        dst_list[bg + ibg][m * N + n] += bias_list[bg + ibg][m];
                    }
                }
            }
        }
    }
    delete[] src_list;
    delete[] im2col_list;
    delete[] dst_list;
    delete[] filter_list;
    delete[] bias_list;

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode x86_conv2d_dynamic_ndarray_with_relu_fp32_avx512(
    const float *input,
    const float *filter,
    const float *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t group,
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
    float *tmp_buffer,
    float *output)
{
    const int64_t M      = num_output;
    const int64_t N      = dst_h * dst_w;
    const int64_t K      = channels * kernel_h * kernel_w;
    const bool do_im2col = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);

    gemm_v2_param_fp32 param;
    param.M        = M;
    param.N        = N;
    param.K        = K;
    param.lda      = param.K;
    param.ldb      = param.N;
    param.ldy      = param.N;
    param.isa_flag = ppl::common::ISA_X86_AVX512;
    param.trans_A   = 0;
    param.trans_B   = 0;
    auto executor  = std::unique_ptr<gemm_v2_executor_fp32>(create_gemm_v2_executor_fp32(param));
    if (!executor) {
        return ppl::common::RC_UNSUPPORTED;
    }
    float *tmp_gemm = tmp_buffer;
    executor->set_temp_buffer(tmp_gemm);

    float *tmp_im2col = tmp_buffer + (executor->get_buffer_bytes() / sizeof(float));

    const int64_t src_off    = channels * src_h * src_w;
    const int64_t dst_off    = num_output * dst_h * dst_w;
    const int64_t flt_off    = num_output * channels * kernel_h * kernel_w;
    const int64_t bia_off    = num_output;
    const int64_t im2col_off = !do_im2col ? 0 : channels * kernel_h * kernel_w * dst_h * dst_w;

    const int64_t simd_w = 16;
#ifdef PPL_USE_X86_OMP
    int64_t lbatch = min(min(batch * group, PPL_OMP_MAX_THREADS()), MAX_PARALLEL_IMG_NUM());
#else
    int64_t lbatch = 1;
#endif
    while (lbatch * im2col_off * sizeof(float) > 1024 * 1024 * 1024) {
        if (lbatch > 1) {
            --lbatch;
        } else {
            break;
        }
    }

    const float **src_list    = new const float *[batch * group];
    float **im2col_list       = new float *[lbatch];
    float **dst_list          = new float *[batch * group];
    const float **filter_list = new const float *[batch * group];
    const float **bias_list   = new const float *[batch * group];

    for (int64_t lb = 0; lb < lbatch; ++lb) {
        im2col_list[lb] = tmp_im2col + lb * im2col_off;
    }

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t g = 0; g < group; ++g) {
            src_list[(b * group + g)]    = input + (b * group + g) * src_off;
            dst_list[(b * group + g)]    = output + (b * group + g) * dst_off;
            filter_list[(b * group + g)] = filter + g * flt_off;
            bias_list[(b * group + g)]   = bias + g * bia_off;
        }
    }

    for (int64_t bg = 0; bg < batch * group; bg += lbatch) {
        const int64_t bg_len_eff = min(batch * group - bg, lbatch);

        if (do_im2col) {
            batch_im2col_ndarray_fp32_avx512(
                src_list + bg,
                im2col_list,
                bg_len_eff,
                channels,
                src_h,
                src_w,
                dst_h,
                dst_w,
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                hole_h,
                hole_w);
        } else {
            for (int64_t ibg = 0; ibg < bg_len_eff; ++ibg) {
                im2col_list[ibg] = const_cast<float *>(src_list[bg + ibg]);
            }
        }

        for (int64_t b = 0; b < bg_len_eff; b++) {
            executor->get_param_mutable().src_A = filter_list[bg + b];
            executor->get_param_mutable().src_B = im2col_list[b];
            executor->get_param_mutable().dst_Y = dst_list[bg + b];
            executor->execute();
        }

        if (bias != nullptr) {
            __m512 mmzero = _mm512_setzero_ps();
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#else
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
            for (int64_t ibg = 0; ibg < bg_len_eff; ++ibg) {
                for (int64_t m = 0; m < M; ++m) {
                    __m512 mmbias = _mm512_set1_ps(bias_list[bg + ibg][m]);
                    for (int64_t n = 0; n < round(N, simd_w); n += simd_w) {
                        _mm512_storeu_ps(dst_list[bg + ibg] + m * N + n,
                                         _mm512_max_ps(mmzero, _mm512_add_ps(mmbias, _mm512_loadu_ps(dst_list[bg + ibg] + m * N + n))));
                    }
                    for (int64_t n = round(N, simd_w); n < N; ++n) {
                        dst_list[bg + ibg][m * N + n] = max(0.0f, dst_list[bg + ibg][m * N + n] + bias_list[bg + ibg][m]);
                    }
                }
            }
        } else {
            __m512 mmzero = _mm512_setzero_ps();
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#else
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
            for (int64_t ibg = 0; ibg < bg_len_eff; ++ibg) {
                for (int64_t m = 0; m < M; ++m) {
                    for (int64_t n = 0; n < round(N, simd_w); n += simd_w) {
                        _mm512_storeu_ps(dst_list[bg + ibg] + m * N + n,
                                         _mm512_max_ps(mmzero, _mm512_loadu_ps(dst_list[bg + ibg] + m * N + n)));
                    }
                    for (int64_t n = round(N, simd_w); n < N; ++n) {
                        dst_list[bg + ibg][m * N + n] = max(0.0f, dst_list[bg + ibg][m * N + n]);
                    }
                }
            }
        }
    }
    delete[] src_list;
    delete[] im2col_list;
    delete[] dst_list;
    delete[] filter_list;
    delete[] bias_list;

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
