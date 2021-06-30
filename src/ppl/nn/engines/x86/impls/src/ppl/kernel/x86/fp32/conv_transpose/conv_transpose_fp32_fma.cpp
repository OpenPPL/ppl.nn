#include <immintrin.h>
#include <string.h>
#include <memory>
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm_v2.h"

namespace ppl { namespace kernel { namespace x86 {

static void col2im_ndarray_fp32(
    const float *col,
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
    const float beta,
    float *image)
{
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t c_img = 0; c_img < num_output; ++c_img) {
        if (beta == 0.0f) {
            memset(image + c_img * img_h * img_w, 0.0f, img_h * img_w * sizeof(float));
        } else {
            for (int64_t hw = 0; hw < img_h * img_w; ++hw) {
                image[c_img * img_h * img_w + hw] *= beta;
            }
        }
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t c_col    = c_img * kernel_h * kernel_w + kh * kernel_w + kw;
                int64_t w_offset = kw * hole_w;
                int64_t h_offset = kh * hole_h;
                for (int64_t h = 0; h < col_h; ++h) {
                    for (int64_t w = 0; w < col_w; ++w) {
                        int64_t h_pad = h * stride_h - pad_h + h_offset;
                        int64_t w_pad = w * stride_w - pad_w + w_offset;
                        if (h_pad >= 0 && h_pad < img_h && w_pad >= 0 && w_pad < img_w) {
                            image[(c_img * img_h + h_pad) * img_w + w_pad] += col[(c_col * col_h + h) * col_w + w];
                        }
                    }
                }
            }
        }
    }
}

int64_t conv_transpose_ndarray_fp32_fma_get_buffer_bytes(
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
    const bool do_col2im     = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 &&
                             pad_w == 0 && stride_h == 1 && stride_w == 1);
    const int64_t col2im_len = !do_col2im ? 0 : int64_t(num_output) * kernel_h * kernel_w * src_h * src_w;

    gemm_v2_param_fp32 param;
    param.M        = num_output * kernel_h * kernel_w;
    param.N        = src_h * src_w;
    param.K        = channels;
    param.lda      = param.M;
    param.ldb      = param.N;
    param.ldy   = param.N;
    param.isa_flag = common::ISA_X86_FMA;
    param.trans_A   = 1;
    param.trans_B   = 0;
    auto executor  = std::unique_ptr<gemm_v2_executor_fp32>(create_gemm_v2_executor_fp32(param));

    const int64_t gemm_bytes = executor != nullptr ? executor->get_buffer_bytes() : 0;

    return col2im_len * sizeof(float) + gemm_bytes;
}

ppl::common::RetCode conv_transpose_ndarray_fp32_fma(
    const float *input,
    const float *filter,
    const float *bias,
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
    float *tmp_buffer,
    float *output)
{
    int64_t M = num_output * kernel_h * kernel_w;
    int64_t N = src_h * src_w;
    int64_t K = channels;

    int64_t lda   = M;
    int64_t ldb   = N;
    int64_t ldout = N;

    gemm_v2_param_fp32 param;
    param.M        = M;
    param.N        = N;
    param.K        = K;
    param.lda      = lda;
    param.ldb      = ldb;
    param.ldy   = ldout;
    param.isa_flag = common::ISA_X86_FMA;
    param.trans_A   = 1;
    param.trans_B   = 0; // other param use default value
    auto executor  = std::unique_ptr<gemm_v2_executor_fp32>(create_gemm_v2_executor_fp32(param));
    if (!executor) {
        return ppl::common::RC_UNSUPPORTED;
    }

    float *gemm_buffer = tmp_buffer;
    executor->set_temp_buffer(gemm_buffer);

    float *col2im_buffer = tmp_buffer + (executor->get_buffer_bytes() / sizeof(float));
    const bool do_col2im = !(kernel_h == 1 && kernel_w == 1 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1);

    for (int64_t b = 0; b < batch; ++b) {
        const float *src_d = input + b * channels * src_h * src_w;
        float *dst_d       = output + b * num_output * dst_h * dst_w;
        float *gemm_out;

        if (do_col2im) {
            gemm_out = col2im_buffer;
        } else {
            gemm_out = dst_d;
        }

        executor->get_param_mutable().src_A = filter;
        executor->get_param_mutable().src_B = src_d;
        executor->get_param_mutable().dst_Y  = gemm_out;
        executor->execute();
        if (do_col2im) {
            col2im_ndarray_fp32(col2im_buffer, src_h, src_w, num_output, dst_h, dst_w, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, hole_h, hole_w, 0.0f, dst_d);
        }

        if (nullptr != bias) {
            const int64_t simd_w = 8;
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t oc = 0; oc < num_output; ++oc) {
                __m256 vbias = _mm256_set1_ps(bias[oc]);
                for (int64_t hw = 0; hw < round(dst_h * dst_w, simd_w); hw += simd_w) {
                    __m256 vdst = _mm256_loadu_ps(dst_d + oc * dst_h * dst_w + hw);
                    vdst        = _mm256_add_ps(vdst, vbias);
                    _mm256_storeu_ps(dst_d + oc * dst_h * dst_w + hw, vdst);
                }
                for (int64_t hw = round(dst_h * dst_w, simd_w); hw < dst_h * dst_w; ++hw) {
                    dst_d[oc * dst_h * dst_w + hw] += bias[oc];
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
