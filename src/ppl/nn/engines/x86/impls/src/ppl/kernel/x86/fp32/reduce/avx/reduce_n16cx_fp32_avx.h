#ifndef __ST_PPL_KERNEL_X86_FP32_REDUCE_AVX_REDUCE_N16CX_FP32_AVX_H_
#define __ST_PPL_KERNEL_X86_FP32_REDUCE_AVX_REDUCE_N16CX_FP32_AVX_H_

#include "ppl/kernel/x86/fp32/reduce/avx/reduce_kernel_fp32_avx.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

#define C_BLK() ((int64_t)16)

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_no_reduce_fp32_avx(
    const float *src,
    const int64_t width,
    const int64_t remain_c,
    float *dst)
{
    const int64_t simd_w     = 8;
    const int64_t unroll_len = 2;

    int64_t i = 0;
    for (; i + unroll_len <= width; i += unroll_len) {
        __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
        __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);
        __m256 v_src_2 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 2);
        __m256 v_src_3 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 3);

        __m256 v_dst_0 = _mm256_loadu_ps(dst + i * C_BLK() + simd_w * 0);
        __m256 v_dst_1 = _mm256_loadu_ps(dst + i * C_BLK() + simd_w * 1);
        __m256 v_dst_2 = _mm256_loadu_ps(dst + i * C_BLK() + simd_w * 2);
        __m256 v_dst_3 = _mm256_loadu_ps(dst + i * C_BLK() + simd_w * 3);

        v_dst_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_dst_0);
        v_dst_1 = reduce_vector_kernel_fp32_avx<_op>(v_src_1, v_dst_1);
        v_dst_2 = reduce_vector_kernel_fp32_avx<_op>(v_src_2, v_dst_2);
        v_dst_3 = reduce_vector_kernel_fp32_avx<_op>(v_src_3, v_dst_3);

        _mm256_storeu_ps(dst + i * C_BLK() + simd_w * 0, v_dst_0);
        _mm256_storeu_ps(dst + i * C_BLK() + simd_w * 1, v_dst_1);
        _mm256_storeu_ps(dst + i * C_BLK() + simd_w * 2, v_dst_2);
        _mm256_storeu_ps(dst + i * C_BLK() + simd_w * 3, v_dst_3);
    }
    for (; i < width; i++) {
        __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
        __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);

        __m256 v_dst_0 = _mm256_loadu_ps(dst + i * C_BLK() + simd_w * 0);
        __m256 v_dst_1 = _mm256_loadu_ps(dst + i * C_BLK() + simd_w * 1);

        v_dst_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_dst_0);
        v_dst_1 = reduce_vector_kernel_fp32_avx<_op>(v_src_1, v_dst_1);

        _mm256_storeu_ps(dst + i * C_BLK() + simd_w * 0, v_dst_0);
        _mm256_storeu_ps(dst + i * C_BLK() + simd_w * 1, v_dst_1);
    }
}

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_reduce_w_fp32_avx(
    const float *src,
    const int64_t width,
    const int64_t remain_c,
    float *dst)
{
    const int64_t simd_w     = 8;
    const int64_t unroll_len = 2;
    __m256 v_reduce_val_0    = _mm256_loadu_ps(dst + simd_w * 0);
    __m256 v_reduce_val_1    = _mm256_loadu_ps(dst + simd_w * 1);
    __m256 v_reduce_val_2    = _mm256_set1_ps(reduce_init_val_fp32<_op>());
    __m256 v_reduce_val_3    = _mm256_set1_ps(reduce_init_val_fp32<_op>());

    int64_t i = 0;
    for (; i + unroll_len <= width; i += unroll_len) {
        __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
        __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);
        __m256 v_src_2 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 2);
        __m256 v_src_3 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 3);

        v_reduce_val_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_reduce_val_0);
        v_reduce_val_1 = reduce_vector_kernel_fp32_avx<_op>(v_src_1, v_reduce_val_1);
        v_reduce_val_2 = reduce_vector_kernel_fp32_avx<_op>(v_src_2, v_reduce_val_2);
        v_reduce_val_3 = reduce_vector_kernel_fp32_avx<_op>(v_src_3, v_reduce_val_3);
    }
    for (; i < width; i++) {
        __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
        __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);

        v_reduce_val_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_reduce_val_0);
        v_reduce_val_1 = reduce_vector_kernel_fp32_avx<_op>(v_src_1, v_reduce_val_1);
    }

    if (width >= unroll_len) {
        v_reduce_val_0 = reduce_vector_kernel_fp32_avx<_op>(v_reduce_val_2, v_reduce_val_0);
        v_reduce_val_1 = reduce_vector_kernel_fp32_avx<_op>(v_reduce_val_3, v_reduce_val_1);
    }
    _mm256_storeu_ps(dst + simd_w * 0, v_reduce_val_0);
    _mm256_storeu_ps(dst + simd_w * 1, v_reduce_val_1);
}

#define VM  0xFFFFFFFF // valid mask
#define NVM 0x0 // not valid mask
static const uint32_t reduce_n16cx_mask_table[9][8]{
    {NVM, NVM, NVM, NVM, NVM, NVM, NVM, NVM},
    {VM, NVM, NVM, NVM, NVM, NVM, NVM, NVM},
    {VM, VM, NVM, NVM, NVM, NVM, NVM, NVM},
    {VM, VM, VM, NVM, NVM, NVM, NVM, NVM},
    {VM, VM, VM, VM, NVM, NVM, NVM, NVM},
    {VM, VM, VM, VM, VM, NVM, NVM, NVM},
    {VM, VM, VM, VM, VM, VM, NVM, NVM},
    {VM, VM, VM, VM, VM, VM, VM, NVM},
    {VM, VM, VM, VM, VM, VM, VM, VM},
};
#undef VM
#undef NVM

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_reduce_c_fp32_avx(
    const float *src,
    const int64_t width,
    const int64_t remain_c,
    float *dst)
{
    const int64_t simd_w     = 8;
    const int64_t unroll_len = 2;
    if (remain_c >= C_BLK()) {
        int64_t i = 0;
        for (; i + unroll_len <= width; i += unroll_len) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);
            __m256 v_src_2 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 2);
            __m256 v_src_3 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 3);

            v_src_0            = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);
            v_src_2            = reduce_vector_kernel_fp32_avx<_op>(v_src_2, v_src_3);
            float reduce_val_0 = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_src_0);
            float reduce_val_2 = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_src_2);

            dst[(i + 0) * C_BLK()] = reduce_scalar_kernel_fp32<_op>(dst[(i + 0) * C_BLK()], reduce_val_0);
            dst[(i + 1) * C_BLK()] = reduce_scalar_kernel_fp32<_op>(dst[(i + 1) * C_BLK()], reduce_val_2);
        }
        for (; i < width; i++) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);

            v_src_0            = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);
            float reduce_val_0 = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_src_0);

            dst[(i + 0) * C_BLK()] = reduce_scalar_kernel_fp32<_op>(dst[(i + 0) * C_BLK()], reduce_val_0);
        }
    } else {
        const int64_t valid_num_0 = min<int64_t>(remain_c, simd_w);
        const int64_t valid_num_1 = max<int64_t>(remain_c - simd_w, 0);
        const __m256 v_mask_0     = _mm256_loadu_ps((float *)reduce_n16cx_mask_table[valid_num_0]);
        const __m256 v_mask_1     = _mm256_loadu_ps((float *)reduce_n16cx_mask_table[valid_num_1]);

        const __m256 v_init_val = _mm256_set1_ps(reduce_init_val_fp32<_op>());
        const __m256 v_fill_0   = _mm256_blendv_ps(v_init_val, _mm256_setzero_ps(), v_mask_0);
        const __m256 v_fill_1   = _mm256_blendv_ps(v_init_val, _mm256_setzero_ps(), v_mask_1);

        int64_t i = 0;
        for (; i + unroll_len <= width; i += unroll_len) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);
            __m256 v_src_2 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 2);
            __m256 v_src_3 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 3);

            v_src_0 = _mm256_or_ps(_mm256_and_ps(v_src_0, v_mask_0), v_fill_0);
            v_src_1 = _mm256_or_ps(_mm256_and_ps(v_src_1, v_mask_1), v_fill_1);
            v_src_2 = _mm256_or_ps(_mm256_and_ps(v_src_2, v_mask_0), v_fill_0);
            v_src_3 = _mm256_or_ps(_mm256_and_ps(v_src_3, v_mask_1), v_fill_1);

            v_src_0            = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);
            v_src_2            = reduce_vector_kernel_fp32_avx<_op>(v_src_2, v_src_3);
            float reduce_val_0 = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_src_0);
            float reduce_val_2 = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_src_2);

            dst[(i + 0) * C_BLK()] = reduce_scalar_kernel_fp32<_op>(dst[(i + 0) * C_BLK()], reduce_val_0);
            dst[(i + 1) * C_BLK()] = reduce_scalar_kernel_fp32<_op>(dst[(i + 1) * C_BLK()], reduce_val_2);
        }
        for (; i < width; i++) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);

            v_src_0 = _mm256_or_ps(_mm256_and_ps(v_src_0, v_mask_0), v_fill_0);
            v_src_1 = _mm256_or_ps(_mm256_and_ps(v_src_1, v_mask_1), v_fill_1);

            v_src_0            = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);
            float reduce_val_0 = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_src_0);

            dst[(i + 0) * C_BLK()] = reduce_scalar_kernel_fp32<_op>(dst[(i + 0) * C_BLK()], reduce_val_0);
        }
    }
}

template <reduce_op_type_t _op>
void reduce_n16cx_lastdim_reduce_cw_fp32_avx(
    const float *src,
    const int64_t width,
    const int64_t remain_c,
    float *dst)
{
    const int64_t simd_w     = 8;
    const int64_t unroll_len = 2;
    if (remain_c >= C_BLK()) {
        __m256 v_reduce_val = _mm256_set1_ps(reduce_init_val_fp32<_op>());

        int64_t i = 0;
        for (; i + unroll_len <= width; i += unroll_len) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);
            __m256 v_src_2 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 2);
            __m256 v_src_3 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 3);

            v_src_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);
            v_src_2 = reduce_vector_kernel_fp32_avx<_op>(v_src_2, v_src_3);
            v_src_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_2);

            v_reduce_val = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_reduce_val);
        }
        for (; i < width; i++) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);

            v_src_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);

            v_reduce_val = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_reduce_val);
        }
        float reduce_val = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_reduce_val);
        dst[0]           = reduce_scalar_kernel_fp32<_op>(reduce_val, dst[0]);
    } else {
        __m256 v_reduce_val = _mm256_set1_ps(reduce_init_val_fp32<_op>());

        const int64_t valid_num_0 = min<int64_t>(remain_c, simd_w);
        const int64_t valid_num_1 = max<int64_t>(remain_c - simd_w, 0);
        const __m256 v_mask_0     = _mm256_loadu_ps((float *)reduce_n16cx_mask_table[valid_num_0]);
        const __m256 v_mask_1     = _mm256_loadu_ps((float *)reduce_n16cx_mask_table[valid_num_1]);

        const __m256 v_init_val = _mm256_set1_ps(reduce_init_val_fp32<_op>());
        const __m256 v_fill_0   = _mm256_blendv_ps(v_init_val, _mm256_setzero_ps(), v_mask_0);
        const __m256 v_fill_1   = _mm256_blendv_ps(v_init_val, _mm256_setzero_ps(), v_mask_1);

        int64_t i = 0;
        for (; i + unroll_len <= width; i += unroll_len) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);
            __m256 v_src_2 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 2);
            __m256 v_src_3 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 3);

            v_src_0 = _mm256_or_ps(_mm256_and_ps(v_src_0, v_mask_0), v_fill_0);
            v_src_1 = _mm256_or_ps(_mm256_and_ps(v_src_1, v_mask_1), v_fill_1);
            v_src_2 = _mm256_or_ps(_mm256_and_ps(v_src_2, v_mask_0), v_fill_0);
            v_src_3 = _mm256_or_ps(_mm256_and_ps(v_src_3, v_mask_1), v_fill_1);

            v_src_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);
            v_src_2 = reduce_vector_kernel_fp32_avx<_op>(v_src_2, v_src_3);
            v_src_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_2);

            v_reduce_val = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_reduce_val);
        }
        for (; i < width; i++) {
            __m256 v_src_0 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 0);
            __m256 v_src_1 = _mm256_loadu_ps(src + i * C_BLK() + simd_w * 1);

            v_src_0 = _mm256_or_ps(_mm256_and_ps(v_src_0, v_mask_0), v_fill_0);
            v_src_1 = _mm256_or_ps(_mm256_and_ps(v_src_1, v_mask_1), v_fill_1);

            v_src_0 = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_src_1);

            v_reduce_val = reduce_vector_kernel_fp32_avx<_op>(v_src_0, v_reduce_val);
        }
        float reduce_val = reduce_vector_all_lanes_kernel_fp32_avx<_op>(v_reduce_val);
        dst[0]           = reduce_scalar_kernel_fp32<_op>(reduce_val, dst[0]);
    }
}

template <reduce_op_type_t _op>
void reduce_n16cx_recursive_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t dim_idx,
    const int64_t *inc_src,
    const int64_t *inc_dst,
    const single_parallel_loop_config_t *pc,
    const int64_t c_dim_idx,
    int64_t remain_c,
    float *dst)
{
    if (dim_idx == src_shape->GetDimCount() - 1) { // last dim
        const bool reduce_on_w = src_shape->GetDim(dim_idx) != dst_shape->GetDim(dim_idx);
        const bool reduce_on_c = src_shape->GetDim(c_dim_idx) != dst_shape->GetDim(c_dim_idx);
        const int64_t width    = src_shape->GetDim(dim_idx);
        if (!reduce_on_c && !reduce_on_w) {
            reduce_n16cx_lastdim_no_reduce_fp32_avx<_op>(src, width, remain_c, dst);
        } else if (!reduce_on_c && reduce_on_w) {
            reduce_n16cx_lastdim_reduce_w_fp32_avx<_op>(src, width, remain_c, dst);
        } else if (reduce_on_c && !reduce_on_w) {
            reduce_n16cx_lastdim_reduce_c_fp32_avx<_op>(src, width, remain_c, dst);
        } else { // reduce_on_c && reduce_on_w
            reduce_n16cx_lastdim_reduce_cw_fp32_avx<_op>(src, width, remain_c, dst);
        }
    } else {
        const int64_t len = dim_idx == c_dim_idx ? div_up(src_shape->GetDim(dim_idx), C_BLK()) : src_shape->GetDim(dim_idx);
        if (pc->depth_of_loop == dim_idx && pc->num_threads > 1) { // parallel on this dim
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc->num_threads; t++) {
                const int64_t len_per_thread = div_up(len, pc->num_threads);
                const int64_t start_idx      = t * len_per_thread;
                const int64_t end_idx        = min(start_idx + len_per_thread, len);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    if (dim_idx == c_dim_idx) {
                        remain_c = src_shape->GetDim(c_dim_idx) - i * C_BLK();
                    }
                    reduce_n16cx_recursive_fp32_avx<_op>(
                        src_shape,
                        dst_shape,
                        src + i * inc_src[dim_idx],
                        dim_idx + 1,
                        inc_src,
                        inc_dst,
                        pc,
                        c_dim_idx,
                        remain_c,
                        dst + i * inc_dst[dim_idx]);
                }
            }
        } else {
            for (int64_t i = 0; i < len; i++) {
                if (dim_idx == c_dim_idx) {
                    remain_c = src_shape->GetDim(c_dim_idx) - i * C_BLK();
                }
                reduce_n16cx_recursive_fp32_avx<_op>(
                    src_shape,
                    dst_shape,
                    src + i * inc_src[dim_idx],
                    dim_idx + 1,
                    inc_src,
                    inc_dst,
                    pc,
                    c_dim_idx,
                    remain_c,
                    dst + i * inc_dst[dim_idx]);
            }
        }
    }
}

template <reduce_op_type_t _op>
ppl::common::RetCode reduce_n16cx_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    const int64_t c_dim_idx,
    float *dst)
{
    if (src_shape->GetDimCount() > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    // pad 1 to dst shape to keepdims
    ppl::nn::TensorShape padded_dst_shape = *src_shape;
    for (int64_t i = 0; i < num_axes; i++) {
        padded_dst_shape.SetDim(axes[i], 1);
    }
    padded_dst_shape.CalcPadding();

    // pre process
    reduce_preprocess_fp32_avx<_op>(dst, padded_dst_shape.GetElementsIncludingPadding());

    // prepare incs
    int64_t dim_count = padded_dst_shape.GetDimCount();
    int64_t inc_src[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc_dst[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_src = C_BLK();
    int64_t stride_dst = C_BLK();

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        int64_t src_dim = src_shape->GetDim(i);
        int64_t dst_dim = padded_dst_shape.GetDim(i);
        inc_src[i]      = src_dim == 1 ? 0 : stride_src;
        inc_dst[i]      = dst_dim == 1 ? 0 : stride_dst;

        if (i == c_dim_idx) {
            src_dim = div_up(src_dim, C_BLK());
            dst_dim = div_up(dst_dim, C_BLK());
        }
        stride_src *= src_dim;
        stride_dst *= dst_dim;
    }

    // calc parallel config
    std::vector<int64_t> loop_iter(src_shape->GetDims(), src_shape->GetDims() + dim_count);
    loop_iter[c_dim_idx] = div_up(loop_iter[c_dim_idx], C_BLK());
    std::vector<bool> forbid_mask(dim_count, false);
    for (int64_t i = 0; i < num_axes; i++) { // reduce dims cannot parallel
        forbid_mask[axes[i]] = true;
    }
    forbid_mask[dim_count - 1] = true; // last dim will not use omp because have much overhead when reduce on all before dims, or have error when reduce on last dim

    const bool reduce_on_c = src_shape->GetDim(c_dim_idx) != padded_dst_shape.GetDim(c_dim_idx);
    const bool reduce_on_w = src_shape->GetDim(dim_count - 1) != padded_dst_shape.GetDim(dim_count - 1);
    int64_t load_per_task;
    int64_t store_per_task;
    if (!reduce_on_c && !reduce_on_w) {
        load_per_task  = C_BLK() * 2 * sizeof(float);
        store_per_task = C_BLK() * sizeof(float);
    } else if (!reduce_on_c && reduce_on_w) {
        load_per_task  = C_BLK() * sizeof(float);
        store_per_task = 0;
    } else if (reduce_on_c && !reduce_on_w) {
        load_per_task  = (C_BLK() + 1) * sizeof(float);
        store_per_task = 1 * sizeof(float);
    } else {
        load_per_task  = C_BLK() * sizeof(float);
        store_per_task = 0;
    }

    auto pc = select_single_parallel_loop_with_mask(
        loop_iter,
        forbid_mask,
        ppl::common::ISA_X86_AVX,
        load_per_task,
        store_per_task,
        C_BLK() * sizeof(float),
        1);

    // reduce
    reduce_n16cx_recursive_fp32_avx<_op>(
        src_shape,
        &padded_dst_shape,
        src,
        0,
        inc_src,
        inc_dst,
        &pc,
        c_dim_idx,
        src_shape->GetDim(c_dim_idx),
        dst);

    // post process
    int64_t reduce_factor = 1;
    for (int64_t i = 0; i < dim_count; i++) {
        reduce_factor *= src_shape->GetDim(i) / padded_dst_shape.GetDim(i);
    }
    reduce_postprocess_fp32_avx<_op>(dst, padded_dst_shape.GetElementsIncludingPadding(), reduce_factor);

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_REDUCE_AVX_REDUCE_N16CX_FP32_AVX_H_
