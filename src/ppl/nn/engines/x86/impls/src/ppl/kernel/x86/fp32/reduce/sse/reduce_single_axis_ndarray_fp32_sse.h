#ifndef __ST_PPL_KERNEL_X86_FP32_REDUCE_SSE_REDUCE_SINGLE_AXIS_NDARRAY_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_REDUCE_SSE_REDUCE_SINGLE_AXIS_NDARRAY_FP32_SSE_H_

#include <nmmintrin.h>
#include <float.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/reduce/reduce_common.h"

#define _MM_ROP_PS(DST, A, B)                          \
    do {                                               \
        if (_op == REDUCE_MAX) {                       \
            DST = _mm_max_ps(A, B);                    \
        }                                              \
        if (_op == REDUCE_MIN) {                       \
            DST = _mm_min_ps(A, B);                    \
        }                                              \
        if (_op == REDUCE_SUM || _op == REDUCE_MEAN) { \
            DST = _mm_add_ps(A, B);                    \
        }                                              \
    } while (0)

#define ROP_ADD(A, B) (A) + (B)

#define ROP(DST, A, B)                                 \
    do {                                               \
        if (_op == REDUCE_MAX) {                       \
            DST = max(A, B);                           \
        }                                              \
        if (_op == REDUCE_MIN) {                       \
            DST = min(A, B);                           \
        }                                              \
        if (_op == REDUCE_SUM || _op == REDUCE_MEAN) { \
            DST = ROP_ADD(A, B);                       \
        }                                              \
    } while (0)

#define REDUCE_LOOP(R)                                                                     \
    do {                                                                                   \
        _MM_ROP_PS(mm_res0, _mm_loadu_ps(base_src + (R)*inner_dim + 0 * simd_w), mm_res0); \
        _MM_ROP_PS(mm_res1, _mm_loadu_ps(base_src + (R)*inner_dim + 1 * simd_w), mm_res1); \
        _MM_ROP_PS(mm_res2, _mm_loadu_ps(base_src + (R)*inner_dim + 2 * simd_w), mm_res2); \
        _MM_ROP_PS(mm_res3, _mm_loadu_ps(base_src + (R)*inner_dim + 3 * simd_w), mm_res3); \
    } while (0)

namespace ppl { namespace kernel { namespace x86 {

template <reduce_op_type_t _op>
ppl::common::RetCode reduce_single_axis_ndarray_fp32_sse(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int32_t *axes,
    const int32_t num_axes,
    float *dst)
{
    int64_t outer_dim       = 1;
    int64_t reduce_dim      = 1;
    int64_t inner_dim       = 1;
    const int64_t dim_count = src_shape->GetDimCount();
    for (int64_t i = 0; i < dim_count; i++) {
        if (i < axes[0]) {
            outer_dim *= src_shape->GetDim(i);
        } else if (i > axes[num_axes - 1]) {
            inner_dim *= src_shape->GetDim(i);
        } else {
            reduce_dim *= src_shape->GetDim(i);
        }
    }

    const int64_t simd_w        = 4;
    const int64_t unroll_reduce = simd_w;
    const int64_t unroll_inner  = 4 * simd_w;
    const int64_t reduce_body   = round(reduce_dim, unroll_reduce);
    const int64_t reduce_tail   = reduce_dim - reduce_body;
    float init_val              = 0.0f;
    if (_op == REDUCE_MAX) {
        init_val = -FLT_MAX;
    }
    if (_op == REDUCE_MIN) {
        init_val = FLT_MAX;
    }

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t o = 0; o < outer_dim; ++o) {
        for (int64_t i = 0; i < inner_dim; i += unroll_inner) {
            const int64_t inner_eff = min<int64_t>(inner_dim - i, unroll_inner);
            const float *base_src   = src + o * reduce_dim * inner_dim + i;
            float *base_dst         = dst + o * inner_dim + i;
            if (inner_eff == unroll_inner) {
                __m128 mm_res0, mm_res1, mm_res2, mm_res3;
                mm_res0 = _mm_set1_ps(init_val);
                mm_res1 = _mm_set1_ps(init_val);
                mm_res2 = _mm_set1_ps(init_val);
                mm_res3 = _mm_set1_ps(init_val);
                for (int64_t r = 0; r < reduce_body; r += unroll_reduce) {
                    REDUCE_LOOP(0);
                    REDUCE_LOOP(1);
                    REDUCE_LOOP(2);
                    REDUCE_LOOP(3);
                    base_src += unroll_reduce * inner_dim;
                }
                for (int64_t r = reduce_body; r < reduce_dim; ++r) {
                    REDUCE_LOOP(0);
                    base_src += inner_dim;
                }
                if (_op == REDUCE_MEAN) {
                    __m128 mm_rr = _mm_set1_ps(1.0f / reduce_dim);
                    mm_res0      = _mm_mul_ps(mm_res0, mm_rr);
                    mm_res1      = _mm_mul_ps(mm_res1, mm_rr);
                    mm_res2      = _mm_mul_ps(mm_res2, mm_rr);
                    mm_res3      = _mm_mul_ps(mm_res3, mm_rr);
                }
                _mm_storeu_ps(base_dst + 0 * simd_w, mm_res0);
                _mm_storeu_ps(base_dst + 1 * simd_w, mm_res1);
                _mm_storeu_ps(base_dst + 2 * simd_w, mm_res2);
                _mm_storeu_ps(base_dst + 3 * simd_w, mm_res3);
            } else if (inner_dim == 1) {
                float res_val = init_val;
                if (reduce_body) {
                    __m128 mm_res0;
                    float res[simd_w];
                    mm_res0 = _mm_set1_ps(init_val);
                    for (int64_t r = 0; r < reduce_body; r += unroll_reduce) {
                        _MM_ROP_PS(mm_res0, _mm_loadu_ps(base_src + r), mm_res0);
                    }
                    _mm_storeu_ps(res, mm_res0);

                    ROP(res[0], res[0], res[1]);
                    ROP(res[2], res[2], res[3]);
                    ROP(res[0], res[0], res[2]);
                    res_val = res[0];
                }
                if (reduce_tail) {
                    for (int64_t r = reduce_body; r < reduce_dim; ++r) {
                        ROP(res_val, res_val, base_src[r]);
                    }
                }
                if (_op == REDUCE_MEAN) {
                    res_val /= reduce_dim;
                }
                base_dst[0] = res_val;
            } else {
                const int64_t tail_offset = simd_w - (inner_eff % simd_w);
                if (inner_eff > 3 * simd_w) {
                    __m128 mm_res0, mm_res1, mm_res2, mm_res3;
                    mm_res0 = _mm_set1_ps(init_val);
                    mm_res1 = _mm_set1_ps(init_val);
                    mm_res2 = _mm_set1_ps(init_val);
                    mm_res3 = _mm_set1_ps(init_val);
                    for (int64_t r = 0; r < reduce_dim; ++r) {
                        if (_op == REDUCE_MAX) {
                            _MM_ROP_PS(mm_res0, _mm_loadu_ps(base_src + 0 * inner_dim + 0 * simd_w), mm_res0);
                            _MM_ROP_PS(mm_res1, _mm_loadu_ps(base_src + 0 * inner_dim + 1 * simd_w), mm_res1);
                            _MM_ROP_PS(mm_res2, _mm_loadu_ps(base_src + 0 * inner_dim + 2 * simd_w), mm_res2);
                            _MM_ROP_PS(mm_res3, _mm_loadu_ps(base_src + 0 * inner_dim + 3 * simd_w - tail_offset), mm_res3);
                        }
                        base_src += inner_dim;
                    }
                    if (_op == REDUCE_MEAN) {
                        __m128 mm_rr = _mm_set1_ps(1.0f / reduce_dim);
                        mm_res0      = _mm_mul_ps(mm_res0, mm_rr);
                        mm_res1      = _mm_mul_ps(mm_res1, mm_rr);
                        mm_res2      = _mm_mul_ps(mm_res2, mm_rr);
                        mm_res3      = _mm_mul_ps(mm_res3, mm_rr);
                    }
                    _mm_storeu_ps(base_dst + 0 * simd_w, mm_res0);
                    _mm_storeu_ps(base_dst + 1 * simd_w, mm_res1);
                    _mm_storeu_ps(base_dst + 2 * simd_w, mm_res2);
                    _mm_storeu_ps(base_dst + 3 * simd_w - tail_offset, mm_res3);
                } else if (inner_eff > 2 * simd_w) {
                    __m128 mm_res0, mm_res1, mm_res2;
                    mm_res0 = _mm_set1_ps(init_val);
                    mm_res1 = _mm_set1_ps(init_val);
                    mm_res2 = _mm_set1_ps(init_val);
                    for (int64_t r = 0; r < reduce_dim; ++r) {
                        _MM_ROP_PS(mm_res0, _mm_loadu_ps(base_src + 0 * inner_dim + 0 * simd_w), mm_res0);
                        _MM_ROP_PS(mm_res1, _mm_loadu_ps(base_src + 0 * inner_dim + 1 * simd_w), mm_res1);
                        _MM_ROP_PS(mm_res2, _mm_loadu_ps(base_src + 0 * inner_dim + 2 * simd_w - tail_offset), mm_res2);
                        base_src += inner_dim;
                    }
                    if (_op == REDUCE_MEAN) {
                        __m128 mm_rr = _mm_set1_ps(1.0f / reduce_dim);
                        mm_res0      = _mm_mul_ps(mm_res0, mm_rr);
                        mm_res1      = _mm_mul_ps(mm_res1, mm_rr);
                        mm_res2      = _mm_mul_ps(mm_res2, mm_rr);
                    }
                    _mm_storeu_ps(base_dst + 0 * simd_w, mm_res0);
                    _mm_storeu_ps(base_dst + 1 * simd_w, mm_res1);
                    _mm_storeu_ps(base_dst + 2 * simd_w - tail_offset, mm_res2);
                } else if (inner_eff > 1 * simd_w) {
                    __m128 mm_res0, mm_res1;
                    mm_res0 = _mm_set1_ps(init_val);
                    mm_res1 = _mm_set1_ps(init_val);
                    for (int64_t r = 0; r < reduce_dim; ++r) {
                        _MM_ROP_PS(mm_res0, _mm_loadu_ps(base_src + 0 * inner_dim + 0 * simd_w), mm_res0);
                        _MM_ROP_PS(mm_res1, _mm_loadu_ps(base_src + 0 * inner_dim + 1 * simd_w - tail_offset), mm_res1);
                        base_src += inner_dim;
                    }
                    if (_op == REDUCE_MEAN) {
                        __m128 mm_rr = _mm_set1_ps(1.0f / reduce_dim);
                        mm_res0      = _mm_mul_ps(mm_res0, mm_rr);
                        mm_res1      = _mm_mul_ps(mm_res1, mm_rr);
                    }
                    _mm_storeu_ps(base_dst + 0 * simd_w, mm_res0);
                    _mm_storeu_ps(base_dst + 1 * simd_w - tail_offset, mm_res1);
                } else {
                    float res[4] = {init_val, init_val, init_val, init_val};
                    for (int64_t r = 0; r < reduce_dim; ++r) {
                        for (int64_t i = 0; i < inner_eff; ++i) {
                            ROP(res[i], base_src[i], res[i]);
                        }
                        base_src += inner_dim;
                    }
                    if (_op == REDUCE_MEAN) {
                        for (int64_t i = 0; i < inner_eff; ++i) {
                            res[i] /= reduce_dim;
                        }
                    }
                    for (int64_t i = 0; i < inner_eff; ++i) {
                        base_dst[i] = res[i];
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_REDUCE_SSE_REDUCE_SINGLE_AXIS_NDARRAY_FP32_SSE_H_
