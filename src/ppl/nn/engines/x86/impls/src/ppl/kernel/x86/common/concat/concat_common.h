#ifndef __ST_PPL_KERNEL_X86_COMMON_CONCAT_CONCAT_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_CONCAT_CONCAT_COMMON_H_

#include <vector>
#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

struct concat_parallel_info {
    concat_parallel_info() {}
    concat_parallel_info(
        const int32_t blk_idx,
        const int32_t start,
        const int32_t end)
    {
        this->blk_idx = blk_idx;
        this->start   = start;
        this->end     = end;
    }

    int32_t blk_idx;
    int32_t start;
    int32_t end;
};

template <typename eT>
ppl::common::RetCode concat_ndarray(
    const ppl::nn::TensorShape **src_shape_list,
    const eT **src_list,
    const int32_t num_src,
    const int32_t axis,
    eT *dst)
{
    const int32_t ndims      = int32_t(src_shape_list[0]->GetDimCount());
    const int32_t fixed_axis = axis < 0 ? ndims + axis : axis;

    int64_t outer_dim = 1;
    int64_t inner_dim = 1;
    for (int32_t i = 0; i < fixed_axis; ++i) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int32_t i = fixed_axis + 1; i < ndims; ++i) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0]       = 0;
    for (int32_t i = 1; i < num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(fixed_axis);
    }
    const int64_t dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(fixed_axis);

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (dst_concat_dim * inner_dim * sizeof(eT) < 16 && num_src >= num_threads) { // when has small inner dims(usually occured when fixed_axis == ndims - 1), use scalar code to replace memcpy & change index calculating method
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t n = 0; n < num_src; n++) {
            eT *p_dst                    = dst + dst_offset[n] * inner_dim;
            const eT *p_src              = src_list[n];
            const int64_t copy_elements = src_shape_list[n]->GetDim(fixed_axis) * inner_dim;
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t j = 0; j < copy_elements; j++) {
                    p_dst[j] = p_src[j];
                }
                p_dst += dst_concat_dim * inner_dim;
                p_src += copy_elements;
            }
        }
    } else if (num_src * outer_dim >= num_threads) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t i = 0; i < outer_dim; i++) {
            for (int64_t n = 0; n < num_src; n++) {
                memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                       src_list[n] + i * src_shape_list[n]->GetDim(fixed_axis) * inner_dim,
                       src_shape_list[n]->GetDim(fixed_axis) * inner_dim * sizeof(eT));
            }
        }
    } else { // parallel divide along concat dim, may across different src
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), dst_concat_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t concat_dim_per_thread = div_up(dst_concat_dim, num_threads);
            const int64_t start_concat_dim      = concat_dim_per_thread * thread_id;
            const int64_t end_concat_dim        = min(concat_dim_per_thread * (thread_id + 1), dst_concat_dim);

            if (start_concat_dim < end_concat_dim) {
                int64_t start_blk_idx = num_src - 1;
                int64_t end_blk_idx   = num_src - 1;
                for (int64_t i = 0; i < num_src - 1; i++) {
                    if (start_concat_dim >= dst_offset[i] && start_concat_dim < dst_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_concat_dim >= dst_offset[i] && end_concat_dim < dst_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_concat_dim - dst_offset[start_blk_idx];
                int64_t end_axis_idx   = end_concat_dim - dst_offset[end_blk_idx];

                std::vector<concat_parallel_info> infos;
                if (start_blk_idx == end_blk_idx) { // copy from single src
                    infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                } else { // start_blk_idx < end_blk_idx, copy from multiple src
                    infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, src_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                    for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                        infos.emplace_back(concat_parallel_info(i, 0, src_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                    }
                    infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                }

                for (int64_t i = 0; i < outer_dim; i++) {
                    for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                        const concat_parallel_info &info = infos[j];

                        const eT *p_src = src_list[info.blk_idx] + (i * src_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dim;
                        eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                        const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                        memcpy(p_dst, p_src, size);
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}


template <typename eT>
ppl::common::RetCode concat_n16cx_interleave_channels(
    const ppl::nn::TensorShape **src_shape_list,
    const eT **src_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx,
    eT *dst)
{
    const int32_t ndims = int32_t(src_shape_list[0]->GetDimCount());
    int64_t outer_dim   = 1;
    int64_t inner_dim   = 1;
    for (int32_t i = 0; i < c_dim_idx; ++i) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int32_t i = c_dim_idx + 1; i < ndims; ++i) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0]       = 0;
    for (int32_t i = 1; i < num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t c_blk        = 16;
    const int64_t dst_channels = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(c_dim_idx);
    const int64_t padded_oc    = round_up(dst_channels, c_blk);

    const int64_t num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), inner_dim);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        const int64_t inner_dim_per_thread = div_up(inner_dim, num_threads);
        const int64_t start_inner_dim      = inner_dim_per_thread * thread_id;
        const int64_t end_inner_dim        = min(inner_dim_per_thread * (thread_id + 1), inner_dim);

        if (start_inner_dim < end_inner_dim) {
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int32_t n = 0; n < num_src; n++) {
                    const int32_t src_channels = src_shape_list[n]->GetDim(c_dim_idx);
                    const int32_t padded_ic    = round_up(src_channels, c_blk);
                    for (int32_t ic = 0; ic < padded_ic; ic += c_blk) {
                        const int32_t oc = dst_offset[n] + ic;
                        const eT *p_src   = src_list[n] + i * padded_ic * inner_dim + ic * inner_dim;
                        eT *p_dst         = dst + i * padded_oc * inner_dim + round(oc, c_blk) * inner_dim;
                        if (oc % c_blk == 0) { // no interleave on this 16c
                            memcpy(p_dst + start_inner_dim * c_blk, p_src + start_inner_dim * c_blk, (end_inner_dim - start_inner_dim) * c_blk * sizeof(eT));
                        } else { // has interleave on this 16c
                            const int32_t c_offset = c_blk - (oc % c_blk);
                            const int32_t c_end    = min(src_channels - ic, (int32_t)c_blk);
                            eT *p_dst_next_16c      = p_dst + c_blk * inner_dim;
                            for (int64_t id = start_inner_dim; id < end_inner_dim; id++) {
                                // interleave copy
                                for (int32_t c = 0; c < c_offset; c++) {
                                    p_dst[id * c_blk + c_blk - c_offset + c] = p_src[id * c_blk + c];
                                }
                                for (int32_t c = c_offset; c < c_end; c++) {
                                    p_dst_next_16c[id * c_blk + c - c_offset] = p_src[id * c_blk + c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode concat_n16cx(
    const ppl::nn::TensorShape **src_shape_list,
    const eT **src_list,
    const int32_t num_src,
    const int32_t axis,
    eT *dst)
{
    const int32_t ndims      = int32_t(src_shape_list[0]->GetDimCount());
    const int32_t fixed_axis = axis < 0 ? ndims + axis : axis;
    const int32_t c_dim_idx  = 1;
    const int64_t c_blk      = 16;

    if (fixed_axis == c_dim_idx) { // concat on C dim
        for (int32_t i = 0; i < num_src - 1; i++) {
            if (src_shape_list[i]->GetDim(c_dim_idx) % c_blk != 0) {
                return concat_n16cx_interleave_channels<eT>(
                    src_shape_list,
                    src_list,
                    num_src,
                    axis,
                    c_dim_idx,
                    dst);
            }
        }
    }

    int64_t outer_dim = 1;
    int64_t inner_dim = c_blk;
    for (int32_t i = 0; i < fixed_axis; ++i) {
        if (i == c_dim_idx) {
            outer_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            outer_dim *= src_shape_list[0]->GetDim(i);
        }
    }
    for (int32_t i = fixed_axis + 1; i < ndims; ++i) {
        if (i == c_dim_idx) {
            inner_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            inner_dim *= src_shape_list[0]->GetDim(i);
        }
    }

    std::vector<int64_t> dst_offset(num_src);
    int64_t dst_concat_dim = 0;
    dst_offset[0]          = 0;
    if (fixed_axis == c_dim_idx) {
        for (int32_t i = 1; i < num_src; ++i) {
            dst_offset[i] = dst_offset[i - 1] + div_up(src_shape_list[i - 1]->GetDim(fixed_axis), c_blk);
        }
        dst_concat_dim = dst_offset[num_src - 1] + div_up(src_shape_list[num_src - 1]->GetDim(fixed_axis), c_blk);
    } else {
        for (int32_t i = 1; i < num_src; ++i) {
            dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(fixed_axis);
        }
        dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(fixed_axis);
    }

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (num_src * outer_dim >= num_threads) {
        if (fixed_axis == c_dim_idx) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t n = 0; n < num_src; n++) {
                    memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                           src_list[n] + i * div_up(src_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dim,
                           div_up(src_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dim * sizeof(eT));
                }
            }
        } else {
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t n = 0; n < num_src; n++) {
                    memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                           src_list[n] + i * src_shape_list[n]->GetDim(fixed_axis) * inner_dim,
                           src_shape_list[n]->GetDim(fixed_axis) * inner_dim * sizeof(eT));
                }
            }
        }
    } else { // parallel divide along concat dim, may across different src
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), dst_concat_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t concat_dim_per_thread = div_up(dst_concat_dim, num_threads);
            const int64_t start_concat_dim      = concat_dim_per_thread * thread_id;
            const int64_t end_concat_dim =      min(concat_dim_per_thread * (thread_id + 1), dst_concat_dim);

            if (start_concat_dim < end_concat_dim) {
                int64_t start_blk_idx = num_src - 1;
                int64_t end_blk_idx   = num_src - 1;
                for (int64_t i = 0; i < num_src - 1; i++) {
                    if (start_concat_dim >= dst_offset[i] && start_concat_dim < dst_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_concat_dim >= dst_offset[i] && end_concat_dim < dst_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_concat_dim - dst_offset[start_blk_idx];
                int64_t end_axis_idx   = end_concat_dim - dst_offset[end_blk_idx];

                if (fixed_axis == c_dim_idx) {
                    std::vector<concat_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy from single src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy from multiple src
                        infos.emplace_back(concat_parallel_info(
                            start_blk_idx, start_axis_idx, div_up(src_shape_list[start_blk_idx]->GetDim(fixed_axis), c_blk))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(concat_parallel_info(i, 0, div_up(src_shape_list[i]->GetDim(fixed_axis), c_blk))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dim; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const concat_parallel_info& info = infos[j];

                            const eT *p_src = src_list[info.blk_idx] + (i * div_up(src_shape_list[info.blk_idx]->GetDim(fixed_axis), c_blk) + info.start) * inner_dim;
                            eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                            const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                            memcpy(p_dst, p_src, size);
                        }
                    }
                } else {
                    std::vector<concat_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy from single src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy from multiple src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, src_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(concat_parallel_info(i, 0, src_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dim; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const concat_parallel_info& info = infos[j];

                            const eT *p_src = src_list[info.blk_idx] + (i * src_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dim;
                            eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                            const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                            memcpy(p_dst, p_src, size);
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_CONCAT_CONCAT_COMMON_H_
