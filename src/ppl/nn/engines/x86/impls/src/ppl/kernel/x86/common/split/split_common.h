#ifndef __ST_PPL_KERNEL_X86_COMMON_SPLIT_SPLIT_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_SPLIT_SPLIT_COMMON_H_

#include <vector>
#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

struct split_parallel_info {
    split_parallel_info() {}
    split_parallel_info(
        int32_t blk_idx,
        int32_t start,
        int32_t end)
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
ppl::common::RetCode split_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const eT *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    eT **dst_list)
{
    const int32_t ndims         = src_shape->GetDimCount();
    const int32_t fixed_axis    = slice_axis < 0 ? slice_axis + ndims : slice_axis;
    const int64_t src_split_dim = src_shape->GetDim(fixed_axis);

    int64_t outer_dims = 1;
    int64_t inner_dims = 1;
    for (int32_t i = 0; i < fixed_axis; i++) {
        outer_dims *= src_shape->GetDim(i);
    }
    for (int32_t i = fixed_axis + 1; i < ndims; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    std::vector<int64_t> src_offset;
    src_offset.resize(num_dst);
    src_offset[0] = 0;
    for (int32_t i = 1; i < num_dst; i++) {
        src_offset[i] = src_offset[i - 1] + dst_shape_list[i - 1]->GetDim(fixed_axis);
    }

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (outer_dims * num_dst >= num_threads) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int32_t i = 0; i < outer_dims; i++) {
            for (int32_t n = 0; n < num_dst; n++) {
                const eT *p_src = src + i * src_split_dim * inner_dims + src_offset[n] * inner_dims;
                eT *p_dst       = dst_list[n] + i * dst_shape_list[n]->GetDim(fixed_axis) * inner_dims;

                const size_t size = dst_shape_list[n]->GetDim(fixed_axis) * inner_dims * sizeof(eT);
                memcpy(p_dst, p_src, size);
            }
        }
    } else { // parallel along axis, may cross different dst
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), src_split_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t split_dim_per_thread = div_up(src_split_dim, num_threads);
            const int64_t start_split_dim      = split_dim_per_thread * thread_id;
            const int64_t end_split_dim        = min(start_split_dim + split_dim_per_thread, src_split_dim);

            if (start_split_dim < end_split_dim) {
                int64_t start_blk_idx = num_dst - 1;
                int64_t end_blk_idx   = num_dst - 1;
                for (int64_t i = 0; i < num_dst - 1; i++) {
                    if (start_split_dim >= src_offset[i] && start_split_dim < src_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_split_dim >= src_offset[i] && end_split_dim < src_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_split_dim - src_offset[start_blk_idx];
                int64_t end_axis_idx   = end_split_dim - src_offset[end_blk_idx];

                std::vector<split_parallel_info> infos;
                if (start_blk_idx == end_blk_idx) { // copy to single dst
                    infos.emplace_back(split_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                } else { // start_blk_idx < end_blk_idx, copy to multiple dst
                    infos.emplace_back(split_parallel_info(start_blk_idx, start_axis_idx, dst_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                    for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                        infos.emplace_back(split_parallel_info(i, 0, dst_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                    }
                    infos.emplace_back(split_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                }

                for (int64_t i = 0; i < outer_dims; i++) {
                    for (uint64_t j = 0; j < infos.size(); j++) {
                        const split_parallel_info& info = infos[j];

                        const eT *p_src = src + (i * src_split_dim + src_offset[info.blk_idx] + info.start) * inner_dims;
                        eT *p_dst       = dst_list[info.blk_idx] + (i * dst_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dims;

                        size_t size = (info.end - info.start) * inner_dims * sizeof(eT);
                        memcpy(p_dst, p_src, size);
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}


template <typename eT>
ppl::common::RetCode split_n16cx_interleave_channels(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const eT *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    const int32_t c_dim_idx,
    eT **dst_list)
{
    const int32_t ndims = src_shape->GetDimCount();

    int64_t outer_dims = 1;
    int64_t inner_dims = 1;
    for (int32_t i = 0; i < c_dim_idx; i++) {
        outer_dims *= src_shape->GetDim(i);
    }
    for (int32_t i = c_dim_idx + 1; i < ndims; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    std::vector<int64_t> src_offset;
    src_offset.resize(num_dst);
    src_offset[0] = 0;
    for (int32_t i = 1; i < num_dst; i++) {
        src_offset[i] = src_offset[i - 1] + dst_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t c_blk        = 16;
    const int64_t src_channels = src_shape->GetDim(c_dim_idx);
    const int64_t padded_ic    = round_up(src_channels, c_blk);

    const int64_t num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), inner_dims);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        const int64_t inner_dims_per_thread = div_up(inner_dims, num_threads);
        const int64_t start_inner_dims      = inner_dims_per_thread * thread_id;
        const int64_t end_inner_dims        = min(start_inner_dims + inner_dims_per_thread, inner_dims);

        if (start_inner_dims < end_inner_dims) {
            for (int64_t i = 0; i < outer_dims; i++) {
                for (int32_t n = 0; n < num_dst; n++) {
                    const int32_t dst_channels = dst_shape_list[n]->GetDim(c_dim_idx);
                    const int32_t padded_oc    = round_up(dst_channels, c_blk);
                    for (int32_t oc = 0; oc < padded_oc; oc += c_blk) {
                        const int32_t ic = src_offset[n] + oc;
                        const eT *p_src   = src + i * padded_ic * inner_dims + round(ic, c_blk) * inner_dims;
                        eT *p_dst         = dst_list[n] + i * padded_oc * inner_dims + oc * inner_dims;
                        if (ic % c_blk == 0) { // no interleave on this 16c
                            memcpy(p_dst + start_inner_dims * c_blk, p_src + start_inner_dims * c_blk, (end_inner_dims - start_inner_dims) * c_blk * sizeof(eT));
                        } else { // has interleave on this 16c
                            const int32_t c_offset  = c_blk - (ic % c_blk);
                            const int32_t c_end     = min(dst_channels - oc, (int32_t)c_blk);
                            const eT *p_src_next_16c = p_src + c_blk * inner_dims;

                            if (oc + c_blk == padded_oc && dst_channels < padded_oc) { // last 16c need to pad 0
                                for (int64_t id = start_inner_dims; id < end_inner_dims; id++) {
                                    // interleave copy
                                    for (int32_t c = 0; c < c_offset; c++) {
                                        p_dst[id * c_blk + c] = p_src[id * c_blk + c_blk - c_offset + c];
                                    }
                                    for (int32_t c = c_offset; c < c_end; c++) {
                                        p_dst[id * c_blk + c] = p_src_next_16c[id * c_blk + c - c_offset];
                                    }
                                }
                            } else {
                                for (int64_t id = start_inner_dims; id < end_inner_dims; id++) {
                                    // interleave copy
                                    for (int32_t c = 0; c < c_offset; c++) {
                                        p_dst[id * c_blk + c] = p_src[id * c_blk + c_blk - c_offset + c];
                                    }
                                    for (int32_t c = c_offset; c < c_end; c++) {
                                        p_dst[id * c_blk + c] = p_src_next_16c[id * c_blk + c - c_offset];
                                    }
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
ppl::common::RetCode split_n16cx(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape **dst_shape_list,
    const eT *src,
    const int32_t slice_axis,
    const int32_t num_dst,
    eT **dst_list)
{
    const int32_t ndims      = src_shape->GetDimCount();
    const int32_t fixed_axis = slice_axis < 0 ? slice_axis + ndims : slice_axis;
    const int64_t c_dim_idx  = 1;
    const int64_t c_blk      = 16;

    if (fixed_axis == 1) {
        for (int32_t i = 0; i < num_dst - 1; i++) {
            if (dst_shape_list[i]->GetDim(c_dim_idx) % c_blk != 0) {
                return split_n16cx_interleave_channels<eT>(src_shape, dst_shape_list, src, slice_axis, num_dst, c_dim_idx, dst_list);
            }
        }
    }

    int64_t outer_dims = 1;
    int64_t inner_dims = c_blk;
    for (int32_t i = 0; i < fixed_axis; i++) {
        if (i == c_dim_idx) {
            outer_dims *= div_up(src_shape->GetDim(i), c_blk);
        } else {
            outer_dims *= src_shape->GetDim(i);
        }
    }
    for (int32_t i = fixed_axis + 1; i < ndims; i++) {
        if (i == c_dim_idx) {
            inner_dims *= div_up(src_shape->GetDim(i), c_blk);
        } else {
            inner_dims *= src_shape->GetDim(i);
        }
    }

    std::vector<int64_t> src_offset;
    src_offset.resize(num_dst);
    src_offset[0]         = 0;
    int64_t src_split_dim = 0;
    if (fixed_axis == c_dim_idx) {
        for (int32_t i = 1; i < num_dst; i++) {
            src_offset[i] = src_offset[i - 1] + div_up(dst_shape_list[i - 1]->GetDim(fixed_axis), c_blk);
        }
        src_split_dim = div_up(src_shape->GetDim(fixed_axis), c_blk);
    } else {
        for (int32_t i = 1; i < num_dst; i++) {
            src_offset[i] = src_offset[i - 1] + dst_shape_list[i - 1]->GetDim(fixed_axis);
        }
        src_split_dim = src_shape->GetDim(fixed_axis);
    }

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (outer_dims * num_dst >= num_threads) {
        if (fixed_axis == c_dim_idx) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int32_t i = 0; i < outer_dims; i++) {
                for (int32_t n = 0; n < num_dst; n++) {
                    const eT *p_src = src + i * src_split_dim * inner_dims + src_offset[n] * inner_dims;
                    eT *p_dst       = dst_list[n] + i * div_up(dst_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dims;

                    const size_t size = div_up(dst_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dims * sizeof(eT);
                    memcpy(p_dst, p_src, size);
                }
            }
        } else {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int32_t i = 0; i < outer_dims; i++) {
                for (int32_t n = 0; n < num_dst; n++) {
                    const eT *p_src = src + i * src_split_dim * inner_dims + src_offset[n] * inner_dims;
                    eT *p_dst       = dst_list[n] + i * dst_shape_list[n]->GetDim(fixed_axis) * inner_dims;

                    const size_t size = dst_shape_list[n]->GetDim(fixed_axis) * inner_dims * sizeof(eT);
                    memcpy(p_dst, p_src, size);
                }
            }
        }
    } else { // parallel along axis, may cross different dst
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), src_split_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t split_dim_per_thread = div_up(src_split_dim, num_threads);
            const int64_t start_split_dim      = split_dim_per_thread * thread_id;
            const int64_t end_split_dim        = min(start_split_dim + split_dim_per_thread, src_split_dim);

            if (start_split_dim < end_split_dim) {
                int64_t start_blk_idx = num_dst - 1;
                int64_t end_blk_idx   = num_dst - 1;
                for (int64_t i = 0; i < num_dst - 1; i++) {
                    if (start_split_dim >= src_offset[i] && start_split_dim < src_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_split_dim >= src_offset[i] && end_split_dim < src_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_split_dim - src_offset[start_blk_idx];
                int64_t end_axis_idx   = end_split_dim - src_offset[end_blk_idx];

                if (fixed_axis == c_dim_idx) {
                    std::vector<split_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy to single dst
                        infos.emplace_back(split_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy to multiple dst
                        infos.emplace_back(split_parallel_info(start_blk_idx, start_axis_idx, div_up(dst_shape_list[start_blk_idx]->GetDim(fixed_axis), c_blk))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(split_parallel_info(i, 0, div_up(dst_shape_list[i]->GetDim(fixed_axis), c_blk))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(split_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dims; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const split_parallel_info& info = infos[j];
                            const eT *p_src                  = src + (i * src_split_dim + src_offset[info.blk_idx] + info.start) * inner_dims;
                            eT *p_dst                        = dst_list[info.blk_idx] + (i * div_up(dst_shape_list[info.blk_idx]->GetDim(fixed_axis), c_blk) + info.start) * inner_dims;

                            size_t size = (info.end - info.start) * inner_dims * sizeof(eT);
                            memcpy(p_dst, p_src, size);
                        }
                    }
                } else {
                    std::vector<split_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy to single dst
                        infos.emplace_back(split_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy to multiple dst
                        infos.emplace_back(split_parallel_info(start_blk_idx, start_axis_idx, dst_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(split_parallel_info(i, 0, dst_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(split_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dims; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const split_parallel_info& info = infos[j];
                            const eT *p_src                  = src + (i * src_split_dim + src_offset[info.blk_idx] + info.start) * inner_dims;
                            eT *p_dst                        = dst_list[info.blk_idx] + (i * dst_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dims;

                            size_t size = (info.end - info.start) * inner_dims * sizeof(eT);
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

#endif // __ST_PPL_KERNEL_X86_COMMON_SPLIT_SPLIT_COMMON_H_
