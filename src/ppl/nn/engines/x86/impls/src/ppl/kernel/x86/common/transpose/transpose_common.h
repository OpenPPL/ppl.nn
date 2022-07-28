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

#ifndef __ST_PPL_KERNEL_X86_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_

#include <vector>
#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/threading_tools.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
ppl::common::RetCode transpose2d_ndarray(
    const eT *src,
    const int64_t *dst_dims,
    const int64_t *map_stride,
    const int64_t *dst_stride,
    eT *dst)
{
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(1)
#endif
    for (int64_t d0 = 0; d0 < dst_dims[0]; ++d0) {
        auto l_src = src
            + d0 * map_stride[0];
        auto l_dst = dst
            + d0 * dst_stride[0];

        int64_t inner_dim  = dst_dims[1];
        int64_t src_stride = map_stride[1];

        if (src_stride == 1) {
            for (int64_t i = 0; i < inner_dim; ++i) {
                l_dst[i] = l_src[i];
            }
        } else {
            for (int64_t i = 0; i < inner_dim; ++i) {
                l_dst[i] = l_src[i * src_stride];
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose3d_ndarray(
    const eT *src,
    const int64_t *dst_dims,
    const int64_t *map_stride,
    const int64_t *dst_stride,
    eT *dst)
{
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t d0 = 0; d0 < dst_dims[0]; ++d0) {
        for (int64_t d1 = 0; d1 < dst_dims[1]; ++d1) {
            auto l_src = src
                + d0 * map_stride[0]
                + d1 * map_stride[1];
            auto l_dst = dst
                + d0 * dst_stride[0]
                + d1 * dst_stride[1];

            int64_t inner_dim  = dst_dims[2];
            int64_t src_stride = map_stride[2];

            if (src_stride == 1) {
                for (int64_t i = 0; i < inner_dim; ++i) {
                    l_dst[i] = l_src[i];
                }
            } else {
                for (int64_t i = 0; i < inner_dim; ++i) {
                    l_dst[i] = l_src[i * src_stride];
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}


template <typename eT>
ppl::common::RetCode transpose4d_ndarray(
    const eT *src,
    const int64_t *dst_dims,
    const int64_t *map_stride,
    const int64_t *dst_stride,
    eT *dst)
{
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#endif
    for (int64_t d0 = 0; d0 < dst_dims[0]; ++d0) {
        for (int64_t d1 = 0; d1 < dst_dims[1]; ++d1) {
            for (int64_t d2 = 0; d2 < dst_dims[2]; ++d2) {
                auto l_src = src
                    + d0 * map_stride[0]
                    + d1 * map_stride[1]
                    + d2 * map_stride[2];
                auto l_dst = dst
                    + d0 * dst_stride[0]
                    + d1 * dst_stride[1]
                    + d2 * dst_stride[2];

                int64_t inner_dim  = dst_dims[3];
                int64_t src_stride = map_stride[3];

                if (src_stride == 1) {
                    for (int64_t i = 0; i < inner_dim; ++i) {
                        l_dst[i] = l_src[i];
                    }
                } else {
                    for (int64_t i = 0; i < inner_dim; ++i) {
                        l_dst[i] = l_src[i * src_stride];
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose5d_ndarray(
    const eT *src,
    const int64_t *dst_dims,
    const int64_t *map_stride,
    const int64_t *dst_stride,
    eT *dst)
{
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(4)
#endif
    for (int64_t d0 = 0; d0 < dst_dims[0]; ++d0) {
        for (int64_t d1 = 0; d1 < dst_dims[1]; ++d1) {
            for (int64_t d2 = 0; d2 < dst_dims[2]; ++d2) {
                for (int64_t d3 = 0; d3 < dst_dims[3]; ++d3) {
                    auto l_src = src
                        + d0 * map_stride[0]
                        + d1 * map_stride[1]
                        + d2 * map_stride[2]
                        + d3 * map_stride[3];
                    auto l_dst = dst
                        + d0 * dst_stride[0]
                        + d1 * dst_stride[1]
                        + d2 * dst_stride[2]
                        + d3 * dst_stride[3];

                    int64_t inner_dim  = dst_dims[4];
                    int64_t src_stride = map_stride[4];

                    if (src_stride == 1) {
                        for (int64_t i = 0; i < inner_dim; ++i) {
                            l_dst[i] = l_src[i];
                        }
                    } else {
                        for (int64_t i = 0; i < inner_dim; ++i) {
                            l_dst[i] = l_src[i * src_stride];
                        }
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose6d_ndarray(
    const eT *src,
    const int64_t *dst_dims,
    const int64_t *map_stride,
    const int64_t *dst_stride,
    eT *dst)
{
#ifndef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR()
#else
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(5)
#endif
    for (int64_t d0 = 0; d0 < dst_dims[0]; ++d0) {
        for (int64_t d1 = 0; d1 < dst_dims[1]; ++d1) {
            for (int64_t d2 = 0; d2 < dst_dims[2]; ++d2) {
                for (int64_t d3 = 0; d3 < dst_dims[3]; ++d3) {
                    for (int64_t d4 = 0; d4 < dst_dims[4]; ++d4) {
                        auto l_src = src
                            + d0 * map_stride[0]
                            + d1 * map_stride[1]
                            + d2 * map_stride[2]
                            + d3 * map_stride[3]
                            + d4 * map_stride[4];
                        auto l_dst = dst
                            + d0 * dst_stride[0]
                            + d1 * dst_stride[1]
                            + d2 * dst_stride[2]
                            + d3 * dst_stride[3]
                            + d4 * dst_stride[4];

                        int64_t inner_dim  = dst_dims[5];
                        int64_t src_stride = map_stride[5];

                        if (src_stride == 1) {
                            for (int64_t i = 0; i < inner_dim; ++i) {
                                l_dst[i] = l_src[i];
                            }
                        } else {
                            for (int64_t i = 0; i < inner_dim; ++i) {
                                l_dst[i] = l_src[i * src_stride];
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
void transpose_ndarray_recursive(
    const single_parallel_loop_config_t &pc,
    const int64_t *dst_dims,
    const int64_t *map_stride,
    const int64_t *dst_stride,
    const uint32_t dim_index,
    const uint32_t dim_count,
    const int64_t src_offset,
    const int64_t dst_offset,
    const eT *src,
    eT *dst)
{
    const int64_t dim_len = dst_dims[dim_index];
    if (dim_index == dim_count - 1) {
        const int64_t src_stride = map_stride[dim_index];
        auto l_dst = dst + dst_offset;
        auto l_src = src + src_offset;
        if (dim_index == pc.depth_of_loop && dim_len > 1) {
            const int64_t thr_len = div_up(dim_len, pc.num_threads);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc.num_threads; ++t) {
                const int64_t beg_idx = t * thr_len;
                const int64_t end_idx = min<int64_t>(beg_idx + thr_len, dim_len);
                if (src_stride == 1) {
                    for (int64_t i = beg_idx; i < end_idx; i++) {
                        l_dst[i] = l_src[i];
                    }
                } else {
                    for (int64_t i = beg_idx; i < end_idx; i++) {
                        l_dst[i] = l_src[i * src_stride];
                    }
                }
            }
        } else {
            if (src_stride == 1) {
                for (int64_t i = 0; i < dim_len; i++) {
                    l_dst[i] = l_src[i];
                }
            } else {
                for (int64_t i = 0; i < dim_len; i++) {
                    l_dst[i] = l_src[i * src_stride];
                }
            }
        }
    } else {
        if (dim_index == pc.depth_of_loop && dim_len > 1) {
            const int64_t thr_len = div_up(dim_len, pc.num_threads);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc.num_threads; ++t) {
                const int64_t beg_idx = t * thr_len;
                const int64_t end_idx = min<int64_t>(beg_idx + thr_len, dim_len);
                for (int64_t i = beg_idx; i < end_idx; i++) {
                    transpose_ndarray_recursive<eT>(
                        pc, dst_dims, map_stride, dst_stride,
                        dim_index + 1, dim_count,
                        src_offset + i * map_stride[dim_index],
                        dst_offset + i * dst_stride[dim_index],
                        src, dst);
                }
            }
        } else {
            for (int64_t i = 0; i < dim_len; i++) {
                transpose_ndarray_recursive<eT>(
                        pc, dst_dims, map_stride, dst_stride,
                        dim_index + 1, dim_count,
                        src_offset + i * map_stride[dim_index],
                        dst_offset + i * dst_stride[dim_index],
                        src, dst);
            }
        }
    }
}

template <typename eT>
ppl::common::RetCode transpose_ndarray(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int32_t *perm,
    const eT *src,
    eT *dst)
{
    const int64_t dim_count = src_shape->GetDimCount();

    if (dim_count <= 1) {
        return ppl::common::RC_SUCCESS;
    }

    std::vector<int32_t> inv_perm(dim_count);
    std::vector<int64_t> dst_dims(dim_count);
    std::vector<int64_t> dst_stride(dim_count);
    std::vector<int64_t> src_dims(dim_count);
    std::vector<int64_t> src_stride(dim_count);
    std::vector<int64_t> map_stride(dim_count);

    for (int64_t i = 0; i < dim_count; ++i) {
        auto perm_val = perm[i];
        if (perm_val < 0) perm_val += dim_count;
        inv_perm[perm_val] = i;
        src_dims[i] = src_shape->GetDim(i);
        dst_dims[i] = src_shape->GetDim(perm_val);
    }

    src_stride[dim_count - 1] = 1;
    dst_stride[dim_count - 1] = 1;
    for (int64_t i = dim_count - 2; i >= 0; i--) {
        src_stride[i] = src_stride[i + 1] * src_dims[i + 1];
        dst_stride[i] = dst_stride[i + 1] * dst_dims[i + 1];
    }

    for (int64_t i = 0; i < dim_count; ++i) {
        map_stride[inv_perm[i]] = src_stride[i];
    }

#ifndef PPL_USE_X86_OMP_COLLAPSE
    const bool enable_omp_collapse = false;
#else
    const bool enable_omp_collapse = true;
#endif

    if (dim_count <= 6 && (enable_omp_collapse || dst_dims[0] > PPL_OMP_MAX_THREADS())) {
        auto transpose_func = transpose2d_ndarray<eT>;
        if (dim_count == 3) transpose_func = transpose3d_ndarray<eT>;
        if (dim_count == 4) transpose_func = transpose4d_ndarray<eT>;
        if (dim_count == 5) transpose_func = transpose5d_ndarray<eT>;
        if (dim_count == 6) transpose_func = transpose6d_ndarray<eT>;

        return transpose_func(
            src, dst_dims.data(), map_stride.data(),
            dst_stride.data(), dst);
    }

    auto pc = select_single_parallel_loop(
        dst_dims, ppl::common::ISA_UNKNOWN,
        sizeof(eT), sizeof(eT), sizeof(eT), 1);

    transpose_ndarray_recursive(
        pc, dst_dims.data(), map_stride.data(), dst_stride.data(),
        0, dim_count, 0, 0, src, dst);

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
ppl::common::RetCode transpose_ndarray_continous2d(
    const ppl::nn::TensorShape *src_shape,
    const uint32_t axis0,
    const uint32_t axis1,
    const eT *src,
    eT *dst)
{
    const uint32_t dim_count = src_shape->GetDimCount();
    const int64_t dim0       = src_shape->GetDim(axis0);
    const int64_t dim1       = src_shape->GetDim(axis1);

    int64_t outer_dims = 1;
    for (uint32_t i = 0; i < axis0; i++) {
        outer_dims *= src_shape->GetDim(i);
    }

    int64_t inner_dims = 1;
    for (uint32_t i = axis1 + 1; i < dim_count; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t od = 0; od < outer_dims; od++) {
        for (int64_t j = 0; j < dim1; j++) {
            for (int64_t i = 0; i < dim0; i++) {
                auto l_src = src + od * dim0 * dim1 * inner_dims + i * dim1 * inner_dims + j * inner_dims;
                auto l_dst = dst + od * dim1 * dim0 * inner_dims + j * dim0 * inner_dims + i * inner_dims;
                memcpy(l_dst, l_src, inner_dims * sizeof(eT));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_COMMON_TRANSPOSE_TRANSPOSE_COMMON_H_
