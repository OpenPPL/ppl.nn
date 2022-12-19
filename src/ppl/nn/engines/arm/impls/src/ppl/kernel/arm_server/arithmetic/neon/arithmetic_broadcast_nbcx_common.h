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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_BROADCAST_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_BROADCAST_NBCX_COMMON_H_

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/threading_tools.h"
#include "ppl/kernel/arm_server/arithmetic/neon/arithmetic_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, int32_t c_blk, arithmetic_op_type_t op_type, bool fuse_relu>
inline void arithmetic_broadcast_nbcx_lastdim_no_broadcast_common(
    const eT *src0,
    const eT *src1,
    const int64_t length,
    const bool c0_broadcast,
    const bool c1_broadcast,
    const bool parallel,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    const vecType v_zero = vdup_n<eT, eN>(0);

    if (!c0_broadcast && !c1_broadcast) {
        if (parallel) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    } else if (c0_broadcast) {
        if (parallel) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vdup_n<eT, eN>(src0[i * c_blk]);
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vdup_n<eT, eN>(src0[i * c_blk]);
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    } else {
        if (parallel) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_src1 = vdup_n<eT, eN>(src1[i * c_blk]);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_src1 = vdup_n<eT, eN>(src1[i * c_blk]);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    }
}

template <typename eT, int32_t c_blk, arithmetic_op_type_t op_type, bool fuse_relu>
inline void arithmetic_broadcast_nbcx_lastdim_broadcast0_common(
    const eT *src0,
    const eT *src1,
    const int64_t length,
    const bool c0_broadcast,
    const bool c1_broadcast,
    const bool parallel,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    const vecType v_zero = vdup_n<eT, eN>(0);

    if (!c0_broadcast && !c1_broadcast) {
        if (parallel) {
            vecType v_src0 = vld<eT, eN>(src0);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            vecType v_src0 = vld<eT, eN>(src0);
            for (int64_t i = 0; i < length; i++) {
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    } else if (c0_broadcast) {
        if (parallel) {
            vecType v_src0 = vdup_n<eT, eN>(src0[0]);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            vecType v_src0 = vdup_n<eT, eN>(src0[0]);
            for (int64_t i = 0; i < length; i++) {
                vecType v_src1 = vld<eT, eN>(src1 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    } else {
        if (parallel) {
            vecType v_src0 = vld<eT, eN>(src0);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src1 = vdup_n<eT, eN>(src1[i * c_blk]);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            vecType v_src0 = vld<eT, eN>(src0);
            for (int64_t i = 0; i < length; i++) {
                vecType v_src1 = vdup_n<eT, eN>(src1[i * c_blk]);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    }
}

template <typename eT, int32_t c_blk, arithmetic_op_type_t op_type, bool fuse_relu>
inline void arithmetic_broadcast_nbcx_lastdim_broadcast1_common(
    const eT *src0,
    const eT *src1,
    const int64_t length,
    const bool c0_broadcast,
    const bool c1_broadcast,
    const bool parallel,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    const vecType v_zero = vdup_n<eT, eN>(0);

    if (!c0_broadcast && !c1_broadcast) {
        if (parallel) {
            vecType v_src1 = vld<eT, eN>(src1);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            vecType v_src1 = vld<eT, eN>(src1);
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    } else if (c0_broadcast) {
        if (parallel) {
            vecType v_src1 = vld<eT, eN>(src1);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vdup_n<eT, eN>(src0[i * c_blk]);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            vecType v_src1 = vld<eT, eN>(src1);
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vdup_n<eT, eN>(src0[i * c_blk]);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    } else {
        if (parallel) {
            vecType v_src1 = vdup_n<eT, eN>(src1[0]);
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        } else {
            vecType v_src1 = vdup_n<eT, eN>(src1[0]);
            for (int64_t i = 0; i < length; i++) {
                vecType v_src0 = vld<eT, eN>(src0 + i * c_blk);
                vecType v_dst  = arithmetic_vector_kernel<vecType, op_type>(v_src0, v_src1);
                if (fuse_relu) {
                    v_dst = vmax(v_dst, v_zero);
                }
                vst<eT, eN>(dst + i * c_blk, v_dst);
            }
        }
    }
}

template <typename eT, int32_t c_blk, arithmetic_op_type_t op_type, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_nbcx_recursive_common(
    const int64_t *src0_shape,
    const int64_t *src1_shape,
    const int64_t *dst_shape,
    const eT *src0,
    const eT *src1,
    const int64_t *inc0,
    const int64_t *inc1,
    const int64_t *inc_out,
    const int64_t dim_count,
    const int64_t dim_idx,
    const int64_t c_dim_idx,
    const single_parallel_loop_config_t *loop_config,
    eT *dst)
{
    const int64_t length = dim_idx == c_dim_idx ? div_up(dst_shape[dim_idx], c_blk) : dst_shape[dim_idx];

    if (dim_idx == dim_count - 1) { // last dim
        const bool c0_broadcast     = src0_shape[c_dim_idx] != src1_shape[c_dim_idx] && src0_shape[c_dim_idx] == 1;
        const bool c1_broadcast     = src0_shape[c_dim_idx] != src1_shape[c_dim_idx] && src1_shape[c_dim_idx] == 1;
        const bool lastdim_parallel = dim_idx == loop_config->depth_of_loop && loop_config->num_threads > 1;
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            arithmetic_broadcast_nbcx_lastdim_no_broadcast_common<eT, c_blk, op_type, fuse_relu>(src0, src1, length, c0_broadcast, c1_broadcast, lastdim_parallel, dst);
        } else if (src0_shape[dim_idx] == 1) {
            arithmetic_broadcast_nbcx_lastdim_broadcast0_common<eT, c_blk, op_type, fuse_relu>(src0, src1, length, c0_broadcast, c1_broadcast, lastdim_parallel, dst);
        } else {
            arithmetic_broadcast_nbcx_lastdim_broadcast1_common<eT, c_blk, op_type, fuse_relu>(src0, src1, length, c0_broadcast, c1_broadcast, lastdim_parallel, dst);
        }
    } else {
        if (dim_idx == loop_config->depth_of_loop && loop_config->num_threads > 1) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                arithmetic_broadcast_nbcx_recursive_common<eT, c_blk, op_type, fuse_relu>(
                    src0_shape,
                    src1_shape,
                    dst_shape,
                    src0 + i * inc0[dim_idx],
                    src1 + i * inc1[dim_idx],
                    inc0,
                    inc1,
                    inc_out,
                    dim_count,
                    dim_idx + 1,
                    c_dim_idx,
                    loop_config,
                    dst + i * inc_out[dim_idx]);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                arithmetic_broadcast_nbcx_recursive_common<eT, c_blk, op_type, fuse_relu>(
                    src0_shape,
                    src1_shape,
                    dst_shape,
                    src0 + i * inc0[dim_idx],
                    src1 + i * inc1[dim_idx],
                    inc0,
                    inc1,
                    inc_out,
                    dim_count,
                    dim_idx + 1,
                    c_dim_idx,
                    loop_config,
                    dst + i * inc_out[dim_idx]);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <int32_t c_blk>
inline void arithmetic_nbcx_prepare_incs(
    const int64_t *src0_shape,
    const int64_t *src1_shape,
    const int64_t *dst_shape,
    const int64_t dim_count,
    const int64_t c_dim_idx,
    int64_t *inc0,
    int64_t *inc1,
    int64_t *inc_out)
{
    int64_t stride0    = c_blk;
    int64_t stride1    = c_blk;
    int64_t stride_out = c_blk;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        inc0[i]    = src0_shape[i] == 1 ? 0 : stride0;
        inc1[i]    = src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= i == c_dim_idx ? div_up(src0_shape[i], c_blk) : src0_shape[i];
        stride1 *= i == c_dim_idx ? div_up(src1_shape[i], c_blk) : src1_shape[i];
        stride_out *= i == c_dim_idx ? div_up(dst_shape[i], c_blk) : dst_shape[i];
    }
}

template <typename eT, int64_t c_blk, arithmetic_op_type_t op_type, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_nbcx_common(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    const int64_t c_dim_idx = 1;

    const int64_t max_dim_count              = dst_shape->GetDimCount();
    std::vector<int64_t> padded_src0_shape(max_dim_count);
    std::vector<int64_t> padded_src1_shape(max_dim_count);
    arithmetic_pad_shape(src0_shape, max_dim_count, padded_src0_shape.data());
    arithmetic_pad_shape(src1_shape, max_dim_count, padded_src1_shape.data());

    int64_t compressed_dim_count                 = 0;
    std::vector<int64_t> compressed_src0_shape(max_dim_count);
    std::vector<int64_t> compressed_src1_shape(max_dim_count);
    std::vector<int64_t> compressed_dst_shape(max_dim_count);
    arithmetic_compress_shape(padded_src0_shape.data(), padded_src1_shape.data(), max_dim_count, &compressed_dim_count, compressed_src0_shape.data(), compressed_src1_shape.data(), compressed_dst_shape.data(), c_dim_idx);

    std::vector<int64_t> inc0(compressed_dim_count);
    std::vector<int64_t> inc1(compressed_dim_count);
    std::vector<int64_t> inc_out(compressed_dim_count);
    arithmetic_nbcx_prepare_incs<c_blk>(compressed_src0_shape.data(), compressed_src1_shape.data(), compressed_dst_shape.data(), compressed_dim_count, c_dim_idx, inc0.data(), inc1.data(), inc_out.data());

    const float omp_div_task_time_ratio       = 20.0f; // assume omp create thread may be 20x slower than one element arthimetic process
    single_parallel_loop_config_t loop_config = select_single_parallel_loop(compressed_dst_shape.data(), compressed_dim_count, omp_div_task_time_ratio);

    return arithmetic_broadcast_nbcx_recursive_common<eT, c_blk, op_type, fuse_relu>(
        compressed_src0_shape.data(),
        compressed_src1_shape.data(),
        compressed_dst_shape.data(),
        src0,
        src1,
        inc0.data(),
        inc1.data(),
        inc_out.data(),
        compressed_dim_count,
        0,
        c_dim_idx,
        &loop_config,
        dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_BROADCAST_NBCX_COMMON_H_
