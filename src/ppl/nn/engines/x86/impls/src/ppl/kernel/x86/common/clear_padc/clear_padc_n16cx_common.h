#ifndef __ST_PPL_KERNEL_X86_COMMON_CLEAR_PADC_N16CX_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_COMMON_CLEAR_PADC_N16CX_FP32_SSE_H_

#include <xmmintrin.h>
#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

template <typename eT>
static void clear_padc_n16cx_common(const ppl::nn::TensorShape *shape, eT *data, const int64_t c_dim_idx = 1)
{
    if (shape->GetDataFormat() == ppl::common::DATAFORMAT_N16CX && shape->GetDim(c_dim_idx) % 16 != 0) { // clear padding channels to 0
        const int64_t c_blk = 16;

        int64_t outer_dims = 1;
        int64_t inner_dims = 1;
        for (int64_t i = 0; i < c_dim_idx; i++) {
            outer_dims *= shape->GetDim(i);
        }
        for (int64_t i = c_dim_idx + 1; i < shape->GetDimCount(); i++) {
            inner_dims *= shape->GetDim(i);
        }
        const int64_t channels = shape->GetDim(c_dim_idx);
        const int64_t pad_c    = round_up(channels, c_blk);

#ifdef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
        for (int64_t od = 0; od < outer_dims; od++) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
            PRAGMA_OMP_PARALLEL_FOR()
#endif
            for (int64_t id = 0; id < inner_dims; id++) {
                eT* p_data = data + od * pad_c * inner_dims + (pad_c - c_blk) * inner_dims + id * c_blk;
                for (int64_t c = 0; c < pad_c - channels; c++) {
                    p_data[c] = 0;
                }
            }
        }
    }
}

}}} // namespace ppl::kernel::x86

#endif