#ifndef __ST_PPL_KERNEL_X86_COMMON_GEMM_V2_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_GEMM_V2_COMMON_H_

#include "ppl/kernel/x86/common/general_include.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

class gemm_v2_fuse_flag {
public:
    enum {
        none = 0,
        relu = 1 << 0,
    };
};
typedef uint32_t gemm_v2_fuse_flag_t;

class gemm_v2_C_type {
public:
    enum {
        empty    = 0,
        scalar   = 1,
        vector_h = 2,
        vector_w = 3,
        matrix   = 4,
    };
};
typedef uint32_t gemm_v2_C_type_t;

struct gemm_v2_param_fp32 {
    const float* src_A = nullptr;
    const float* src_B = nullptr;
    const float* src_C = nullptr;
    float* dst_Y       = nullptr;
    int32_t trans_A    = 0;
    int32_t trans_B    = 0;
    int32_t M          = 0;
    int32_t N          = 0;
    int32_t K          = 0;
    int32_t lda        = 0;
    int32_t ldb        = 0;
    int32_t ldc        = 0;
    int32_t ldy        = 0;
    float alpha        = 1.0f;
    float beta         = 0.0f;

    ppl::common::isa_t isa_flag   = ppl::common::ISA_undef;
    gemm_v2_fuse_flag_t fuse_flag = gemm_v2_fuse_flag::none;
    gemm_v2_C_type_t c_type       = gemm_v2_C_type::empty;
};

}}} // namespace ppl::kernel::x86

#endif //!__ST_PPL_KERNEL_X86_COMMON_GEMM_V2_COMMON_H_