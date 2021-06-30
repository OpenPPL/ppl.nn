#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_FMA_CONV2D_IM2COL_GEMM_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_GEMM_DIRECT_FMA_CONV2D_IM2COL_GEMM_FP32_FMA_H_

#include "ppl/kernel/x86/fp32/conv2d.h"
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/timer.h"

namespace ppl { namespace kernel { namespace x86 {

// forward declare;
class conv2d_im2col_gemm_fp32_fma_manager;

class conv2d_im2col_gemm_fp32_fma_executor final : public conv2d_fp32_executor {
public:
    conv2d_im2col_gemm_fp32_fma_executor() {}
    conv2d_im2col_gemm_fp32_fma_executor(const conv2d_fp32_param *conv_param, const float *cvt_filter, const float *bias)
        : conv2d_fp32_executor(conv_param, cvt_filter, bias) {}
    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

private:
    struct kernel_schedule_param {
        // Preprocessed param
        int32_t ic_per_gp;
        int32_t oc_per_gp;
        int32_t k_per_gp;
        int64_t padded_k;

        // Kernel tunning
        int32_t hw_l2_blk;
        int32_t mb_l3_blk;
        int32_t gp_l3_blk;
        int32_t use_nt_store;
    } schedule_param_;

    void init_preproc_param();
    void cal_kernel_tunning_param();

    static int32_t cal_ic_l2_blk(const conv2d_fp32_param &param);

    friend conv2d_im2col_gemm_fp32_fma_manager;
};

class conv2d_im2col_gemm_fp32_fma_manager final : public conv2d_fp32_manager {
public:
    conv2d_im2col_gemm_fp32_fma_manager() {}
    conv2d_im2col_gemm_fp32_fma_manager(const conv2d_fp32_param &param, ppl::common::Allocator *allocator)
        : conv2d_fp32_manager(param, allocator) {}
    bool is_supported() override;
    ppl::common::RetCode gen_cvt_weights(const float *filter, const float *bias) override;
    conv2d_fp32_executor *gen_executor() override;
};

}}}; // namespace ppl::kernel::x86

#endif
