#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_FMA_CONV2D_N16CX_DIRECT_NDARRAY_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DIRECT_NDARRAY_FMA_CONV2D_N16CX_DIRECT_NDARRAY_FP32_FMA_H_

#include "ppl/kernel/x86/fp32/conv2d.h"
#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

// forward declare;
class conv2d_n16cx_direct_ndarray_fp32_fma_manager;

class conv2d_n16cx_direct_ndarray_fp32_fma_executor final : public conv2d_fp32_executor {
public:
    conv2d_n16cx_direct_ndarray_fp32_fma_executor() {}
    conv2d_n16cx_direct_ndarray_fp32_fma_executor(const conv2d_fp32_param *conv_param, const float *cvt_filter, const float *bias)
        : conv2d_fp32_executor(conv_param, cvt_filter, bias) {}
    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

private:
    struct kernel_schedule_param {
        // Preprocessed param
        int32_t ic_per_gp;
        int32_t oc_per_gp;
        int32_t padded_oc;

        // Kernel tunning
        int32_t oc_l2_blk;
        int32_t use_nt_store;
        int32_t unroll_ow_start;
        int32_t unroll_ow_end;
    } schedule_param_;

    void init_preproc_param();
    void cal_kernel_tunning_param();

    template <bool use_nt_store>
    ppl::common::RetCode execute_inner();

    friend conv2d_n16cx_direct_ndarray_fp32_fma_manager;
};

class conv2d_n16cx_direct_ndarray_fp32_fma_manager final : public conv2d_fp32_manager {
public:
    conv2d_n16cx_direct_ndarray_fp32_fma_manager() {}
    conv2d_n16cx_direct_ndarray_fp32_fma_manager(const conv2d_fp32_param &param, ppl::common::Allocator *allocator)
        : conv2d_fp32_manager(param, allocator) {}
    bool is_supported() override;
    ppl::common::RetCode gen_cvt_weights(const float *filter, const float *bias) override;
    conv2d_fp32_executor *gen_executor() override;
};

}}}; // namespace ppl::kernel::x86

#endif
