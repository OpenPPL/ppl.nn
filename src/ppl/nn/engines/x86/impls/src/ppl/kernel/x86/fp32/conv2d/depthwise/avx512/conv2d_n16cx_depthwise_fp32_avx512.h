#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_AVX512_CONV2D_N16CX_DEPTHWISE_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_DEPTHWISE_AVX512_CONV2D_N16CX_DEPTHWISE_FP32_AVX512_H_

#include "ppl/kernel/x86/fp32/conv2d.h"
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/timer.h"

namespace ppl { namespace kernel { namespace x86 {

// forward declare;
class conv2d_n16cx_depthwise_fp32_avx512_manager;

class conv2d_n16cx_depthwise_fp32_avx512_executor final : public conv2d_fp32_executor {
public:
    conv2d_n16cx_depthwise_fp32_avx512_executor() {}
    conv2d_n16cx_depthwise_fp32_avx512_executor(const conv2d_fp32_param *conv_param, const float *cvt_filter, const float *bias)
        : conv2d_fp32_executor(conv_param, cvt_filter, bias) {}
    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

private:
    struct kernel_schedule_param {
        // Preprocessed param
        int32_t padded_ch;

        // Kernel tunning
        int32_t ow_kr_blk;
        int32_t use_nt_store;

        int32_t unroll_ow_start;
        int32_t unroll_ow_end;

        int32_t padding_policy;
    } schedule_param_;

    void init_preproc_param();
    void cal_kernel_tunning_param();

    friend conv2d_n16cx_depthwise_fp32_avx512_manager;
};

class conv2d_n16cx_depthwise_fp32_avx512_manager final : public conv2d_fp32_manager {
public:
    conv2d_n16cx_depthwise_fp32_avx512_manager() {}
    conv2d_n16cx_depthwise_fp32_avx512_manager(const conv2d_fp32_param &param, ppl::common::Allocator *allocator)
        : conv2d_fp32_manager(param, allocator) {}
    bool is_supported() override;
    ppl::common::RetCode gen_cvt_weights(const float *filter, const float *bias) override;
    conv2d_fp32_executor *gen_executor() override;
};

}}}; // namespace ppl::kernel::x86

#endif
