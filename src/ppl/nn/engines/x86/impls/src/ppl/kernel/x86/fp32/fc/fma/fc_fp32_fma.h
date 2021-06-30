#ifndef __ST_PPL_KERNEL_X86_FP32_FC_FMA_FC_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_FC_FMA_FC_FP32_FMA_H_

#include "ppl/kernel/x86/fp32/fc.h"
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/timer.h"

namespace ppl { namespace kernel { namespace x86 {

// forward declare;
class fc_fp32_fma_manager;

class fc_fp32_fma_executor final : public fc_fp32_executor {
public:
    fc_fp32_fma_executor() {}
    fc_fp32_fma_executor(const fc_fp32_param *fc_param, const float *cvt_filter, const float *bias)
        : fc_fp32_executor(fc_param, cvt_filter, bias) {}
    uint64_t cal_temp_buffer_size() override;
    ppl::common::RetCode prepare() override;
    ppl::common::RetCode execute() override;

private:
    struct kernel_schedule_param {
        int32_t ic_l2_blk;
        int32_t ic_l2_cnt;
        int32_t oc_l2_blk;
        int32_t multi_batch;
        int32_t unaligned_oc;
    } schedule_param_;

    void cal_kernel_tunning_param();
    static int32_t cal_ic_l2_blk(const fc_fp32_param &param);

    friend fc_fp32_fma_manager;
};

class fc_fp32_fma_manager final : public fc_fp32_manager {
public:
    fc_fp32_fma_manager() {}
    fc_fp32_fma_manager(const fc_fp32_param &param, ppl::common::Allocator *allocator)
        : fc_fp32_manager(param, allocator) {}
    ppl::common::RetCode gen_cvt_weights(const float *filter, const float *bias) override;
    fc_fp32_executor *gen_executor() override;
};

}}}; // namespace ppl::kernel::x86

#endif
