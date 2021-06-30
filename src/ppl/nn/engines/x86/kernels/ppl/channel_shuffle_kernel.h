#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_PPL_CHANNEL_SHUFFLE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_PPL_CHANNEL_SHUFFLE_KERNEL_H_

#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/nn/params/ppl/channel_shuffle_param.h"

namespace ppl { namespace nn { namespace x86 {

class ChannelShuffleKernel : public X86Kernel {
public:
    ChannelShuffleKernel(const ir::Node* node) : X86Kernel(node) {}

    void SetParam(const ppl::nn::common::ChannelShuffleParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ChannelShuffleParam* param_ = nullptr;
};

}}} // namespace ppl::nn::x86

#endif
