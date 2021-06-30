#ifndef _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_KERNEL_H_
#define _ST_HPC_PPL_NN_SAMPLES_CPP_ENGINE_DEMO_KERNEL_H_

#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/kernel_impl.h"

namespace ppl { namespace nn { namespace demo {

class DemoKernel final : public KernelImpl {
public:
    DemoKernel(const ir::Node* node) : KernelImpl(node) {}
    DemoKernel(DemoKernel&&) = default;

    ppl::common::RetCode Execute(KernelExecContext* ctx) override;
};

class DemoOptKernel final : public OptKernel {
public:
    DemoOptKernel(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override {
        return new DemoKernel(GetNode());
    }
};

}}} // namespace ppl::nn::demo

#endif
