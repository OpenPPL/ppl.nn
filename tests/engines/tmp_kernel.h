#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/opt_kernel.h"

namespace ppl { namespace nn { namespace test {

class TmpKernelOne final : public KernelImpl {
public:
    TmpKernelOne(const ir::Node* node) : KernelImpl(node) {}
    ppl::common::RetCode Execute(KernelExecContext*) override {
        return ppl::common::RC_SUCCESS;
    };
};

class TmpKernelTwo final : public KernelImpl {
public:
    TmpKernelTwo(const ir::Node* node) : KernelImpl(node) {}
    ppl::common::RetCode Execute(KernelExecContext*) override {
        return ppl::common::RC_SUCCESS;
    };
};

class TmpOptKernelOne : public OptKernel {
public:
    TmpOptKernelOne(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override {
        return new TmpKernelOne(GetNode());
    }
};

class TmpOptKernelTwo : public OptKernel {
public:
    TmpOptKernelTwo(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override {
        return new TmpKernelTwo(GetNode());
    }
};

}}} // namespace ppl::nn::test
