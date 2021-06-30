#ifndef _ST_HPC_PPL_NN_TESTS_ENGINES_ECHO_OP_H_
#define _ST_HPC_PPL_NN_TESTS_ENGINES_ECHO_OP_H_

#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/kernel_impl.h"

namespace ppl { namespace nn { namespace test {

class TestKernel final : public KernelImpl {
public:
    TestKernel(const ir::Node* node) : KernelImpl(node) {}
    TestKernel(TestKernel&&) = default;

    ppl::common::RetCode Execute(KernelExecContext* ctx) override;
};

class TestOptKernel final : public OptKernel {
public:
    TestOptKernel(const ir::Node* node) : OptKernel(node) {}
    KernelImpl* CreateKernelImpl() const override {
        return new TestKernel(GetNode());
    }
};

}}} // namespace ppl::nn::test

#endif
