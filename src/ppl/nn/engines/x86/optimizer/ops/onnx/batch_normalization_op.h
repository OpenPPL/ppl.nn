#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_BATCH_NORMALIZATION_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_BATCH_NORMALIZATION_OP_H_

#include "ppl/nn/params/onnx/batch_normalization_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class BatchNormalizationOp final : public X86OptKernel {
public:
    BatchNormalizationOp(const ir::Node* node) : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    KernelImpl* CreateKernelImpl() const override;
    void SetFuseReLU(bool fuse_relu) {
        fuse_relu_ = fuse_relu;
    }

private:
    std::shared_ptr<ppl::nn::common::BatchNormalizationParam> param_;
    bool fuse_relu_ = false;
};

}}} // namespace ppl::nn::x86

#endif
