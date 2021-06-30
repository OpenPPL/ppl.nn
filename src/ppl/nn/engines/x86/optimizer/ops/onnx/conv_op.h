#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_CONV_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPS_ONNX_CONV_OP_H_

#include "ppl/nn/params/onnx/convolution_param.h"
#include "ppl/nn/engines/x86/params/convolution_param.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"

namespace ppl { namespace nn { namespace x86 {

class ConvOp final : public X86OptKernel {
public:
    ConvOp(const ir::Node* node) : X86OptKernel(node), conv2d_param_(nullptr) {}

    ~ConvOp();
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    ppl::common::RetCode SelectAlgorithm(const InputOutputInfo& info, const OptKernelOptions& options) override;
    bool SetFuseReLU();
    bool SetFuseReLU6();
    bool SetFuseSum();

private:
    Convolution2DParam* conv2d_param_;
    std::shared_ptr<ppl::nn::common::ConvolutionParam> param_;
};

}}} // namespace ppl::nn::x86

#endif
