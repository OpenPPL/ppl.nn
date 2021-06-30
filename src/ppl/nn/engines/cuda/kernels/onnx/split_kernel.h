#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SPLIT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_SPLIT_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/split_param.h"

namespace ppl { namespace nn { namespace cuda {

class SplitKernel : public CudaKernel {
public:
    SplitKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::SplitParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::SplitParam* param_ = nullptr;
    mutable std::vector<std::vector<int64_t>> dst_dims_;
    mutable std::vector<void*> dst_list_;
};

}}} // namespace ppl::nn::cuda

#endif
