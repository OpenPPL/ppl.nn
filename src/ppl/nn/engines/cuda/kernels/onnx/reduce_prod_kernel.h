#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_REDUCE_PROD_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_REDUCE_PROD_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"

#include "ppl/nn/params/onnx/reduce_param.h"

namespace ppl { namespace nn { namespace cuda {

class ReduceProdKernel : public CudaKernel {
public:
    ReduceProdKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const ppl::nn::common::ReduceParam* p) {
        param_ = p;
    }

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    const ppl::nn::common::ReduceParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
