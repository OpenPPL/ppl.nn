#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_LSTM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_LSTM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/params/onnx/lstm_param.h"
#include "cudakernel/common/rnn_common.h"

namespace ppl { namespace nn { namespace cuda {

class LstmKernel : public CudaKernel {
public:
    LstmKernel(const ir::Node* node) : CudaKernel(node) {}
    void SetParam(const ppl::nn::common::LSTMParam* p) {
        param_ = p;
        if (p->direction == ppl::nn::common::LSTMParam::DIR_FORWARD) {
            direction_ = RnnDirection::forward;
        }
        if (p->direction == ppl::nn::common::LSTMParam::DIR_REVERSE) {
            direction_ = RnnDirection::reverse;
        }
        if (p->direction == ppl::nn::common::LSTMParam::DIR_BIDIRECTIONAL) {
            direction_ = RnnDirection::bidirectional;
        }
    }

private:
    bool CanDoExecute(const KernelExecContext& ctx) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    const ppl::nn::common::LSTMParam *param_ = nullptr;
    unsigned int direction_;
};

}}} // namespace ppl::nn::cuda

#endif
