#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_LOOP_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_LOOP_KERNEL_H_

#include "ppl/nn/engines/common/common_kernel_impl.h"
#include "ppl/nn/runtime/runtime_impl.h"

namespace ppl { namespace nn { namespace common {

typedef ppl::common::RetCode (*LoopConcatOutputFunc)(const std::vector<TensorBufferInfo>&, BufferDesc*);

class LoopKernel final : public CommonKernelImpl {
public:
    LoopKernel(const ir::Node* node) : CommonKernelImpl(node) {}
    ppl::common::RetCode SetExecutionInfo(const std::shared_ptr<ir::GraphTopo>&, const RuntimeGraphInfo*,
                                          const RuntimeAuxInfo*, utils::SharedResource*, LoopConcatOutputFunc func);

protected:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    RuntimeImpl subgraph_;
    LoopConcatOutputFunc concat_output_func_;
};

}}} // namespace ppl::nn::common

#endif
