#include "ppl/nn/engines/common/onnx/loop_op.h"
#include "ppl/nn/engines/common/onnx/loop_kernel.h"
#include "ppl/nn/models/onnx/params/loop_param.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

RetCode LoopOp::Init(utils::SharedResource* resource, onnx::LoopParam* loop_param,
                     LoopConcatOutputFunc concat_output_func) {
    auto status = utils::ProcessGraph(resource, &loop_param->graph, &graph_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "ProcessGraph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenerateRuntimeAuxInfo(graph_info_, &aux_info_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    graph_ = loop_param->graph;
    resource_ = resource;
    concat_output_func_ = concat_output_func;

    return RC_SUCCESS;
}

KernelImpl* LoopOp::CreateKernelImpl() const {
    auto kernel = unique_ptr<LoopKernel>(new LoopKernel(node_));
    auto status = kernel->SetExecutionInfo(graph_.topo, &graph_info_, &aux_info_, resource_, concat_output_func_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "SetExecutionInfo of kernel[" << kernel->GetName() << "] failed: " << GetRetCodeStr(status);
        return nullptr;
    }

    return kernel.release();
}

}}} // namespace ppl::nn::common
