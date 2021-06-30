#include "demo_engine.h"
#include "demo_kernel.h"
#include "demo_engine_context.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/utils.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace demo {

EngineContext* DemoEngine::CreateEngineContext(const string&, const EngineContextOptions&) {
    return new DemoEngineContext(GetName());
}

static RetCode FillKernels(const ir::Graph* graph, RuntimePartitionInfo* info) {
    auto topo = graph->topo.get();
    for (auto it = topo->CreateNodeIter(); it->IsValid(); it->Forward()) {
        auto node = it->Get();
        info->kernels.emplace(node->GetId(), unique_ptr<OptKernel>(new DemoOptKernel(node)));
    }
    return RC_SUCCESS;
}

RetCode DemoEngine::ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) {
    auto status = utils::LoadConstants(*graph, &device_, &info->constants);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "FillConstants failed: " << GetRetCodeStr(status);
        return status;
    }

    status = FillKernels(graph, info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "FillKernels failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::demo
