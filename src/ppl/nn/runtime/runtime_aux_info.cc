#include "ppl/nn/runtime/runtime_aux_info.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RetCode GenerateRuntimeAuxInfo(const RuntimeGraphInfo& graph_info, RuntimeAuxInfo* aux_info) {
    aux_info->sorted_nodes.resize(graph_info.kernels.size());
    for (uint32_t i = 0; i < graph_info.kernels.size(); ++i) {
        aux_info->sorted_nodes[i] = graph_info.kernels[i].op->GetNode()->GetId();
    }

    return RC_SUCCESS;
}

}} // namespace ppl::nn
