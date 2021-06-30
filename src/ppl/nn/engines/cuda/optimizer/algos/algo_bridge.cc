#include "ppl/nn/engines/cuda/optimizer/algos/algo_bridge.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const double BridgeAlgorithm::ExcuteTimer(ir::Node* node, OptKernelOptions& options) {
    auto data = options.graph->data.get();
    auto preedge_id = node->GetInput(0);
    auto preshape = options.tensors->find(preedge_id)->second.get()->GetShape();
    double timer = 0.0;

    if (input_format_ != output_format_) {
        // if (preshape.GetDimCount() == 1) {
        //     return ALGO_DOUBLE_MAX;
        // }

        if (data->constants.find(preedge_id) != data->constants.end()) {
            return timer = 0.0;
        }

        return 1e-7 * preshape.GetElementsIncludingPadding();
    }
    return timer;
}

void BridgeAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    input_format_ = input_format;
    output_format_ = output_format;
    return;
}

}}} // namespace ppl::nn::cuda
