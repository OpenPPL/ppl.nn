#include "ppl/nn/engines/cuda/optimizer/algos/algo_normal.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

const double NormalAlgorithm::ExcuteTimer(ir::Node* node, OptKernelOptions& options) {
    double timer = 1.0e-5;
    return timer;
}

void NormalAlgorithm::ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                     dataformat_t input_format, dataformat_t output_format) {
    for (uint32_t i = 0; i < node->GetInputCount(); ++i) {
        auto edge_id = node->GetInput(i);
        if (edge_id == INVALID_EDGEID) {
            continue;
        }
        auto shape = &tensors->find(edge_id)->second->GetShape();
        if (shape->GetDimCount() > 1) {
            shape->SetDataFormat(input_format);
        } else {
            shape->SetDataFormat(DATAFORMAT_NDARRAY);
        }
    }

    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto shape = &tensors->find(edge_id)->second->GetShape();
        shape->SetDataFormat(output_format);
    }
    return;
}

}}} // namespace ppl::nn::cuda
