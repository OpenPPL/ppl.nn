#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_BRIDGE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_BRIDGE_H_

#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

class BridgeAlgorithm : public Algorithm {
public:
    BridgeAlgorithm() {
        std::set<dataformat_t> ndarray{DATAFORMAT_NDARRAY, DATAFORMAT_NHWC};
        bridge_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        std::set<dataformat_t> nhwc{DATAFORMAT_NDARRAY, DATAFORMAT_NHWC};
        bridge_formats_.emplace(DATAFORMAT_NHWC, nhwc);
    }

    void GetAttrParam(void*& param) override {
        return;
    };
    void DeleteAttrParam(void*& param) override {
        return;
    };

    const std::map<dataformat_t, std::set<dataformat_t>> Getformats(const std::string& type_name) override {
        return bridge_formats_;
    }

    const double ExcuteTimer(ir::Node* node, OptKernelOptions& options) override;

    RetCode ModifyParam(const ir::Node*, OptKernelOptions& options) override {
        return RC_SUCCESS;
    }

    void ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                        dataformat_t input_format, dataformat_t output_format) override;

private:
    std::map<dataformat_t, std::set<dataformat_t>> bridge_formats_;
    dataformat_t input_format_ = 0;
    dataformat_t output_format_ = 0;
};

}}} // namespace ppl::nn::cuda

#endif
