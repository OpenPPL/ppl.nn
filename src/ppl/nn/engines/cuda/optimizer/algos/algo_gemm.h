#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_GEMM_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_GEMM_H_

#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"

#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

class GemmAlgorithm : public Algorithm {
public:
    GemmAlgorithm() {
        std::set<dataformat_t> nhwc{DATAFORMAT_NHWC};
        gemm_formats_.emplace(DATAFORMAT_NHWC, nhwc);
    }

    const std::map<dataformat_t, std::set<dataformat_t>> Getformats(const std::string& type_name) override {
        return gemm_formats_;
    }

public:
    void GetAttrParam(void*& param) override;
    void DeleteAttrParam(void*& param) override;
    const double ExcuteTimer(ir::Node* node, OptKernelOptions& options) override;
    RetCode ModifyParam(const ir::Node* node, OptKernelOptions& options) override;
    void ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                        dataformat_t input_format, dataformat_t output_format) override;

private:
    CudaGemmParam attr_param_;
    std::map<dataformat_t, std::set<dataformat_t>> gemm_formats_;
};

}}} // namespace ppl::nn::cuda

#endif
