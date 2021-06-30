#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_BATCH_NORMALIZATION_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_BATCH_NORMALIZATION_PARAM_H_

namespace ppl { namespace nn { namespace common {

struct BatchNormalizationParam {
    float epsilon;
    float momentum;

    bool operator==(const BatchNormalizationParam& p) const {
        return this->epsilon == p.epsilon && this->momentum == p.momentum;
    }
};

}}} // namespace ppl::nn::common

#endif
