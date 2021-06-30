#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_PARAMS_SPLIT_TO_SEQUENCE_PARAM_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_PARAMS_SPLIT_TO_SEQUENCE_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace onnx {

struct SplitToSequenceParam {
    int32_t axis;
    int32_t keepdims;

    bool operator==(const SplitToSequenceParam& p) const {
        return ((axis == p.axis) && (keepdims == p.keepdims));
    }
};

}}} // namespace ppl::nn::onnx

#endif
