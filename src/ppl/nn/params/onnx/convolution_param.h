#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CONVOLUTION_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CONVOLUTION_PARAM_H_

#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct ConvolutionParam {
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> dilations;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;

    int32_t group;
    int32_t num_output; // written in op ctx, for converted filter
    int32_t bias_term; // written in op ctx, for multi-input layer fusion

    bool operator==(const ConvolutionParam& p) const {
        return false; // has attr written in op ctx
    }
};

}}} // namespace ppl::nn::common

#endif
