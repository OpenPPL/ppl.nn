#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CONVTRANSPOSE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CONVTRANSPOSE_PARAM_H_

#include <stdint.h>
#include <string>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct ConvTransposeParam {
    std::string auto_pad;
    int64_t group;
    std::vector<int32_t> dilations;
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> pads;
    std::vector<int32_t> strides;
    std::vector<int32_t> output_padding;
    std::vector<int32_t> output_shape;

    bool operator==(const ConvTransposeParam& p) const {
        return this->auto_pad == p.auto_pad && this->group == p.group && this->dilations == p.dilations &&
            this->kernel_shape == p.kernel_shape && this->pads == p.pads && this->strides == p.strides &&
            this->output_padding == p.output_padding && this->output_shape == p.output_shape;
    }
};

}}} // namespace ppl::nn::common

#endif
