#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CONSTANT_OF_SHAPE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CONSTANT_OF_SHAPE_PARAM_H_

#include "ppl/common/types.h"
#include <string>
#include <vector>
#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct ConstantOfShapeParam {
    ppl::common::datatype_t data_type;
    std::vector<int64_t> dims;
    std::string data;

    bool operator==(const ConstantOfShapeParam& p) const {
        return this->data_type == p.data_type && this->dims == p.dims && this->data == p.data;
    }
};

}}} // namespace ppl::nn::common

#endif
