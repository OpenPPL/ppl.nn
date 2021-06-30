#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CONSTANT_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CONSTANT_PARAM_H_

#include "ppl/common/types.h"
#include <string>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct ConstantParam {
    ppl::common::datatype_t data_type;
    ppl::common::dataformat_t data_format;
    std::vector<int64_t> dims;
    std::string data;

    bool operator==(const ConstantParam& p) const {
        return this->data_type == p.data_type && this->data_format == p.data_format && this->dims == p.dims &&
            this->data == p.data;
    }
};

}}} // namespace ppl::nn::common

#endif
