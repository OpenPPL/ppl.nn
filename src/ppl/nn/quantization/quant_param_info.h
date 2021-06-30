#ifndef _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_INFO_H_
#define _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_INFO_H_

#include <stdint.h>
#include <string>
#include <map>

namespace ppl { namespace nn {

struct QuantParam {
    struct Value {
        /** `content` is binary data. */
        std::string content;
    };
    std::map<std::string, Value> fields;
};

struct QuantParamInfo {
    std::map<std::string, QuantParam> tensor_params;
};

}} // namespace ppl::nn

#endif
