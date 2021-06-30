#ifndef _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_PARSER_H_
#define _ST_HPC_PPL_NN_QUANTIZATION_QUANT_PARAM_PARSER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/quantization/quant_param_info.h"

namespace ppl { namespace nn {

class QuantParamParser final {
public:
    static ppl::common::RetCode Parse(const char* json_file, QuantParamInfo*);
};

}} // namespace ppl::nn

#endif
