#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_SPLIT_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_SPLIT_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/split_param.h"
#include "ppl/nn/common/input_output_info.h"
#include <vector>

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeSplit(InputOutputInfo*, const void*);

}}} // namespace ppl::nn::oputils

#endif
