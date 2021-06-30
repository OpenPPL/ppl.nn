#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_CONSTANT_OF_SHAPE_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_CONSTANT_OF_SHAPE_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/constant_of_shape_param.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeConstantOfShape(InputOutputInfo* info, const void* arg, const int64_t* input_host);
ppl::common::RetCode ReshapeConstantOfShape(InputOutputInfo*, const void*);

}}} // namespace ppl::nn::oputils

#endif
