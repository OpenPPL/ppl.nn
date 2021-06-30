#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_SCATTER_ELEMENTS_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_SCATTER_ELEMENTS_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeScatterElements(InputOutputInfo*, const void*);

}}} // namespace ppl::nn::oputils

#endif
