#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_NON_MAX_SUPPRESSION_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_NON_MAX_SUPPRESSION_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeNonMaxSuppression(InputOutputInfo*, int64_t max_output_boxes_per_class);

ppl::common::RetCode ReshapeNonMaxSuppression(InputOutputInfo*);

}}} // namespace ppl::nn::oputils

#endif
