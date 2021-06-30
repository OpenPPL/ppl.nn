#ifndef _ST_HPC_PPL_NN_OPUTILS_MMCV_RESHAPE_MMCV_NON_MAX_SUPPRESSION_H_
#define _ST_HPC_PPL_NN_OPUTILS_MMCV_RESHAPE_MMCV_NON_MAX_SUPPRESSION_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeMMCVNonMaxSuppression(InputOutputInfo*, const void*);

}}} // namespace ppl::nn::oputils

#endif
