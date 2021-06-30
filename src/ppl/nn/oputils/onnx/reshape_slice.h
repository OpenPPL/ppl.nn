#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_SLICE_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_SLICE_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeSlice(InputOutputInfo* info, const int64_t* starts, const int64_t* ends,
                                  const int64_t* axes, const int64_t* steps);

ppl::common::RetCode ReshapeSlice(InputOutputInfo*);

}}} // namespace ppl::nn::oputils

#endif
