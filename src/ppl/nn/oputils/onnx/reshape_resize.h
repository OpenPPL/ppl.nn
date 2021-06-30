#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_RESIZE_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_RESIZE_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/resize_param.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeResize(InputOutputInfo*, const void*, const float* roi_data, const float* scales_data,
                                   const int64_t* sizes_data);

ppl::common::RetCode ReshapeResize(InputOutputInfo*, const void*);

}}} // namespace ppl::nn::oputils

#endif
