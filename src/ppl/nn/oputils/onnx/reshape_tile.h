#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_TILE_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_TILE_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/tile_param.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapeTile(InputOutputInfo*, const void*);
ppl::common::RetCode ReshapeTile(InputOutputInfo*, const void*, const int64_t*);

}}} // namespace ppl::nn::oputils

#endif
