#ifndef _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_PAD_H_
#define _ST_HPC_PPL_NN_OPUTILS_ONNX_RESHAPE_PAD_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/input_output_info.h"

namespace ppl { namespace nn { namespace oputils {

ppl::common::RetCode ReshapePad(InputOutputInfo* info, const void* arg, const int64_t* start_pads,
                                const int64_t* end_pads);

ppl::common::RetCode ReshapePad(InputOutputInfo*, const void*);

}}} // namespace ppl::nn::oputils

#endif
