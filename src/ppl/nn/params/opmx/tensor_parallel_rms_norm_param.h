#ifndef _ST_HPC_PPL_NN_PARAMS_OPMX_TENSOR_PARALLEL_RMS_NORM_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_OPMX_TENSOR_PARALLEL_RMS_NORM_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace opmx {

struct TensorParallelRMSNormParam final : public ir::TypedAttr<TensorParallelRMSNormParam> {
    int32_t axis;
    float eps;
    float scale;

    bool operator==(const TensorParallelRMSNormParam& p) const {
        return (axis == p.axis && fabs(eps - p.eps) <= 1e-05 && fabs(scale - p.scale) <= 1e-05);
    }
};

}}} // namespace ppl::nn::opmx

#endif
