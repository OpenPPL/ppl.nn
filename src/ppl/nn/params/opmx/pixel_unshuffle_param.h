#ifndef _ST_HPC_PPL_NN_PARAMS_OPMX_PIXEL_UNSHUFFLE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_OPMX_PIXEL_UNSHUFFLE_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace opmx {

struct PixelUnshuffleParam final : public ir::TypedAttr<PixelUnshuffleParam> {
    enum {
        DATA_LAYOUT_NONE = 0,
        DATA_LAYOUT_NHWC = 1,
    };

    int32_t scale_factor;
    int32_t data_layout;

    bool operator==(const PixelUnshuffleParam& p) const {
        return (scale_factor == p.scale_factor
            && data_layout == p.data_layout);
    }
};

}}} // namespace ppl::nn::opmx

#endif
