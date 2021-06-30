#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_TILE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_TILE_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct TileParam {
    int32_t axis;
    int32_t tiles;

    bool operator==(const TileParam& p) const {
        return this->axis == p.axis && this->tiles == p.tiles;
    }
};

}}} // namespace ppl::nn::common

#endif
