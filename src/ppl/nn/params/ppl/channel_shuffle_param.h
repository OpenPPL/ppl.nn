#ifndef _ST_HPC_PPL_NN_PARAMS_PPL_CHANNEL_SHUFFLE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PPL_CHANNEL_SHUFFLE_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct ChannelShuffleParam {
    int32_t group;

    bool operator==(const ChannelShuffleParam& p) const {
        return this->group == p.group;
    }
};

}}} // namespace ppl::nn::common

#endif
