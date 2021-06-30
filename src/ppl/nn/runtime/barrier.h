#ifndef _ST_HPC_PPL_NN_RUNTIME_BARRIER_H_
#define _ST_HPC_PPL_NN_RUNTIME_BARRIER_H_

#include "ppl/common/retcode.h"

namespace ppl { namespace nn {

class Barrier {
public:
    virtual ~Barrier() {}
    virtual ppl::common::RetCode Refresh(uint32_t task_queue_id) = 0;
    virtual ppl::common::RetCode Sync() = 0;
};

}} // namespace ppl::nn

#endif
