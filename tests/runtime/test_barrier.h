#ifndef _ST_HPC_PPL_NN_TESTS_RUNTIME_TEST_BARRIER_H_
#define _ST_HPC_PPL_NN_TESTS_RUNTIME_TEST_BARRIER_H_

#include "ppl/nn/runtime/barrier.h"

namespace ppl { namespace nn { namespace test {

class TestBarrier final : public Barrier {
public:
    ppl::common::RetCode Refresh(uint32_t) override {
        return ppl::common::RC_SUCCESS;
    }
    ppl::common::RetCode Sync() override {
        return ppl::common::RC_SUCCESS;
    }
};

}}} // namespace ppl::nn::test

#endif
