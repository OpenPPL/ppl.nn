#pragma once

#include "ppl/common/threadpool.h"
#include "ppl/common/log.h"

class ThreadPool {
private:
    ppl::common::StaticThreadPool pool_;
    std::vector<ppl::common::RetCode> retcode_;

public:
    void Init(const int32_t nthr) {
        pool_.Init(nthr);
        retcode_.resize(nthr);
    }

    void Run(const std::function<ppl::common::RetCode(uint32_t nr_threads, uint32_t thread_idx)>& f) {
        pool_.Run([&] (uint32_t nthr, uint32_t tid) {
            retcode_[tid] = f(nthr, tid);
        });
        for (size_t i = 0; i < retcode_.size(); ++i) {
            if (retcode_[i] != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "thread " << i << " exit with error: "
                    << ppl::common::GetRetCodeStr(retcode_[i]);
                exit(-1);
            }
        }
    }
};

