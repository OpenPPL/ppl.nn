// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_PPL_KERNEL_X86_COMMON_TIMER_H_
#define __ST_PPL_KERNEL_X86_COMMON_TIMER_H_

#include <string>
#include <vector>
#include <chrono>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

class timer_t {
public:
    timer_t() {}

    void init(const int32_t num_timers)
    {
        num_timers_ = max(num_timers, 1);
        records_.resize(num_timers_);
        temp_points_.resize(num_timers_);
        clear();
    }
    void clear()
    {
        for (int32_t i = 0; i < num_timers_; ++i) {
            records_[i] = 0.0;
        }
    }

    void tic(const int32_t id)
    {
        temp_points_[id] = std::chrono::high_resolution_clock::now();
    }
    void toc(const int32_t id)
    {
        auto end = std::chrono::high_resolution_clock::now();
        records_[id] += std::chrono::duration_cast<std::chrono::microseconds>(end - temp_points_[id]).count();
    }

    double Seconds(const int32_t id) const
    {
        return records_[id] / 1e6;
    }
    double Milliseconds(const int32_t id) const
    {
        return records_[id] / 1e3;
    }
    double Microseconds(const int32_t id) const
    {
        return double(records_[id]);
    }

    int32_t num_timers() const
    {
        return num_timers_;
    }

private:
    int32_t num_timers_;
    std::vector<uint64_t> records_;
    std::vector<std::chrono::high_resolution_clock::time_point> temp_points_;
};

class thread_timer_t {
public:
    thread_timer_t() {}

    void init(const int32_t num_timers);
    void clear();

    void tic(const int32_t id);
    void toc(const int32_t id);

    double Seconds(const int32_t id) const;
    double Milliseconds(const int32_t id) const;
    double Microseconds(const int32_t id) const;

    std::vector<double> gather_seconds(const int32_t id) const;
    std::vector<double> gather_milliseconds(const int32_t id) const;
    std::vector<double> gather_microseconds(const int32_t id) const;

    std::string export_csv(const char **headers, const bool percentage) const;

    int32_t num_timers() const
    {
        return thread_timers_[0].num_timers();
    }

private:
    std::vector<timer_t> thread_timers_;
};

}}}; // namespace ppl::kernel::x86

#endif
