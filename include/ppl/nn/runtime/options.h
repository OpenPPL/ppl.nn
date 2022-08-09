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

#ifndef _ST_HPC_PPL_NN_RUNTIME_OPTIONS_H_
#define _ST_HPC_PPL_NN_RUNTIME_OPTIONS_H_

namespace ppl { namespace nn {

/** options for Runtime::Configure() */
enum {
    /**
       @brief args: true/false.
       @note this option may cause performance loss
    */
    RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG = 0,

    /**
       @brief infer shapes before running
       @note input shapes MUST be set first
    */
    RUNTIME_CONF_INFER_SHAPES,

    /**
       @note example:
       @code{.cpp}
       auto sched = new MyScheduler(); // `MyScheduler` is a derived class of `Scheduler`
       runtime->Configure(RUNTIME_CONF_SET_SCHEDULER, sched);
       // using runtime
       delete sched; // delete scheduler after runtime is released
       @endcode
    */
    RUNTIME_CONF_SET_SCHEDULER,

    RUNTIME_CONF_MAX,
};

}} // namespace ppl::nn

#endif
