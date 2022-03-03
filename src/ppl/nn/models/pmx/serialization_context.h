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

#ifndef _ST_HPC_PPL_NN_MODELS_PMX_SERIALIZATION_CONTEXT_H_
#define _ST_HPC_PPL_NN_MODELS_PMX_SERIALIZATION_CONTEXT_H_

namespace ppl { namespace nn { namespace pmx {

struct SerializationContext final {
    /** mappings between in-memory node ids and sequential numbers */
    std::vector<nodeid_t> nid2seq;
    std::vector<nodeid_t> seq2nid;
    /** mappings between in-memory edge ids and sequential numbers */
    std::vector<edgeid_t> eid2seq;
    std::vector<edgeid_t> seq2eid;
};

}}} // namespace ppl::nn::pmx

#endif
