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

#ifndef _ST_HPC_PPL_NN_IR_EDGE_H_
#define _ST_HPC_PPL_NN_IR_EDGE_H_

#include "ppl/nn/common/types.h"
#include "ppl/nn/utils/vector_utils.h"
#include <string>

namespace ppl { namespace nn { namespace ir {

class Edge {
public:
    typedef utils::VecIter<nodeid_t> ConsumerIter;

public:
    virtual ~Edge() {}

    /** @brief get the id of this edge */
    virtual edgeid_t GetId() const = 0;

    virtual void SetName(const std::string& name) = 0;
    virtual const std::string& GetName() const = 0;

    /**
       @brief get producer node id
       @return producer node id, or INVALID_NODEID if this edge is constant or input.
    */
    virtual nodeid_t GetProducer() const = 0;

    /** @brief set this edge's producer to `nid` */
    virtual void SetProducer(nodeid_t nid) = 0;

    /** @brief create an iterator for iterating all consumers */
    virtual ConsumerIter CreateConsumerIter() const = 0;

    /** @brief get the number of consumers */
    virtual uint32_t CalcConsumerCount() const = 0;

    /** @brief add a node specified by `nid` to edge's consumer list */
    virtual void AddConsumer(nodeid_t) = 0;

    /**
       @brief remove consumer specified by `nid`
       @return true if `nid` is found, false otherwise.
    */
    virtual bool DelConsumer(nodeid_t nid) = 0;

    /** @brief remove all consumers */
    virtual void ClearConsumer() = 0;
};

}}} // namespace ppl::nn::ir

#endif
