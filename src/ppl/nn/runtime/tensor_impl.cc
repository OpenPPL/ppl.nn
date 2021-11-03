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

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RetCode TensorImpl::ReallocBuffer() {
    if (!buffer_info_.IsBufferOwner() && buffer_info_.GetBufferPtr()) {
        LOG(WARNING) << "tensor[" << GetName() << "] is not the buffer owner. ReallocBuffer() does nothing.";
        return RC_SUCCESS;
    }

    return buffer_info_.ReallocBuffer();
}

RetCode TensorImpl::CopyToHost(void* dst) const {
    return buffer_info_.GetDevice()->CopyToHost(dst, buffer_info_.GetBufferDesc(), buffer_info_.GetShape());
}

RetCode TensorImpl::CopyFromHost(const void* src) {
    return buffer_info_.GetDevice()->CopyFromHost(&buffer_info_.GetBufferDesc(), src, buffer_info_.GetShape());
}

RetCode TensorImpl::ConvertToHost(void* dst, const TensorShape& dst_desc) const {
    auto converter = buffer_info_.GetDevice()->GetDataConverter();
    return converter->ConvertToHost(dst, dst_desc, buffer_info_.GetBufferDesc(), buffer_info_.GetShape());
}

RetCode TensorImpl::ConvertFromHost(const void* src, const TensorShape& src_desc) {
    auto converter = buffer_info_.GetDevice()->GetDataConverter();
    return converter->ConvertFromHost(&buffer_info_.GetBufferDesc(), buffer_info_.GetShape(), src, src_desc);
}

}} // namespace ppl::nn
