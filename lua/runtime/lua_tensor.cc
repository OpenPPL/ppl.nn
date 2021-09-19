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

#include "ppl/common/retcode.h"
#include "ppl/nn/common/logger.h"
#include "../common/lua_tensor_shape.h"
#include "lua_tensor.h"
#include "luacpp.h"
#include <memory>
using namespace std;
using namespace luacpp;
using namespace ppl::common;

namespace ppl { namespace nn { namespace lua {

void RegisterTensor(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto tensor_shape_class = LuaClass<LuaTensorShape>(lmodule->Get("TensorShape"));

    auto lclass = lstate->CreateClass<LuaTensor>()
        .DefMember("GetName", [](const LuaTensor* tensor) -> const char* {
            return tensor->ptr->GetName();
        })
        .DefMember("GetShape", [tensor_shape_class](const LuaTensor* tensor) -> LuaUserData {
            return tensor_shape_class.CreateUserData(&tensor->ptr->GetShape());
        })
        .DefMember("ConvertFromHost", [](LuaTensor* ltensor, const LuaStringRef& buf,
                                         const LuaTable& dims_table, datatype_t data_type) -> RetCode {
            vector<int64_t> dims;
            dims_table.ForEach([&dims](uint32_t, const LuaObject& lobj) -> bool {
                dims.push_back(lobj.ToInteger());
                return true;
            });

            auto tensor = ltensor->ptr;
            TensorShape& shape = tensor->GetShape();
            shape.Reshape(dims);

            TensorShape src_shape = shape;
            shape.SetDataFormat(DATAFORMAT_NDARRAY);
            shape.SetDataType(data_type);

            auto status = tensor->ReallocBuffer();
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "realloc buffer of [" << shape.GetBytesIncludingPadding()
                           << "] bytes failed when setting data for tensor[" << tensor->GetName()
                           << "]: " << GetRetCodeStr(status);
                return status;
            }

            status = tensor->ConvertFromHost(buf.base, src_shape);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy data to tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }

            return RC_SUCCESS;
        })
        .DefMember("ConvertToHost", [lstate](LuaTensor* ltensor) -> LuaObject {
            auto tensor = ltensor->ptr;
            TensorShape dst_shape = tensor->GetShape();
            dst_shape.SetDataFormat(DATAFORMAT_NDARRAY);

            vector<char> data(dst_shape.GetBytesExcludingPadding());
            auto status = tensor->ConvertToHost(data.data(), dst_shape);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy data of tensor[" << tensor->GetName() << "] to host failed: "
                           << GetRetCodeStr(status);
                return lstate->CreateNil();
            }

            return lstate->CreateString(data.data(), data.size());
        });

    lmodule->Set("Tensor", lclass);
}

}}}
