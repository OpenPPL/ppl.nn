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

#include "lua_tensor_shape.h"
#include "luacpp.h"
#include <memory>
using namespace std;
using namespace luacpp;
using namespace ppl::common;

namespace ppl { namespace nn { namespace lua {

void RegisterTensorShape(const shared_ptr<LuaState>& lstate, const shared_ptr<LuaTable>& lmodule) {
    auto lclass = lstate->CreateClass<LuaTensorShape>()
        .DefMember("GetDims", [lstate](const LuaTensorShape* lshape) -> LuaTable {
            auto table = lstate->CreateTable();
            auto dim_count = lshape->ptr->GetRealDimCount();
            for (uint32_t i = 0; i < dim_count; ++i) {
                table.SetInteger(i + 1, lshape->ptr->GetDim(i));
            }
            return table;
        })
        .DefMember("SetDims", [](LuaTensorShape* lshape, const LuaTable& dims_table) -> void {
            vector<int64_t> dims;
            dims_table.ForEach([&dims](uint32_t, const LuaObject& lobj) -> bool {
                dims.push_back(lobj.ToInteger());
                return true;
            });
            lshape->ptr->Reshape(dims);
        })
        .DefMember("GetDataType", [](const LuaTensorShape* lshape) -> datatype_t {
            return lshape->ptr->GetDataType();
        })
        .DefMember("GetDataFormat", [](const LuaTensorShape* lshape) -> dataformat_t {
            return lshape->ptr->GetDataFormat();
        })
        .DefMember("IsScalar", [](const LuaTensorShape* lshape) -> bool {
            return lshape->ptr->IsScalar();
        });
    lmodule->Set("TensorShape", lclass);
}

}}}
