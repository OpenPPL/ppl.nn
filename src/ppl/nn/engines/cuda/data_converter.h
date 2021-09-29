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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_DATA_CONVERTER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_DATA_CONVERTER_H_

#include "ppl/nn/common/data_converter.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"

namespace ppl { namespace nn { namespace cuda {

class CudaDevice;

class CudaDataConverter final : public DataConverter {
public:
    void SetDevice(CudaDevice* d) {
        device_ = d;
    }

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc) const override;

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc) const override;

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                 const TensorShape& src_desc) const;

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                       const BufferDesc& src, const TensorShape& src_desc,
                                       const CudaTensorQuant& src_quant) const;

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                         const void* src, const TensorShape& src_desc,
                                         const CudaTensorQuant& src_quant) const;

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                 const BufferDesc& src, const TensorShape& src_desc,
                                 const CudaTensorQuant& src_quant) const;

private:
    CudaDevice* device_;
};

}}} // namespace ppl::nn::cuda

#endif
