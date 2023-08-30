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

#include "ppl/nn/engines/cuda/cuda_device.h"
#include "ppl/nn/common/logger.h"
#include "cudakernel/reformat/reformat.h"
#include "ppl/common/cuda/cuda_env.h"
#include "ppl/common/destructor.h"
#include "ppl/common/cuda/cuda_types.h"
#include <stdarg.h>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

CudaDevice::~CudaDevice() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    if (cublas_handle_) {
        cublasLtDestroy(cublas_handle_);
    }
    if (device_id_ != INT_MAX) {
        DestroyCudaEnv(device_id_);
    }
}

RetCode CudaDevice::Init(int device_id, ppl::common::NcclParam* tp_nccl_param, bool enable_cuda_graph) {
    auto status = InitCudaEnv(device_id);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitCudaEnv failed: " << GetRetCodeStr(status);
        return status;
    }

    auto err = cudaGetDeviceProperties(&device_prop_, device_id);
    if (err != cudaSuccess) {
        LOG(ERROR) << "get device properties failed: " << cudaGetErrorString(err);
        return RC_UNSUPPORTED;
    }

    if (!stream_) {
        cudaStreamCreate(&stream_);
    }
    if (!cublas_handle_) {
        cublasLtCreate(&cublas_handle_);
    }

    tp_nccl_param_ = tp_nccl_param;
    device_id_ = device_id;
    enable_cuda_graph_ = enable_cuda_graph;

    return RC_SUCCESS;
}

RetCode CudaDevice::SyncStream() {
    if (stream_) {
        auto rc = CheckCaptureStreamSync(stream_);
        if (rc != cudaSuccess) {
            LOG(ERROR) << "sync stream failed: " << cudaGetErrorString(rc);
            return RC_OTHER_ERROR;
        }
    }
    return RC_SUCCESS;
}

// Copy from host
RetCode CudaDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src, bytes, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const {
    auto rc = CopyFromHostAsync(dst, src, bytes);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CopyFromHostAsync failed.";
        return RC_OTHER_ERROR;
    }
    cudaError_t err = CheckCaptureStreamSync(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode CudaDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

// Copy to host
RetCode CudaDevice::CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst, src.addr, bytes, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    auto rc = CopyToHostAsync(dst, src.addr, bytes);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CopyToHostAsync failed.";
        return RC_OTHER_ERROR;
    }
    cudaError_t err = CheckCaptureStreamSync(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode CudaDevice::CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode CudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src.addr, bytes, cudaMemcpyDeviceToDevice, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode CudaDevice::ConvertToHostCommon(
    void* dst, const TensorShape& dst_desc, const BufferDesc& src, const TensorShape& src_desc, const void* src_info,
    const function<RetCode(void*, const BufferDesc&, const TensorShape&)>& copy_fn) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        // TODO release dst
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() && dst_desc.GetDataType() == src_desc.GetDataType()) {
        auto status = copy_fn(dst, src, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy dst data to Host failed: " << GetRetCodeStr(status);
        }
        return status;
    }

    BufferDesc tmp_buffer_desc;
    auto status = Realloc(dst_desc, &tmp_buffer_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
        return status;
    }
    Destructor __tmp_buffer_guard__([this, &tmp_buffer_desc]() -> void {
        Free(&tmp_buffer_desc);
    });

    status = Convert(&tmp_buffer_desc, dst_desc, src, src_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "convert in device failed during ConvertToHost: " << GetRetCodeStr(status);
        return status;
    }

    status = copy_fn(dst, tmp_buffer_desc, dst_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy dst data to Host failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                  const TensorShape& src_desc, const void* src_info) {
    return ConvertToHostCommon(dst, dst_desc, src, src_desc, src_info,
                               [this](void* dst, const BufferDesc& src, const TensorShape& dst_desc) -> RetCode {
                                   return CopyToHost(dst, src, dst_desc);
                               });
}

RetCode CudaDevice::ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc, const void* src_info) {
    auto rc = ConvertToHostCommon(dst, dst_desc, src, src_desc, src_info,
                                  [this](void* dst, const BufferDesc& src, const TensorShape& dst_desc) -> RetCode {
                                      return CopyToHostAsync(dst, src, dst_desc);
                                  });
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ConvertToHostAsync failed.";
        return rc;
    }

    cudaError_t err = CheckCaptureStreamSync(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::ConvertFromHostCommon(
    BufferDesc* dst, const TensorShape& dst_desc, const void* src, const TensorShape& src_desc, const void* dst_info,
    const function<RetCode(BufferDesc*, const void*, const TensorShape&)>& copy_fn) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        Free(dst);
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() && dst_desc.GetDataType() == src_desc.GetDataType()) {
        auto status = copy_fn(dst, src, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy src data from host failed: " << GetRetCodeStr(status);
        }
        return status;
    }

    BufferDesc tmp_buffer_desc;
    auto status = Realloc(src_desc, &tmp_buffer_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
        return status;
    }
    Destructor __tmp_buffer_guard__([this, &tmp_buffer_desc]() -> void {
        Free(&tmp_buffer_desc);
    });

    status = copy_fn(&tmp_buffer_desc, src, src_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy src data from host failed: " << GetRetCodeStr(status);
        return status;
    }

    status = Convert(dst, dst_desc, tmp_buffer_desc, src_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "convert in device failed during ConvertFromHost: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                    const TensorShape& src_desc, const void* dst_info) {
    return ConvertFromHostCommon(dst, dst_desc, src, src_desc, dst_info,
                                 [this](BufferDesc* dst, const void* src, const TensorShape& src_desc) -> RetCode {
                                     return CopyFromHost(dst, src, src_desc);
                                 });
}

RetCode CudaDevice::ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc, const void* dst_info) {
    auto rc = ConvertFromHostCommon(dst, dst_desc, src, src_desc, dst_info,
                                    [this](BufferDesc* dst, const void* src, const TensorShape& src_desc) -> RetCode {
                                        return CopyFromHostAsync(dst, src, src_desc);
                                    });
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ConvertFromHost failed.";
        return rc;
    }

    cudaError_t err = CheckCaptureStreamSync(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                            const TensorShape& src_desc, const void*, const void*) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        Free(dst);
        return RC_SUCCESS;
    }

    ReFormatParam param;
    auto status = SetReLayoutParam(&param, src_desc, dst_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "illegal convert layout condition.";
        return status;
    }

    if (src_desc.GetDimCount() > 1 && dst_desc.GetDimCount() > 1) {
        auto src_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(src_desc.GetDataFormat());
        auto dst_align_size = ppl::common::cuda::GetDataFormatChannelAlignment(dst_desc.GetDataFormat());
        if (src_desc.CalcElementsFromDimensionIncludingPadding(2) == 1 &&
            dst_desc.CalcElementsFromDimensionIncludingPadding(2) == 1) {
            if (src_desc.GetDim(1) % src_align_size == 0 && dst_desc.GetDim(1) % dst_align_size == 0) {
                param.in_format = param.out_format;
                param.mix_format = 0;
            }
        }
    }

    if (param.in_format == param.out_format && param.in_type == param.out_type) {
        status = Copy(dst, src, dst_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy from source to des failed";
            return status;
        }
    } else if (param.in_format == param.out_format || param.in_type == param.out_type) {
        PPLCUDADataConvert(GetStream(), src.addr, dst->addr, nullptr, param);
    } else {
        auto shape_size = src_desc.CalcElementsIncludingPadding() * GetSizeOfDataType(dst_desc.GetDataType());

        BufferDesc tmp_buffer_desc;
        status = Realloc(shape_size, &tmp_buffer_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
            return status;
        }
        Destructor __tmp_buffer_guard__([this, &tmp_buffer_desc]() -> void {
            Free(&tmp_buffer_desc);
        });

        PPLCUDADataConvert(GetStream(), src.addr, dst->addr, tmp_buffer_desc.addr, param);
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::ConvertToHost(void* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                  const BufferDesc& src, const TensorShape& src_desc,
                                  const CudaTensorQuant& src_quant) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        // TODO release dst
        return RC_SUCCESS;
    }

    BufferDesc tmp_buffer_desc;
    auto status = Realloc(dst_desc, &tmp_buffer_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
        return status;
    }
    Destructor __tmp_buffer_guard__([this, &tmp_buffer_desc]() -> void {
        Free(&tmp_buffer_desc);
    });

    status = Convert(&tmp_buffer_desc, dst_desc, dst_quant, src, src_desc, src_quant);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "convert in device failed during ConvertToHost: " << GetRetCodeStr(status);
        return status;
    }

    status = CopyToHost(dst, tmp_buffer_desc, dst_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy dst data to Host failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                    const void* src, const TensorShape& src_desc, const CudaTensorQuant& src_quant) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        Free(dst);
        return RC_SUCCESS;
    }

    BufferDesc tmp_buffer_desc;
    auto status = Realloc(src_desc, &tmp_buffer_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
        return status;
    }
    Destructor __tmp_buffer_guard__([this, &tmp_buffer_desc]() -> void {
        Free(&tmp_buffer_desc);
    });

    status = CopyFromHost(&tmp_buffer_desc, src, src_desc);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy src data from host failed: " << GetRetCodeStr(status);
        return status;
    }

    status = Convert(dst, dst_desc, dst_quant, tmp_buffer_desc, src_desc, src_quant);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "convert in device failed during ConvertFromHost: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::Convert(BufferDesc* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                            const BufferDesc& src, const TensorShape& src_desc, const CudaTensorQuant& src_quant) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        Free(dst);
        return RC_SUCCESS;
    }

    ReFormatParam param;
    CudaTensorKernelQuant src_quant_kernel, dst_quant_kernel;
    src_quant_kernel.format = src_quant.format;
    src_quant_kernel.type = src_quant.type;
    src_quant_kernel.per_channel = src_quant.per_channel;
    src_quant_kernel.bit_width = src_quant.bit_width;
    src_quant_kernel.scale = src_quant.scale;
    src_quant_kernel.zero_point = src_quant.zero_point;
    dst_quant_kernel.format = dst_quant.format;
    dst_quant_kernel.type = dst_quant.type;
    dst_quant_kernel.per_channel = dst_quant.per_channel;
    dst_quant_kernel.bit_width = dst_quant.bit_width;
    dst_quant_kernel.scale = dst_quant.scale;
    dst_quant_kernel.zero_point = dst_quant.zero_point;
    auto status = SetReLayoutParam(&param, src_desc, src_quant_kernel, dst_desc, dst_quant_kernel);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "illegal convert layout condition.";
        return status;
    }

    BufferDesc in_scale_buf, out_scale_buf;
    status = Realloc(param.quant_dim_size * sizeof(float), &in_scale_buf);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
        return status;
    }
    Destructor __in_scale_buf_guard__([this, &in_scale_buf]() -> void {
        Free(&in_scale_buf);
    });
    status = Realloc(param.quant_dim_size * sizeof(float), &out_scale_buf);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
        return status;
    }
    Destructor __out_scale_buf_guard__([this, &out_scale_buf]() -> void {
        Free(&out_scale_buf);
    });

    if (param.mix_type && (src_quant.per_channel || dst_quant.per_channel)) {
        vector<float> in_scale(src_quant.scale);
        if (in_scale.size() == 1) {
            float t_scale = in_scale[0];
            in_scale.resize(param.quant_dim_size, t_scale);
        } else {
            while (in_scale.size() < (unsigned)param.quant_dim_size) {
                in_scale.push_back(1.f);
            }
        }
        status = CopyFromHost(&in_scale_buf, &(in_scale[0]), ((int)param.quant_dim_size) * sizeof(float));
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "CopyFromHost failed";
            return status;
        }
        param.i_step_ptr = (float*)in_scale_buf.addr;

        vector<float> out_scale(dst_quant.scale);
        if (out_scale.size() == 1) {
            float t_scale = out_scale[0];
            out_scale.resize(param.quant_dim_size, t_scale);
        } else {
            while (out_scale.size() < (unsigned)param.quant_dim_size) {
                out_scale.push_back(1.f);
            }
        }
        status = CopyFromHost(&out_scale_buf, &(out_scale[0]), ((int)param.quant_dim_size) * sizeof(float));
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "CopyFromHost failed";
            return status;
        }
        param.o_step_ptr = (float*)out_scale_buf.addr;
    }

    if (!param.mix_format && !param.mix_type) {
        status = Copy(dst, src, dst_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Copy failed";
            return status;
        }
    } else if (!param.mix_format || !param.mix_type) {
        PPLCUDADataConvert(GetStream(), src.addr, dst->addr, nullptr, param);
    } else {
        auto shape_size = src_desc.CalcElementsIncludingPadding() * GetSizeOfDataType(dst_desc.GetDataType());

        BufferDesc tmp_buffer_desc;
        status = Realloc(shape_size, &tmp_buffer_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "alloc tmp buffer for dst tensor failed";
            return status;
        }
        Destructor __tmp_buffer_guard__([this, &tmp_buffer_desc]() -> void {
            Free(&tmp_buffer_desc);
        });

        PPLCUDADataConvert(GetStream(), src.addr, dst->addr, tmp_buffer_desc.addr, param);
    }

    return RC_SUCCESS;
}

RetCode CudaDevice::Synchronize() {
    auto err = CheckCaptureStreamSync(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

/* -------------------------------------------------------------------------- */

RetCode CudaDevice::ConfGetDeviceId(CudaDevice* dev, va_list args) {
    auto did = va_arg(args, int*);
    *did = dev->device_id_;
    return RC_SUCCESS;
}

CudaDevice::ConfHandlerFunc CudaDevice::conf_handlers_[] = {
    CudaDevice::ConfGetDeviceId,
};

RetCode CudaDevice::Configure(uint32_t option, ...) {
    if (option >= DEV_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << (uint32_t)DEV_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

cudaError_t CudaDevice::CheckCaptureStreamSync(cudaStream_t stream) const {
    if (!enable_cuda_graph_) {
        return cudaStreamSynchronize(stream);
    }
    cudaStreamCaptureStatus status;
    auto err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        return err;
    }
    if (status == cudaStreamCaptureStatusActive) {
        return cudaSuccess;
    }
    return cudaStreamSynchronize(stream);
}
}}} // namespace ppl::nn::cuda
