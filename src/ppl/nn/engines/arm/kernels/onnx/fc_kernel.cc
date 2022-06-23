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

#include "ppl/nn/engines/arm/kernels/onnx/fc_kernel.h"
#include "ppl/nn/utils/destructor.h"
#include "ppl/kernel/arm_server/fc/neon/fc.h"

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode FCKernel::DoExecute(KernelExecContext* ctx) {
    TensorImpl* A = ctx->GetInput<TensorImpl>(0);
    TensorImpl* Y = ctx->GetOutput<TensorImpl>(0);

    int64_t num_in = executor_->fc_param()->channels;
    int64_t num_out = executor_->fc_param()->num_output;
    int64_t num_batch = A->GetShape()->GetDim(0);
    uint32_t fuse_type = executor_->fc_param()->fuse_flag;
    (void)fuse_type;

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_ARM_DEBUG_TRACE("Input [A]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(A);
    PPLNN_ARM_DEBUG_TRACE("Output [Y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(Y);
    PPLNN_ARM_DEBUG_TRACE("channels: %ld\n", num_in);
    PPLNN_ARM_DEBUG_TRACE("num_output: %ld\n", num_out);
    PPLNN_ARM_DEBUG_TRACE("filter: %p\n", executor_->cvt_filter());
    PPLNN_ARM_DEBUG_TRACE("bias: %p\n", executor_->cvt_bias());

#ifdef PPLNN_USE_AARCH64
    ppl::common::RetCode rc;

    const auto data_type = A->GetShape()->GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        if (MayUseISA(ppl::common::ISA_ARMV8)) {
            sgemm_m1 = 32;
            sgemm_n1 = 64;
            sgemm_k1 = 128;
            sgemm_m2 = 256;
            sgemm_n2 = 256;
            sgemm_k3 = 2048;

            sgemm_m1 = std::min(num_batch, sgemm_m1);
            sgemm_m2 = std::min(num_batch, sgemm_m2);
            sgemm_n1 = std::min(num_out,   sgemm_n1);
            sgemm_n2 = std::min(num_out,   sgemm_n2);
            sgemm_k1 = std::min(num_in,    sgemm_k1);
            sgemm_k3 = std::min(num_in,    sgemm_k3);

            BufferDesc tmp_buffer_desc;
            auto tmp_buffer_size = ppl::kernel::arm_server::neon::ppl_arm_server_kernel_fp32_fc_get_buffer_size(
                num_in, num_out, num_batch,
                sgemm_m1, sgemm_n1, sgemm_k1, sgemm_k3);
            rc = GetArmDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                        << "] failed: " << ppl::common::GetRetCodeStr(rc);
                return rc;
            }
            utils::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
                GetArmDevice()->FreeTmpBuffer(&tmp_buffer_desc);
            });

            const float * cvt_filter = (const float *)executor_->cvt_filter();
            const float * cvt_bias   = (const float *)executor_->cvt_bias();
            const float * input  = A->GetBufferPtr<float>();
                  float * output = Y->GetBufferPtr<float>();
                  float * tmp_buffer_ptr = (float *)tmp_buffer_desc.addr;

            PPLNN_ARM_DEBUG_TRACE("buffer: %p\n", tmp_buffer_ptr);
            PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

            rc = ppl::kernel::arm_server::neon::fc_fp32(
                cvt_filter, cvt_bias, input, output, tmp_buffer_ptr,
                num_in, num_out, num_batch,
                sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m2, sgemm_n2, sgemm_k3);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
            }
            return rc;

        }
        else {
            LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type)
                    << "with isa " << GetISA() << ".";
            return ppl::common::RC_UNSUPPORTED;
        }
    }
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (data_type == ppl::common::DATATYPE_FLOAT16) {
        if (MayUseISA(ppl::common::ISA_ARMV8_2)) {
            sgemm_m1 = 32;
            sgemm_n1 = 64;
            sgemm_k1 = 256;
            sgemm_m2 = 256;
            sgemm_n2 = 256;
            sgemm_k3 = 1024;

            sgemm_m1 = std::min(num_batch, sgemm_m1);
            sgemm_m2 = std::min(num_batch, sgemm_m2);
            sgemm_n1 = std::min(num_out,   sgemm_n1);
            sgemm_n2 = std::min(num_out,   sgemm_n2);
            sgemm_k1 = std::min(num_in,    sgemm_k1);
            sgemm_k3 = std::min(num_in,    sgemm_k3);

            BufferDesc tmp_buffer_desc;
            auto tmp_buffer_size = ppl::kernel::arm_server::neon::ppl_arm_server_kernel_fp16_fc_get_buffer_size(
                num_in, num_out, num_batch,
                sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m3, sgemm_k3);
            rc = GetArmDevice()->AllocTmpBuffer(tmp_buffer_size, &tmp_buffer_desc);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_size << "] for kernel[" << GetName()
                        << "] failed: " << ppl::common::GetRetCodeStr(rc);
                return rc;
            }
            utils::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
                GetArmDevice()->FreeTmpBuffer(&tmp_buffer_desc);
            });

            const __fp16 * cvt_filter = (const __fp16 *)executor_->cvt_filter();
            const __fp16 * cvt_bias   = (const __fp16 *)executor_->cvt_bias();
            const __fp16 * input  = A->GetBufferPtr<__fp16>();
                  __fp16 * output = Y->GetBufferPtr<__fp16>();
                  __fp16 * tmp_buffer_ptr = (__fp16 *)tmp_buffer_desc.addr;

            PPLNN_ARM_DEBUG_TRACE("buffer: %p\n", tmp_buffer_ptr);
            PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

            rc = ppl::kernel::arm_server::neon::fc_fp16(
                cvt_filter, cvt_bias, input, output, tmp_buffer_ptr,
                num_in, num_out, num_batch,
                sgemm_m1, sgemm_n1, sgemm_k1, sgemm_m2, sgemm_n2, sgemm_k3);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "Execute failed: " << ppl::common::GetRetCodeStr(rc);
            }
            return rc;
        }
        else {
            LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type)
                    << "with isa " << GetISA() << ".";
            return ppl::common::RC_UNSUPPORTED;
        }
    }
#endif
#endif

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::arm
