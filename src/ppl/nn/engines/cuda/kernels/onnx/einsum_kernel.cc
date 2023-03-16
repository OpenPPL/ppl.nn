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

#include "ppl/nn/engines/cuda/kernels/onnx/einsum_kernel.h"

#include "cudakernel/arithmetic/einsum.h"
#include "cudakernel/memory/transpose.h"
#include "cudakernel/memory/unsqueeze.h"
#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/reduce/reduce.h"

#include "ppl/common/destructor.h"

#include "ppl/nn/engines/cuda/module/cuda_module.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/bgemm.h"

#include "ppl/common/types.h"
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <vector>
#include<fstream>
using namespace std;

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode EinSumKernel::TransposeImpl(KernelExecContext* ctx, ppl::nn::onnx::TransposeParam& trans_param, int index) {
    auto input_tensor = ctx->GetInput<TensorImpl>(index);
    auto input_shape = input_tensor->GetShape();
    auto input_ptr = input_tensor->GetBufferPtr();

    ppl::nn::TensorShape usq_shape = *input_shape;
    ppl::nn::TensorShape perm_out_shape(usq_shape);

    std::vector<int64_t> permed_dims(trans_param.perm.size());
    for(uint64_t j=0; j< permed_dims.size();++j)
        permed_dims[j] = usq_shape.GetDim(trans_param.perm[j]);
    perm_out_shape.Reshape(permed_dims);

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = input_shape->CalcBytesIncludingPadding();
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);

    ppl::common::Destructor __tmp_buffer_guard1([this, &tmp_buffer_desc]() -> void {
    GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;
    
    TransposeKernelParam kernel_param;
    kernel_param.perm = trans_param.perm;
    status = PPLCUDATransposeForwardImp(GetStream(), kernel_param, &usq_shape, input_ptr, &perm_out_shape, tmp_buffer);
    cudaMemcpyAsync(input_ptr, tmp_buffer, usq_shape.CalcBytesIncludingPadding(), cudaMemcpyDeviceToDevice, GetStream());
    return status;
}

ppl::common::RetCode EinSumKernel::MatMulImpl(KernelExecContext* ctx) {
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    GemmKernelParam param_kernel_;
    param_kernel_.alpha = 1;
    param_kernel_.beta = 1;
    param_kernel_.transA = 0;
    param_kernel_.transB = 0;

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = *weight->GetShape();
    {
        auto align_size = 8;
        auto dim_count = newshape.GetDimCount();
        newshape.SetDim(dim_count - 2, (newshape.GetDim(dim_count - 2) + align_size - 1) / align_size * align_size);

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDABgemmModifyWeights(stream, weight->GetShape(), weight->GetBufferPtr(), weight_buffer.addr,
                                  &param_kernel_);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

    BufferDesc input0_buffer;
    auto newshape0 = *input0->GetShape();
    auto dim_count = newshape0.GetDimCount();
    auto K = newshape0.GetDim(dim_count - 1);
    auto align_size = 8;
    auto K_pad = (K + align_size - 1) / align_size * align_size;
    bool is_input0_pad = K != K_pad;
    void* bmm_input0;
    if (is_input0_pad) {
        newshape0.SetDim(dim_count - 1, K_pad);
        auto status = GetCudaDevice()->Realloc(newshape0, &input0_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDABgemmPadInput(stream, input0->GetShape(), input0->GetBufferPtr(), input0_buffer.addr, &param_kernel_);
        bmm_input0 = input0_buffer.addr;
    } else {
        bmm_input0 = input0->GetBufferPtr();
    }
    ppl::common::Destructor __input0_buffer_guard__([this, &input0_buffer]() -> void {
        GetCudaDevice()->Free(&input0_buffer);
    });

    auto newshape_out = *output->GetShape();
    auto out_dim_count = newshape_out.GetDimCount();
    auto N = newshape_out.GetDim(out_dim_count - 1);
    auto N_pad = (N + align_size - 1) / align_size * align_size;
    BufferDesc output_buffer;
    bool is_output_pad = N != N_pad;
    void* bgemm_out;
    if (is_output_pad) {
        newshape_out.SetDim(out_dim_count - 1, N_pad);
        auto status = GetCudaDevice()->Realloc(newshape_out, &output_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        bgemm_out = output_buffer.addr;
    } else {
        bgemm_out = output->GetBufferPtr();
    }
    ppl::common::Destructor __output_buffer_guard__([this, &output_buffer]() -> void {
        GetCudaDevice()->Free(&output_buffer);
    });

    fuse_param_t temp_fuse_param;

    auto stream = GetStream();
    CUfunction module_func = nullptr;
#ifdef PPLNN_ENABLE_CUDA_JIT
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);
    module_func = module->GetKernelFunc();
#endif

    const TensorShape& shape_in0 = *input0->GetShape();
    algo_param_t algo_info;
    algo_info.UseDefaultF1Kernel(GetCudaDevice()->GetDeviceProp());
    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        status = PPLCUDABgemmForwardImp(GetCudaDevice()->GetDeviceProp(), stream, module_func, input0->GetShape(), bmm_input0,
                                        weight->GetShape(), weight_buffer.addr, output->GetShape(), bgemm_out,
                                        param_kernel_, tmp_buffer, temp_fuse_param, algo_info);
    }

    if (is_output_pad) {
        PPLCUDABgemmCvtOutput(stream, output->GetShape(), output->GetBufferPtr(), bgemm_out);
    }

    return status;
}

ppl::common::RetCode EinSumKernel::DoExecute(KernelExecContext* ctx) {
    auto input_tensor0 = ctx->GetInput<TensorImpl>(0);
    auto input_shape0 = input_tensor0->GetShape();
    auto input0 = input_tensor0->GetBufferPtr();

    auto output_tensor = ctx->GetOutput<TensorImpl>(0);
    auto output_shape = output_tensor->GetShape();
    void* output = output_tensor->GetBufferPtr();

    auto input_tensor1 = ctx->GetInput<TensorImpl>(1);
    auto input_shape1 = input_tensor1->GetShape();
    auto input1 = input_tensor1->GetBufferPtr();

    ppl::common::RetCode status = RC_SUCCESS;
    auto equation = param_->param.equation;
    if(ctx->GetInputCount()==2 && equation == "...bac,...dae->...bdce" 
        && input_shape0->GetDimCount()==4){
        status = PPLCUDAEinSum_nbac_ndae_nbdce_2_ForwardImp(GetStream(), input_shape0, input0, input_shape1, input1, output_shape, output, equation);
    } else if(ctx->GetInputCount()==2 && equation == "...abc,...adc->...bdc" 
        && input_shape0->GetDimCount()==4){ 
        status = PPLCUDAEinSum_nabc_nadc_nbdc_ForwardImp(GetStream(), input_shape0, input0, input_shape1, input1, output_shape, output, equation);
    } else if (ctx->GetInputCount()==2 && equation == "i , j -> i j" ){
        status = PPLCUDAEinSum_i_j_ij_ForwardImp(GetStream(), input_shape0, input0, input_shape1, input1, output_shape, output, equation);
    } else if (ctx->GetInputCount()==2 && equation == "b i j, b j d -> b i d") { // matmul
        return MatMulImpl(ctx);
    } else if (ctx->GetInputCount()==2 && equation == "b i d, b j d -> b i j") { // transpose + matmul
        ppl::nn::onnx::TransposeParam trans_param;
        trans_param.perm = {0, 2, 1};
        TransposeImpl(ctx, trans_param, 1);
        return MatMulImpl(ctx);
    } else {
        LOG(INFO) << "Not support equation: "<< equation <<" with input count " << input_shape0->GetDimCount();
        status =  ppl::common::RC_UNSUPPORTED;
    }
    
    return status; 
    // nchw
    // deprecated ↓
    /*
    if(ctx->GetInputCount()==2 && equation == "...bac,...dae->...bdce" ){

        auto input_tensor1 = ctx->GetInput<TensorImpl>(1);
        auto input_shape1 = input_tensor1->GetShape();
        auto input1 = input_tensor1->GetBufferPtr();

        // input0 transpoe
        ppl::nn::TensorShape usq_shape0 = *input_shape0;
        std::vector<int64_t> usq_dim0(input_shape0->GetDims(), input_shape0->GetDims()+input_shape0->GetDimCount());
        usq_dim0.push_back(1);
        usq_dim0.push_back(1);
        usq_shape0.Reshape(usq_dim0);

        ppl::nn::onnx::TransposeParam param0;
        if(input_shape0->GetDimCount()==3)
            param0.perm = {1, 0, 2, 3, 4};
        else if(input_shape0->GetDimCount()==4)
             param0.perm = {0, 2, 1, 3, 4, 5};

        ppl::nn::TensorShape perm_out_shape0(usq_shape0);

        std::vector<int64_t> permed_dims0(param0.perm.size());
        for(uint64_t j=0; j< permed_dims0.size();++j)
            permed_dims0[j] = usq_shape0.GetDim(param0.perm[j]);
        perm_out_shape0.Reshape(permed_dims0);

            BufferDesc tmp_buffer_desc0;
            auto tmp_buffer_bytes = input_shape0->CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc0);

            ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc0]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc0);
            });
            auto tmp_buffer0 = tmp_buffer_desc0.addr;

        PPLCUDATransposeForwardImp(GetStream(), param0, &usq_shape0, input0, &perm_out_shape0, tmp_buffer0);
        //

        // input1 transpoe
        ppl::nn::TensorShape usq_shape1 = *input_shape1;
        std::vector<int64_t> usq_dim1(input_shape1->GetDims(), input_shape1->GetDims()+input_shape1->GetDimCount());
        usq_dim1.push_back(1);
        usq_dim1.push_back(1);
        usq_shape1.Reshape(usq_dim1);

        ppl::nn::onnx::TransposeParam param1;
        if(input_shape0->GetDimCount()==3)
            param1.perm = {1, 3, 4, 0, 2};
        else if(input_shape0->GetDimCount()==4)
            param1.perm = {0, 2, 4, 5, 1, 3};

        ppl::nn::TensorShape perm_out_shape1(usq_shape1);

        std::vector<int64_t> permed_dims1(param1.perm.size());
        for(uint64_t j=0; j< permed_dims1.size();++j)
            permed_dims1[j] = usq_shape1.GetDim(param1.perm[j]);
        perm_out_shape1.Reshape(permed_dims1);

            BufferDesc tmp_buffer_desc1;
            tmp_buffer_bytes = input_shape1->CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc1);

            ppl::common::Destructor __tmp_buffer_guard1([this, &tmp_buffer_desc1]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc1);
            });
            auto tmp_buffer1 = tmp_buffer_desc1.addr;

        PPLCUDATransposeForwardImp(GetStream(), param1, &usq_shape1, input1, &perm_out_shape1, tmp_buffer1);

        // mul
        ppl::nn::TensorShape mul_out_shape(perm_out_shape1);
        std::vector<int64_t> mul_out_dims(permed_dims1.size());
        for(uint64_t j=0;j<mul_out_dims.size();++j){
            mul_out_dims[j] = std::max(perm_out_shape0.GetDim(j), perm_out_shape1.GetDim(j));
        }
        mul_out_shape.Reshape(mul_out_dims);
            BufferDesc tmp_buffer_desc2;
            tmp_buffer_bytes = mul_out_shape.CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc2);

            ppl::common::Destructor __tmp_buffer_guard2([this, &tmp_buffer_desc2]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc2);
            });
            auto tmp_buffer2 = tmp_buffer_desc2.addr;

        PPLCUDAArithMeticMulForwardImp(GetStream(), &perm_out_shape0, tmp_buffer0, &perm_out_shape1,
                                       tmp_buffer1, &mul_out_shape, tmp_buffer2);

        // bytes = tmp_buffer_bytes;
        // buffer.clear();
        // buffer.resize(bytes);
        // cudaMemcpyAsync(buffer.data(), tmp_buffer2, bytes, cudaMemcpyDeviceToHost);
        // const string out_file_name2 = "mul_out.dat";
        // ofstream ofs2(out_file_name2, ios_base::out | ios_base::binary | ios_base::trunc);
        // if (!ofs2.is_open()) {
        //     LOG(ERROR) << "open output file[" << out_file_name << "]";
        //     return false;
        // }
        // ofs2.write(buffer.data(), bytes);

        // reduce sum
        uint32_t n_outer=1, n_reduce =1, n_inner=1;

        uint32_t reduce_axis = 0;
        if(input_shape0->GetDimCount()==4)
            reduce_axis=1;

        for(uint64_t j=0;j<reduce_axis;++j)
            n_outer *= mul_out_shape.GetDim(j);
        n_reduce *= mul_out_shape.GetDim(reduce_axis);
        for(uint64_t j=reduce_axis+1;j<mul_out_shape.GetDimCount();++j)
            n_inner *= mul_out_shape.GetDim(j);

        PPLReduceDimDes des(n_inner, n_reduce, n_outer);

        ppl::nn::TensorShape reduce_out_shape(mul_out_shape);
        std::vector<int64_t> reduce_out_dims(mul_out_shape.GetDimCount() - 1);
        for(uint64_t j=0;j<reduce_out_dims.size();++j){
            reduce_out_dims[j] = mul_out_dims[j+1];
        }
        reduce_out_shape.Reshape(reduce_out_dims);
            BufferDesc tmp_buffer_desc3;
            tmp_buffer_bytes = reduce_out_shape.CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc3);

            ppl::common::Destructor __tmp_buffer_guard3([this, &tmp_buffer_desc3]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc3);
            });
            auto tmp_buffer3 = tmp_buffer_desc3.addr;
        QuantParamCuda qparam;
        PPLCUDAReduceForwardImp(GetStream(), ReduceParam::ReduceSum,  des, &mul_out_shape, tmp_buffer2, &reduce_out_shape, tmp_buffer3, nullptr, &qparam);

        // perm back to output shape
        ppl::nn::onnx::TransposeParam param3;
        
        if(input_shape0->GetDimCount()==3)
            param3.perm = {0, 2, 1, 3};
        else if(input_shape0->GetDimCount()==4)
            param3.perm = {0, 1, 3, 2, 4};

        PPLCUDATransposeForwardImp(GetStream(), param3, &reduce_out_shape, tmp_buffer3, output_shape, output);

    }
    if(ctx->GetInputCount()==2 && equation == "...abc,...adc->...bdc" && 
        input_shape0->GetDimCount()==3){

        // input0 transpoe
        ppl::nn::TensorShape usq_shape0 = *input_shape0;
        std::vector<int64_t> usq_dim0(input_shape0->GetDims(), input_shape0->GetDims()+input_shape0->GetDimCount());
        usq_dim0.push_back(1);
        usq_shape0.Reshape(usq_dim0);

        ppl::nn::onnx::TransposeParam param0;
        param0.perm = {0, 1, 2, 3};

        ppl::nn::TensorShape perm_out_shape0(usq_shape0);

        std::vector<int64_t> permed_dims0(param0.perm.size());
        for(uint64_t j=0; j< permed_dims0.size();++j)
            permed_dims0[j] = usq_shape0.GetDim(param0.perm[j]);
        perm_out_shape0.Reshape(permed_dims0);

            BufferDesc tmp_buffer_desc0;
            auto tmp_buffer_bytes = input_shape0->CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc0);

            ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc0]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc0);
            });
            auto tmp_buffer0 = tmp_buffer_desc0.addr;

        PPLCUDATransposeForwardImp(GetStream(), param0, &usq_shape0, input0, &perm_out_shape0, tmp_buffer0);
        //

        // input1 transpoe
        ppl::nn::TensorShape usq_shape1 = *input_shape1;
        std::vector<int64_t> usq_dim1(input_shape1->GetDims(), input_shape1->GetDims()+input_shape1->GetDimCount());
        usq_dim1.push_back(1);
        usq_shape1.Reshape(usq_dim1);

        ppl::nn::onnx::TransposeParam param1;
        param1.perm = {0, 3, 2, 1};

        ppl::nn::TensorShape perm_out_shape1(usq_shape1);

        std::vector<int64_t> permed_dims1(param1.perm.size());
        for(uint64_t j=0; j< permed_dims1.size();++j)
            permed_dims1[j] = usq_shape1.GetDim(param1.perm[j]);
        perm_out_shape1.Reshape(permed_dims1);

            BufferDesc tmp_buffer_desc1;
            tmp_buffer_bytes = input_shape1->CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc1);

            ppl::common::Destructor __tmp_buffer_guard1([this, &tmp_buffer_desc1]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc1);
            });
            auto tmp_buffer1 = tmp_buffer_desc1.addr;

        PPLCUDATransposeForwardImp(GetStream(), param1, &usq_shape1, input1, &perm_out_shape1, tmp_buffer1);

        // mul
        ppl::nn::TensorShape mul_out_shape(perm_out_shape1);
        std::vector<int64_t> mul_out_dims(permed_dims1.size());
        for(uint64_t j=0;j<mul_out_dims.size();++j){
            mul_out_dims[j] = std::max(perm_out_shape0.GetDim(j), perm_out_shape1.GetDim(j));
        }
        mul_out_shape.Reshape(mul_out_dims);
            BufferDesc tmp_buffer_desc2;
            tmp_buffer_bytes = mul_out_shape.CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc2);

            ppl::common::Destructor __tmp_buffer_guard2([this, &tmp_buffer_desc2]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc2);
            });
            auto tmp_buffer2 = tmp_buffer_desc2.addr;

        PPLCUDAArithMeticMulForwardImp(GetStream(), &perm_out_shape0, tmp_buffer0, &perm_out_shape1,
                                       tmp_buffer1, &mul_out_shape, tmp_buffer2);

        // bytes = tmp_buffer_bytes;
        // buffer.clear();
        // buffer.resize(bytes);
        // cudaMemcpyAsync(buffer.data(), tmp_buffer2, bytes, cudaMemcpyDeviceToHost);
        // const string out_file_name2 = "mul_out.dat";
        // ofstream ofs2(out_file_name2, ios_base::out | ios_base::binary | ios_base::trunc);
        // if (!ofs2.is_open()) {
        //     LOG(ERROR) << "open output file[" << out_file_name << "]";
        //     return false;
        // }
        // ofs2.write(buffer.data(), bytes);

        // reduce sum
        uint32_t n_outer=1, n_reduce =1, n_inner=1;

        std::vector<uint32_t> reduce_axis = {0};
        for(uint64_t j=0;j<reduce_axis[0];++j)
            n_outer *= mul_out_shape.GetDim(j);

        for(uint64_t j=0;j<reduce_axis.size();++j)
        n_reduce *= mul_out_shape.GetDim(reduce_axis[j]);
        for(uint64_t j=reduce_axis[reduce_axis.size()-1]+1;j<mul_out_shape.GetDimCount();++j)
            n_inner *= mul_out_shape.GetDim(j);

        PPLReduceDimDes des(n_inner, n_reduce, n_outer);

        ppl::nn::TensorShape reduce_out_shape(mul_out_shape);
        std::vector<int64_t> reduce_out_dims(mul_out_shape.GetDimCount() - reduce_axis.size());
        for(uint64_t j=0;j<reduce_out_dims.size();++j){
            reduce_out_dims[j] = mul_out_dims[j+reduce_axis.size()];
        }
        reduce_out_shape.Reshape(reduce_out_dims);
            BufferDesc tmp_buffer_desc3;
            tmp_buffer_bytes = reduce_out_shape.CalcBytesIncludingPadding();
            status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc3);

            ppl::common::Destructor __tmp_buffer_guard3([this, &tmp_buffer_desc3]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc3);
            });
            auto tmp_buffer3 = tmp_buffer_desc3.addr;
        QuantParamCuda qparam;
        PPLCUDAReduceForwardImp(GetStream(), ReduceParam::ReduceSum,  des, &mul_out_shape, tmp_buffer2, &reduce_out_shape, tmp_buffer3, nullptr, &qparam);

        // perm back to output shape
        ppl::nn::onnx::TransposeParam param3;
        param3.perm = {0, 2, 1};

        PPLCUDATransposeForwardImp(GetStream(), param3, &reduce_out_shape, tmp_buffer3, output_shape, output);

    }
    return status;
    */
}
}}} // namespace ppl::nn::cuda
