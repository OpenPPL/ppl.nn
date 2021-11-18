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

#include <fstream>
#include <sstream>

#include "ppl/nn/engines/cuda/module/conv_compiler.h"
#include "ppl/nn/engines/cuda/module/cuda_compiler.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

#include "cudakernel/nn/conv/gene_kernel.h"

using namespace std;
namespace ppl { namespace nn { namespace cuda {

const ppl::common::RetCode ConvCompiler::Compile(ir::Node* node, const OptKernelOptions& options) {
    auto node_id = node->GetId();
    auto opt_kerenl = options.info->kernels.find(node_id)->second.get();
    CudaOptKernel* cuda_kernel = static_cast<CudaOptKernel*>(opt_kerenl);
    auto param = cuda_kernel->GetCommparam();
    CudaCommonParam* cuda_param = static_cast<CudaCommonParam*>(param);
    CudaConvParam* conv_param = static_cast<CudaConvParam*>(cuda_kernel->GetParam());
    auto algo_param = conv_param->extra_param.algo_info;
    auto edge_quant = cuda_param->cuda_tensor_info->at(node->GetInput(0));
    auto input_type = edge_quant.type;
    auto mgr = CodeGeneFactorManager::Instance();
    auto gene_factor = mgr->FindKernel(input_type);

    std::string source = "";
    if (algo_param.algo_name.find("Idxn") != std::string::npos) {
        gene_factor->GeneIdxnKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta,
                       algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta,
                       algo_param.tiles.k_per_step, 0);
        gene_factor->ReplaceFusionForIdxn(source, conv_param->extra_param.fuse_info);
    } else if (algo_param.algo_name.find("2spk") != std::string::npos) {
        gene_factor->Gene2spkKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta,
                       algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta,
                       algo_param.tiles.k_per_set, algo_param.splitk, algo_param.splitf, algo_param.tiles.buf, 0);
        gene_factor->ReplaceFusionFor2spk(source, conv_param->extra_param.fuse_info);
    } else if (algo_param.algo_name.find("Swzl") != std::string::npos) {
        gene_factor->GeneSwzlKernel(source, algo_param.algo_name, algo_param.tiles.m_cta, algo_param.tiles.n_cta,
                       algo_param.tiles.m_warp, algo_param.tiles.n_warp, algo_param.tiles.k_cta,
                       algo_param.splitk, algo_param.tiles.buf, 0);
        gene_factor->ReplaceFusionForSwzl(source, conv_param->extra_param.fuse_info);
    }
    std::string name = algo_param.algo_name;
    std::vector<std::string> compile_params;
    std::vector<const char*> param_cstring{};
    for (auto& string : compile_params) {
        param_cstring.push_back(string.c_str());
    }

    CUDAModuleWrapper* wrapper = new CUDAModuleWrapper();
    CUDAModule* cuda_module = new CUDAModule();
    cuda_param->module = (void*)cuda_module;
    auto ptx_code =
        CUDANVRTCCompile(pair<string, string>(name, source), param_cstring, options.device->GetDeviceId(), true);
    cuda_module->SetSourceCode(name, ptx_code);
    wrapper->Init(cuda_module, name, options.device);
    ModuleMap* module_map = options.cuda_module_manager->GetModule();
    module_map->emplace(pair<nodeid_t, CUDAModuleWrapper*>(node_id, move(wrapper)));

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda