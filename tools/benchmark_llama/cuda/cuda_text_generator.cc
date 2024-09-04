#include "cuda_text_generator.h"

#include "ppl/nn/engines/llm_cuda/engine_factory.h"
#include "ppl/common/cuda/cuda_env.h"

#include <cuda_runtime.h>

bool CudaTextGenerator::CheckParameters() {
    if (model_config_.auto_causal != true) {
        LOG(ERROR) << "only support auto_causal == true";
        return false;
    }

    if (model_config_.cache_layout != 0 && model_config_.cache_layout != 3) {
        LOG(ERROR) << "only support cache_layout == 0 || cache_layout == 3";
        return false;
    }

    if (model_config_.cache_quant_bit != 8 && model_config_.cache_quant_group != 8) {
        LOG(ERROR) << "only support cache_quant_bit == 8 and cache_quant_group == 8";
        return false;
    }

    if (model_config_.dynamic_batching != true) {
        LOG(ERROR) << "only support dynamic_batching == true";
        return false;
    }

    return true;
}

void CudaTextGenerator::InitCudaThread() {
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        auto cu_ret = cudaSetDevice(tid);
        if (cu_ret != cudaSuccess) {
            LOG(ERROR) << "cudaSetDevice(" << tid << ") failed: " << cudaGetErrorString(cu_ret);
            return ppl::common::RC_OTHER_ERROR;
        }
        auto rc = ppl::common::InitCudaEnv(tid);
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "InitCudaEnv(" << tid << ") failed: " << ppl::common::GetRetCodeStr(rc);
            return ppl::common::RC_OTHER_ERROR;
        }
        return ppl::common::RC_SUCCESS;
    });
}

void CudaTextGenerator::FinalizeCudaThread() {
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        auto rc = ppl::common::DestroyCudaEnv(tid);
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "InitCudaEnv(" << tid << ") failed: " << ppl::common::GetRetCodeStr(rc);
            return ppl::common::RC_OTHER_ERROR;
        }
        return ppl::common::RC_SUCCESS;
    });
}

#ifdef PPLNN_CUDA_ENABLE_NCCL
#define NCCL_CHECK(cmd, emsg)                                                \
    do {                                                                     \
        ncclResult_t e = (cmd);                                              \
        if (e != ncclSuccess) {                                              \
            LOG(ERROR) << "NCCL error(code:" << (int32_t)e << ") on " << (emsg); \
            return ppl::common::RC_OTHER_ERROR;                              \
        }                                                                    \
    } while (0);

void CudaTextGenerator::InitNccl() {
    nccl_comm_list_.resize(model_config_.tensor_parallel_size);
    ncclUniqueId uuid;
    ncclGetUniqueId(&uuid);
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        NCCL_CHECK(ncclCommInitRank(&nccl_comm_list_.at(tid), model_config_.tensor_parallel_size, uuid, tid), "ncclCommInitRank");
        return ppl::common::RC_SUCCESS;
    });
}

void CudaTextGenerator::FinalizeNccl() {
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        NCCL_CHECK(ncclCommDestroy(nccl_comm_list_[tid]), "ncclCommDestroy");
        return ppl::common::RC_SUCCESS;
    });
}

#endif

ppl::common::RetCode CudaTextGenerator::DoInit() {
    InitCudaThread();
#ifdef PPLNN_CUDA_ENABLE_NCCL
    InitNccl();
#endif

    host_device_list_.resize(model_config_.tensor_parallel_size);
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        host_device_list_[tid].reset(
            ppl::nn::llm::cuda::EngineFactory::CreateHostDeviceContext(
                ppl::nn::llm::cuda::HostDeviceOptions()));
        return ppl::common::RC_SUCCESS;
    });

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode CudaTextGenerator::DoFinalize() {
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        host_device_list_[tid].reset();
        return ppl::common::RC_SUCCESS;
    });
#ifdef PPLNN_CUDA_ENABLE_NCCL
    FinalizeNccl();
#endif
    FinalizeCudaThread();

    return ppl::common::RC_SUCCESS;
}

ppl::nn::Engine* CudaTextGenerator::ThreadCreateEngine(const int32_t tid) {
    ppl::nn::llm::cuda::EngineOptions options;
    options.device_id = tid;
    options.mm_policy = ppl::nn::llm::cuda::MM_COMPACT;

    if (model_config_.quant_method == "none") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_NONE;
    } else if (model_config_.quant_method == "online_i8i8") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_ONLINE_I8I8;
    } else if (model_config_.quant_method == "online_f8f8") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_ONLINE_F8F8;
    } else if (model_config_.quant_method == "online_i4f16") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_ONLINE_I4F16;
    } else {
        LOG(ERROR) << "unknown/unsupported --quant-method option: " << model_config_.quant_method;
        return nullptr;
    }

    if (construct_options_.cublas_layout_hint == "default") {
        options.cublas_layout_hint = ppl::nn::llm::cuda::CUBLAS_LAYOUT_DEFAULT;
    } else if (construct_options_.cublas_layout_hint == "ampere") {
        options.cublas_layout_hint = ppl::nn::llm::cuda::CUBLAS_LAYOUT_AMPERE;
    } else {
        LOG(ERROR) << "unknown/unsupported --cublas-layout-hint option: " << construct_options_.cublas_layout_hint;
        return nullptr;
    }

    auto engine = std::unique_ptr<ppl::nn::Engine>(ppl::nn::llm::cuda::EngineFactory::Create(options));
    if (!engine) {
        LOG(ERROR) << "create cuda engine failed.";
        return nullptr;
    }

    ppl::common::RetCode rc;
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_CACHE_PREFILL, construct_options_.enable_cache_prefill ? 1 : 0);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_CACHE_PREFILL failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_SHM_MHA, construct_options_.disable_decoding_shm_mha ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_SHM_MHA failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_INF_MHA, construct_options_.disable_decoding_inf_mha ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_INF_MHA failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_INF_GQA, construct_options_.disable_decoding_inf_gqa ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_INF_GQA failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_ATTN_SPLIT_K, construct_options_.configure_decoding_attn_split_k);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_ATTN_SPLIT_K failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_ATTN_TPB, construct_options_.specify_decoding_attn_tpb);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_ATTN_TPB failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_GRAPH_FUSION, construct_options_.disable_graph_fusion ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_GRAPH_FUSION failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

#ifdef PPLNN_CUDA_ENABLE_NCCL
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_SET_TP_NCCL_COMM, nccl_comm_list_[tid]);
    if (rc != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "engine configure failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }
#endif

    return engine.release();
}

void CudaTextGenerator::ThreadExtractTensors(const int32_t tid) {
    auto* arg = &runtime_thread_args_[tid];
    arg->token_ids = arg->runtime->GetInputTensor(0);
    arg->attn_mask = arg->runtime->GetInputTensor(1);
    arg->seq_starts = arg->runtime->GetInputTensor(2);
    arg->kv_starts = arg->runtime->GetInputTensor(3);
    arg->cache_indices = arg->runtime->GetInputTensor(4);
    arg->decoding_batches = arg->runtime->GetInputTensor(5);
    arg->start_pos = arg->runtime->GetInputTensor(6);
    arg->max_seq_len = arg->runtime->GetInputTensor(7);
    arg->max_kv_len = arg->runtime->GetInputTensor(8);
    arg->kv_cache = arg->runtime->GetInputTensor(9);
    arg->kv_scale = arg->runtime->GetInputTensor(10);

    arg->logits = arg->runtime->GetOutputTensor(0);

    arg->decoding_batches->SetDeviceContext(host_device_list_[tid].get());
    arg->max_seq_len->SetDeviceContext(host_device_list_[tid].get());
    arg->max_kv_len->SetDeviceContext(host_device_list_[tid].get());
}

ppl::common::RetCode CudaTextGenerator::ThreadReallocKVCache(const int32_t tid) {
    auto arg = &runtime_thread_args_[tid];

    if (arg->kv_cache_ptr != nullptr) {
        auto cu_ret = cudaFree(arg->kv_cache_ptr);
        if (cu_ret != cudaSuccess) {
            LOG(ERROR) << "free kv cache failed: " << cudaGetErrorString(cu_ret);
            return ppl::common::RC_OTHER_ERROR;
        }
        cu_ret = cudaFree(arg->kv_scale_ptr);
        if (cu_ret != cudaSuccess) {
            LOG(ERROR) << "free kv scale failed: " << cudaGetErrorString(cu_ret);
            return ppl::common::RC_OTHER_ERROR;
        }
    }

    {
        auto cu_ret = cudaMalloc(&arg->kv_cache_ptr, arg->kv_cache->GetShape()->CalcBytesIncludingPadding());
        if (cu_ret != cudaSuccess) {
            LOG(ERROR) << "alloc kv cache [" << arg->kv_cache->GetShape()->CalcBytesIncludingPadding()
                    << "] failed: " << cudaGetErrorString(cu_ret);
            return ppl::common::RC_OTHER_ERROR;
        }
        cu_ret = cudaMalloc(&arg->kv_scale_ptr, arg->kv_scale->GetShape()->CalcBytesIncludingPadding());
        if (cu_ret != cudaSuccess) {
            cudaFree(arg->kv_cache_ptr);
            arg->kv_cache_ptr = nullptr;
            LOG(ERROR) << "alloc kv scale [" << arg->kv_scale->GetShape()->CalcBytesIncludingPadding()
                    << "] failed: " << cudaGetErrorString(cu_ret);
            return ppl::common::RC_OTHER_ERROR;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode CudaTextGenerator::ThreadFreeKVCache(const int32_t tid) {
    auto arg = &runtime_thread_args_[tid];

    if (arg->kv_cache_ptr != nullptr) {
        auto cu_ret = cudaFree(arg->kv_cache_ptr);
        if (cu_ret != cudaSuccess) {
            LOG(ERROR) << "free kv cache failed: " << cudaGetErrorString(cu_ret);
            return ppl::common::RC_OTHER_ERROR;
        }
        cu_ret = cudaFree(arg->kv_scale_ptr);
        if (cu_ret != cudaSuccess) {
            LOG(ERROR) << "free kv scale failed: " << cudaGetErrorString(cu_ret);
            return ppl::common::RC_OTHER_ERROR;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode CudaTextGenerator::CreateSampler() {
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        if (tid == 0) {
            auto arg = &runtime_thread_args_[tid];
            auto runtime = arg->runtime.get();

            ppl::nn::DeviceContext::Type needed_type;
            *((int64_t*)needed_type.str) = 0;
            needed_type.str[0] = 'c';
            needed_type.str[1] = 'u';
            needed_type.str[2] = 'd';
            needed_type.str[3] = 'a';

            ppl::nn::DeviceContext* dev = nullptr;
            for (uint32_t i = 0; i < runtime->GetDeviceContextCount(); ++i) {
                if (runtime->GetDeviceContext(i)->GetType() == needed_type) {
                    dev = runtime->GetDeviceContext(i);
                    break;
                }
            }

            if (!dev) {
                LOG(ERROR) << "cannot find cuda device in runtime.";
                return ppl::common::RC_NOT_FOUND;
            }

            cudaStream_t stream;
            auto rc = dev->Configure(ppl::nn::llm::cuda::DEV_CONF_GET_STREAM, &stream);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "Configure ppl::nn::llm::cuda::DEV_CONF_GET_STREAM failed: " << ppl::common::GetRetCodeStr(rc);
                return ppl::common::RC_OTHER_ERROR;
            }

            sampler_.reset(new CudaSampler());
            rc = sampler_->Init(stream);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "cuda sampler Init failed: " << ppl::common::GetRetCodeStr(rc);
                return ppl::common::RC_OTHER_ERROR;
            }
        }
        return ppl::common::RC_SUCCESS;
    });

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode CudaTextGenerator::DestorySampler() {
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        if (tid == 0) {
            auto rc = sampler_->Clear();
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "cuda sampler Clear failed: " << ppl::common::GetRetCodeStr(rc);
                return ppl::common::RC_OTHER_ERROR;
            }
            sampler_.reset();
        }
        return ppl::common::RC_SUCCESS;
    });

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode CudaTextGenerator::ThreadSampleArgMax(
        const int32_t tid,
        const float* logits,
        const int32_t batch,
        const int32_t vocab_size,
        const int32_t batch_stride,
        int32_t* output)
{
    if (tid == 0) {
        auto rc = sampler_->SampleArgMax(
            logits,
            batch,
            vocab_size,
            batch_stride,
            output
        );
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "cuda sampler SampleArgMax failed: " << ppl::common::GetRetCodeStr(rc);
            return ppl::common::RC_OTHER_ERROR;
        }
    }
    return ppl::common::RC_SUCCESS;
}

uint64_t CudaTextGenerator::GetMemUsage() {
    size_t avail_bytes = 0, total = 0;
    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        if (tid == 0)
            cudaMemGetInfo(&avail_bytes, &total);
        return ppl::common::RC_SUCCESS;
    });
    return total - avail_bytes;
}
