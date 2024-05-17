#include "text_generator.h"

#include <string>

#include "ppl/common/log.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/runtime_builder_factory.h"
#include "ppl/nn/models/pmx/load_model_options.h"
#include "ppl/nn/models/pmx/save_model_options.h"
#endif

ppl::nn::Runtime* TextGenerator::ThreadCreateRuntime(const int32_t tid, const bool use_pmx_format) {
    auto engine = engine_list_[tid].get();

    if (use_pmx_format) {
#ifdef PPLNN_ENABLE_PMX_MODEL
        const std::string model_path = model_config_.model_dir + "/model_slice_" + std::to_string(tid) + "/model.pmx";
        auto builder = std::unique_ptr<ppl::nn::pmx::RuntimeBuilder>(ppl::nn::pmx::RuntimeBuilderFactory::Create());
        if (!builder) {
            LOG(ERROR) << "create PmxRuntimeBuilder failed.";
            return nullptr;
        }

        ppl::nn::pmx::RuntimeBuilder::Resources resources;
        resources.engines = &engine;
        resources.engine_num = 1;

        std::string external_data_dir_fix;
        ppl::nn::pmx::LoadModelOptions opt;
        auto status = builder->LoadModel(model_path.c_str(), resources, opt);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "PmxRuntimeBuilder LoadModel failed: " << ppl::common::GetRetCodeStr(status);
            return nullptr;
        }
        
        status = builder->Preprocess();
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "pmx preprocess failed: " << ppl::common::GetRetCodeStr(status);
            return nullptr;
        }

        return builder->CreateRuntime();
#else
        LOG(ERROR) << "PMX format has been disabled. please compile with PPLNN_ENABLE_PMX_MODEL";
        return nullptr;
#endif
    } else {
        const std::string model_path = model_config_.model_dir + "/model_slice_" + std::to_string(tid) + "/model.onnx";
        auto builder = std::unique_ptr<ppl::nn::onnx::RuntimeBuilder>(ppl::nn::onnx::RuntimeBuilderFactory::Create());
        if (!builder) {
            LOG(ERROR) << "create onnx builder failed.";
            return nullptr;
        }

        auto rc = builder->LoadModel(model_path.c_str());
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "load model [" << model_path << "] failed: " << ppl::common::GetRetCodeStr(rc);
            return nullptr;
        }

        ppl::nn::onnx::RuntimeBuilder::Resources resources;
        resources.engines = &engine;
        resources.engine_num = 1;

        rc = builder->SetResources(resources);
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "set resources for builder failed: " << ppl::common::GetRetCodeStr(rc);
            return nullptr;
        }

        rc = builder->Preprocess();
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "builder preprocess failed: " << ppl::common::GetRetCodeStr(rc);
            return nullptr;
        }

        return builder->CreateRuntime();
    }
}


ppl::common::RetCode TextGenerator::InitModel(
    const std::string &model_type,
    const std::string &model_dir,
    const std::string &model_param_path,
    const std::string &quant_method,
    const int32_t tensor_parallel_size,
    const bool use_pmx_format
) {
    model_config_.model_type = model_type;
    model_config_.model_dir = model_dir;
    model_config_.model_param_path = model_param_path;
    model_config_.tensor_parallel_size = tensor_parallel_size;
    model_config_.quant_method = quant_method;

    runtime_thread_pool_.Init(tensor_parallel_size);
    engine_list_.resize(tensor_parallel_size);
    runtime_thread_args_.resize(tensor_parallel_size);

    LOG(INFO) << "model_type = " << model_config_.model_type;
    LOG(INFO) << "model_dir = " << model_config_.model_dir;
    LOG(INFO) << "model_param_path = " << model_config_.model_param_path;
    LOG(INFO) << "tensor_parallel_size = " << model_config_.tensor_parallel_size;
    LOG(INFO) << "quant_method = " << model_config_.quant_method;
    LOG(INFO) << "use_pmx_format = " << use_pmx_format;

    if (!model_config_.ParseModelParam()) {
        LOG(ERROR) << "ParseModelParam failed.";
        return ppl::common::RC_OTHER_ERROR;
    }

    if (!CheckParameters()) {
        LOG(ERROR) << "CheckParameters failed.";
        return ppl::common::RC_OTHER_ERROR;
    }

    ppl::common::RetCode rc;
    rc = DoInit();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "DoInit failed";
        return rc;
    }

    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        engine_list_[tid] = std::unique_ptr<ppl::nn::Engine>(ThreadCreateEngine(tid));
        if (!engine_list_[tid]) {
            LOG(ERROR) << "create engine [" << tid << "] failed.";
            return ppl::common::RC_OTHER_ERROR;
        }
        LOG(INFO) << "Create engine [" << tid << "] success";

        runtime_thread_args_[tid].runtime = std::unique_ptr<ppl::nn::Runtime>(ThreadCreateRuntime(tid, use_pmx_format));
        if (!runtime_thread_args_[tid].runtime) {
            LOG(ERROR) << "create runtime [" << tid << "] failed.";
            return ppl::common::RC_OTHER_ERROR;
        }
        LOG(INFO) << "Create runtime [" << tid << "] success";

        ThreadExtractTensors(tid);

        return ppl::common::RC_SUCCESS;
    });

    rc = CreateSampler();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "CreateSampler failed";
        return rc;
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode TextGenerator::FinalizeModel() {
    auto rc = DestorySampler();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "DestorySampler failed";
        return rc;
    }

    RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
        auto arg = &runtime_thread_args_[tid];

        arg->runtime.reset();
        engine_list_[tid].reset();

        auto rc = ThreadFreeKVCache(tid);
        if (ppl::common::RC_SUCCESS != rc) {
            LOG(ERROR) << "FreeKVCache failed";
            return ppl::common::RC_OTHER_ERROR;
        }

        return ppl::common::RC_SUCCESS;
    });

    return DoFinalize();
}

ppl::common::RetCode TextGenerator::PrepareGeneration(
    const std::vector<Request>& requests,
    std::vector<Response> *response
)
{
    auto rc = state_.PrepareGeneration(requests, model_config_, response);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "state PrepareGeneration failed";
        return ppl::common::RC_OTHER_ERROR;
    }

    if (alloced_cache_length_ < state_.total_cache_length) {
        alloced_cache_length_ = state_.total_cache_length;
        RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
            auto arg = &runtime_thread_args_[tid];

            // set kv cache, kv scale shape
            int64_t head_dim = model_config_.hidden_dim / model_config_.num_heads;
            int64_t num_local_kv_heads = model_config_.num_kv_heads / model_config_.tensor_parallel_size;
            if (model_config_.cache_layout == 0) {
                arg->kv_cache->GetShape()->Reshape(
                    {state_.total_cache_length, model_config_.num_layers, 2, num_local_kv_heads, head_dim});
                arg->kv_scale->GetShape()->Reshape(
                    {state_.total_cache_length, model_config_.num_layers, 2, num_local_kv_heads,
                    head_dim / model_config_.cache_quant_group});
            } else if (model_config_.cache_layout == 3) {
                arg->kv_cache->GetShape()->Reshape(
                    {model_config_.num_layers, 2, num_local_kv_heads, state_.total_cache_length, head_dim});
                arg->kv_scale->GetShape()->Reshape(
                    {model_config_.num_layers, 2, num_local_kv_heads, state_.total_cache_length,
                    head_dim / model_config_.cache_quant_group});
            } else {
                LOG(ERROR) << "unknown cache_layout: " << model_config_.cache_layout;
                return ppl::common::RC_OTHER_ERROR;
            }

            auto rc = ThreadReallocKVCache(tid);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "ReallocKVCache failed";
                return ppl::common::RC_OTHER_ERROR;
            }

            arg->kv_cache->SetBufferPtr(arg->kv_cache_ptr);
            arg->kv_scale->SetBufferPtr(arg->kv_scale_ptr);

            return ppl::common::RC_SUCCESS;
        });
    }

    return ppl::common::RC_SUCCESS;
}

bool TextGenerator::ThreadSetInputTensors(const int32_t tid)
{
    auto arg = &runtime_thread_args_[tid];
    arg->token_ids->FreeBuffer();
    arg->seq_starts->FreeBuffer();
    arg->kv_starts->FreeBuffer();
    arg->cache_indices->FreeBuffer();
    arg->start_pos->FreeBuffer();
    arg->logits->FreeBuffer();

#define CHECK_COPY_TENSOR(TENSOR_NAME) \
    if (rc != ppl::common::RC_SUCCESS) { \
        LOG(ERROR) << "set " << #TENSOR_NAME << " [" << arg->TENSOR_NAME->GetName() \
                   << "] failed: " << ppl::common::GetRetCodeStr(rc); \
        return false;\
    } do {} while(0)

    ppl::common::RetCode rc;

    arg->token_ids->GetShape()->Reshape({int64_t(state_.model_io.token_ids.size())});
    rc = arg->token_ids->CopyFromHostAsync(state_.model_io.token_ids.data());
    CHECK_COPY_TENSOR(token_ids);

    arg->kv_starts->GetShape()->Reshape({state_.current_batch_size + 1});
    rc = arg->kv_starts->CopyFromHostAsync(state_.model_io.kv_starts.data());
    CHECK_COPY_TENSOR(kv_starts);

    arg->start_pos->GetShape()->Reshape({state_.current_batch_size});
    rc = arg->start_pos->CopyFromHostAsync(state_.model_io.start_pos.data());
    CHECK_COPY_TENSOR(start_pos);

    rc = arg->max_kv_len->CopyFromHostAsync(&state_.model_io.max_kv_len);
    CHECK_COPY_TENSOR(max_kv_len);

    if (model_config_.cache_mode == 0) {
        arg->cache_indices->GetShape()->Reshape({state_.current_batch_size});
    } else if (model_config_.cache_mode == 1) {
        arg->cache_indices->GetShape()->Reshape({state_.current_batch_size, state_.max_cache_pages});
    } else {
        LOG(ERROR) << "unknown cache_mode: " << model_config_.cache_mode;
        return false;
    }
    rc = arg->cache_indices->CopyFromHostAsync(state_.model_io.cache_indices.data());
    CHECK_COPY_TENSOR(cache_indices);

    arg->seq_starts->GetShape()->Reshape({state_.current_batch_size + 1});
    rc = arg->seq_starts->CopyFromHostAsync(state_.model_io.seq_starts.data());
    CHECK_COPY_TENSOR(seq_starts);

    rc = arg->decoding_batches->CopyFromHostAsync(&state_.model_io.decoding_batches);
    CHECK_COPY_TENSOR(decoding_batches);

    rc = arg->max_seq_len->CopyFromHostAsync(&state_.model_io.max_seq_len);
    CHECK_COPY_TENSOR(max_seq_len);

#undef CHECK_COPY_TENSOR

    return true;
}

ppl::common::RetCode TextGenerator::Generate(
    const std::vector<Request>& requests,
    std::vector<Response> *response,
    Profiler *optinal_profiler)
{
    if (state_.max_steps > 0) {
        Timer generate_latency_timer;
        Timer prefill_latency_timer;
        auto profiler = optinal_profiler;
        if (profiler) {
            generate_latency_timer.Tic();
            prefill_latency_timer.Tic();
        }

        bool is_prefill = true;
        state_.FirstStep(requests, model_config_, response);
        do {
            RuntimePoolRun([&](uint32_t nthr, uint32_t tid) {
                auto arg = &runtime_thread_args_[tid];

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
                if (tid == 0 && profiler && profiler->collect_statistics && (state_.current_step == 0 || state_.current_step == 1)) {
                    auto rc = arg->runtime->Configure(ppl::nn::RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG, true);
                    if (rc != ppl::common::RC_SUCCESS)
                        LOG(WARNING) << "enable kernel profiling failed: " << ppl::common::GetRetCodeStr(rc);
                }
#endif

                bool ret = ThreadSetInputTensors(tid);
                if (!ret) {
                    LOG(ERROR) << "SetInputTensor failed";
                    return ppl::common::RC_OTHER_ERROR;
                }

                auto rc = arg->runtime->Run();
                if (rc != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "model run failed";
                    return ppl::common::RC_OTHER_ERROR;
                }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
                if (tid == 0 && profiler && profiler->collect_statistics && (state_.current_step == 0 || state_.current_step == state_.max_steps - 1)) {
                    if (state_.current_step == 0) {
                        auto rc = arg->runtime->GetProfilingStatistics(&profiler->prefill_statistics);
                        if (rc != ppl::common::RC_SUCCESS) 
                            LOG(WARNING) << "get prefill kernel profiling stats failed: " << ppl::common::GetRetCodeStr(rc);
                    } else {
                        auto rc = arg->runtime->GetProfilingStatistics(&profiler->decode_statistics);
                        if (rc != ppl::common::RC_SUCCESS)
                            LOG(WARNING) << "get decode kernel profiling stats failed: " << ppl::common::GetRetCodeStr(rc);
                    }
                    auto rc = arg->runtime->Configure(ppl::nn::RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG, false);
                    if (rc != ppl::common::RC_SUCCESS)
                        LOG(WARNING) << "enable profiling failed: " << ppl::common::GetRetCodeStr(rc);
                }
#endif

                auto logits = arg->logits;
                rc = ThreadSampleArgMax(
                    tid,
                    (float*)logits->GetBufferPtr(),
                    state_.current_batch_size,
                    model_config_.vocab_size,
                    logits->GetShape()->GetDim(1),
                    state_.model_io.output.data());

                if (rc != ppl::common::RC_SUCCESS) {
                    LOG(ERROR) << "SampleArgMax failed: " << ppl::common::GetRetCodeStr(rc);
                    return ppl::common::RC_OTHER_ERROR;
                }
                return ppl::common::RC_SUCCESS;
            });

            if (is_prefill && profiler) {
                prefill_latency_timer.Toc();
                profiler->total_prefill_latency += prefill_latency_timer.GetMilliSecond();
            }
            is_prefill = false;
        } while (state_.NextStep(requests, model_config_, response));

        if (profiler) {
            generate_latency_timer.Toc();
            auto generate_latency = generate_latency_timer.GetMilliSecond();
            auto mem_usage = double(GetMemUsage()) / 1024 / 1024 / 1024;

            profiler->total_input_tokens += state_.total_input_length;
            profiler->total_output_tokens += state_.total_output_length;
            profiler->total_request_count += state_.total_batch_size;
            profiler->total_generate_latency += generate_latency;
            profiler->max_mem_usage = std::max(mem_usage, profiler->max_mem_usage);
            profiler->total_step_count += state_.max_steps;
            profiler->total_run_count++;

            LOG(INFO) << "generation_time = " << generate_latency << " ms"
                << ", mem_usage = " << mem_usage << " gib"
                << ", out_token/s = " << state_.total_output_length / (generate_latency / 1000);
        }
    } else {
        LOG(WARNING) << "max steps == 0, not thing generated.";
    }

    return ppl::common::RC_SUCCESS;
}
