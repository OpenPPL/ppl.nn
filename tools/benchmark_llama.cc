#include "sampler.h"
#include "simple_flags.h"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "ppl/common/log.h"
#include "ppl/common/cuda/cuda_env.h"
#include "ppl/common/threadpool.h"

#include "ppl/nn/runtime/options.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/engines/llm_cuda/engine_factory.h"
#include "ppl/nn/models/onnx/runtime_builder_factory.h"

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cfloat>
#include <unistd.h>
#include <cuda_runtime.h>

#ifdef PPLNN_CUDA_ENABLE_NCCL
#include <nccl.h>
#else
typedef void* ncclComm_t;
#endif

struct Profiler {
    double prefill_latency = 0;
    // std::vector<double> decode_latency = 0; // size = gen len
    std::vector<double> step_latency;
    double total_latency = 0;
    double set_intput_latency = 0;
    double mem_usage = 0; // GB
    void Reset() {
        this->prefill_latency = 0;
        this->set_intput_latency = 0;
        this->step_latency.assign(this->step_latency.size(), 0);
        this->mem_usage = 0;
    }
};

static Profiler profiling;

class ThreadPool {
private:
    ppl::common::StaticThreadPool pool_;
    std::vector<ppl::common::RetCode> retcode_;

public:
    void Init(int nthr) {
        pool_.Init(nthr);
        retcode_.resize(nthr);
    }

    void Run(const std::function<ppl::common::RetCode(uint32_t nr_threads, uint32_t thread_idx)>& f) {
        pool_.Run([&] (uint32_t nthr, uint32_t tid) {
            retcode_[tid] = f(nthr, tid);
        });
        for (auto ret : retcode_) {
            if (ret != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "exit with thread error";
                exit(-1);
            }
        }
    }
};

static ThreadPool gpu_thread_pool;

Define_bool_opt("--help", g_flag_help, false, "show these help information");
Define_string_opt("--model-type", g_flag_model_type, "", "model type");
Define_string_opt("--model-dir", g_flag_model_dir, "", "model directory");
Define_string_opt("--model-param-path", g_flag_model_param_path, "", "path of model params");
Define_uint32_opt("--tensor-parallel-size", g_flag_tensor_parallel_size, 1, "tensor parallel size");
Define_float_opt("--top-p", g_flag_top_p, 0.0, "top p");
Define_uint32_opt("--top-k", g_flag_top_k, 1, "top k");
Define_float_opt("--temperature", g_flag_temperature, 1.0, "temperature");
Define_uint32_opt("--generation-len", g_flag_generation_len, 32, "generation length");
Define_uint32_opt("--warmup-loops", g_flag_warmup_loops, 2, "warm loops");
Define_uint32_opt("--benchmark-loops", g_flag_benchmark_loops, 4, "benchmark loops");
Define_string_opt("--input-file", g_flag_input_file, "", "input file of request's token ids. no effect if --input-len is non-zero");
Define_uint32_opt("--input-len", g_flag_input_len, 0, "input length of request. default: 0(get length from input file)");
Define_uint32_opt("--batch-size", g_flag_batch_size, UINT32_MAX, "batch size");
Define_string_opt("--output-file", g_flag_output_file, "", "output file of output token ids.")
Define_string_opt("--quant-method", g_flag_quant_method, "none",
                        "llm cuda quantization mehtod, only accept "
                        "\"none\", \"online_i8i8\" and \"online_i4f16\", "
                        "default: \"none\"");
Define_string_opt("--cublas-layout-hint", g_cublas_layout_hint, "default",
                        "matrix layout hint for cublas(currently only effect int8 gemm), only accept "
                        "\"default\", \"ampere\". "
                        "default: \"default\"");

Define_bool_opt("--kernel-profiling", g_flag_kernel_profiling, true, "enable kernel profiling and print profiling info");

#ifdef PPLNN_ENABLE_KERNEL_PROFILING

static ppl::nn::ProfilingStatistics prefill_kernel_stat;
static ppl::nn::ProfilingStatistics decode_kernel_stat;

static void PrintProfilingStatistics(const ppl::nn::ProfilingStatistics& stat, int32_t run_count) {
    std::map<std::string, std::pair<double, double>> type_stat;
    std::map<std::string, int> type_count;
    char float_buf_0[128];
    char float_buf_1[128];
    // LOG(INFO) << "----- Op statistics by Node -----";
    for (auto x = stat.prof_info.begin(); x != stat.prof_info.end(); ++x) {
        auto ext_type = (x->domain == "" ? "" : x->domain + ".") + x->type;
        double time = (double)x->exec_microseconds / 1000;
        double avg_time = time / x->exec_count;
        if (type_stat.find(ext_type) == type_stat.end()) {
            type_stat[ext_type] = std::make_pair(avg_time, time);
            type_count[ext_type] = 1;
        } else {
            std::pair<double, double>& time_pair = type_stat[ext_type];
            time_pair.first += avg_time;
            time_pair.second += time;
            type_count[ext_type]++;
        }
        // sprintf(float_buf_0, "%8.4f", avg_time);
        // string temp = x->name;
        // temp.insert(temp.length(), temp.length() > 50 ? 0 : 50 - temp.length(), ' ');
        // LOG(INFO) << "Name: [" << temp << "], "
        //           << "Avg time: [" << float_buf_0 << "], "
        //           << "Exec count: [" << x->exec_count << "]";
    }
    // LOG(INFO) << "----- Op statistics by OpType -----";
    double tot_kernel_time = 0;
    for (auto it = type_stat.begin(); it != type_stat.end(); ++it) {
        tot_kernel_time += it->second.second;
    }
    for (auto it = type_stat.begin(); it != type_stat.end(); ++it) {
        sprintf(float_buf_0, "%8.4f", it->second.first);
        sprintf(float_buf_1, "%8.4f", it->second.second / tot_kernel_time * 100);
        std::string temp = it->first;
        temp.insert(temp.length(), temp.length() > 20 ? 0 : 20 - temp.length(), ' ');
        LOG(INFO) << "Type: [" << temp << "], Avg time: [" << float_buf_0 << "], Percentage: [" << float_buf_1
                  << "], Exec count [" << type_count[it->first] << "]";
    }

    // LOG(INFO) << "----- Total statistics -----";
    sprintf(float_buf_0, "%8.4f", tot_kernel_time / run_count);
    LOG(INFO) << "Run count: [" << run_count << "]";
    LOG(INFO) << "Avg kernel time: [" << float_buf_0 << "]";
    sprintf(float_buf_0, "%8.4f", tot_kernel_time);
    LOG(INFO) << "Total kernel time: [" << float_buf_0 << "]";
}
#endif

class TimingGuard final {
public:
    TimingGuard(double* res) {
        diff_millisec_ = res;
        begin_ = std::chrono::high_resolution_clock::now();
    }
    ~TimingGuard() {
        auto end = std::chrono::high_resolution_clock::now();
        *diff_millisec_ = double(std::chrono::duration_cast<std::chrono::microseconds>(end - begin_).count()) / 1000.0;
    }

private:
    double* diff_millisec_;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_;
};

struct Config {
    std::string model_type;
    std::string model_dir;
    std::string model_param_path;

    int tensor_parallel_size = 0;

    float top_p = 0;
    float top_k = 1;
    float temperature = 1;
    int generation_len = 0;

    int benchmark_loops = 0;

    std::string quant_method;
};

struct ModelConfig final {
    int hidden_dim;
    int intermediate_dim;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int vocab_size;

    float norm_eps; // not used

    int cache_quant_bit;
    int cache_quant_group;

    int cache_layout;
    int cache_mode;

    bool dynamic_batching;
    bool auto_causal;

    std::string quant_method;
};

struct ModelInput {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> seq_starts;
    std::vector<int64_t> kv_starts;
    std::vector<int64_t> cache_indices;
    int decoding_batches = 0;
    std::vector<int64_t> start_pos;
    int64_t max_seq_len = 0;
    int64_t max_kv_len = 0;

    void* kv_cache;
    void* kv_scale;

    std::vector<int64_t> first_fill_len;
};

struct WorkerThreadArg final {
    std::unique_ptr<ppl::nn::Runtime> runtime;

    void* kv_cache_mem = nullptr;
    void* kv_scale_mem = nullptr;

    ppl::nn::Tensor* token_ids;
    ppl::nn::Tensor* attn_mask;
    ppl::nn::Tensor* seq_starts;
    ppl::nn::Tensor* kv_starts;
    ppl::nn::Tensor* cache_indices;
    ppl::nn::Tensor* decoding_batches;
    ppl::nn::Tensor* start_pos;
    ppl::nn::Tensor* max_seq_len;
    ppl::nn::Tensor* max_kv_len;
    ppl::nn::Tensor* kv_cache;
    ppl::nn::Tensor* kv_scale;

    ppl::nn::Tensor* logits;
};

static void InitCudaThread() {
    gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
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

static void FinalizeCudaThread() {
    gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
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
            LOG(ERROR) << "NCCL error(code:" << (int)e << ") on " << (emsg); \
            return ppl::common::RC_OTHER_ERROR;                              \
        }                                                                    \
    } while (0);

static bool InitNccl(uint32_t tensor_parallel_size, std::vector<ncclComm_t>* nccl_comm_list) {
    nccl_comm_list->resize(tensor_parallel_size);
    ncclUniqueId uuid;
    ncclGetUniqueId(&uuid);
    gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
        NCCL_CHECK(ncclCommInitRank(&nccl_comm_list->at(tid), tensor_parallel_size, uuid, tid), "ncclCommInitRank");
        return ppl::common::RC_SUCCESS;
    });
    return true;
}

static void FinalizeNccl(uint32_t tensor_parallel_size, const std::vector<ncclComm_t>& nccl_comm_list) {
    gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
        NCCL_CHECK(ncclCommDestroy(nccl_comm_list[tid]), "ncclCommDestroy");
        return ppl::common::RC_SUCCESS;
    });
}

#endif

static ppl::nn::Engine* CreateCudaEngine(ncclComm_t nccl_comm, int device_id, const std::string& quant_method) {
    ppl::nn::llm::cuda::EngineOptions options;
    options.device_id = device_id;
    options.mm_policy = ppl::nn::llm::cuda::MM_COMPACT;

    if (quant_method == "none") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_NONE;
    } else if (quant_method == "online_i8i8") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_ONLINE_I8I8;
    } else if (quant_method == "online_i4f16") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_ONLINE_I4F16;
    } else {
        LOG(ERROR) << "unknown/unsupported --quant-method option: " << quant_method;
        return nullptr;
    }

    if (g_cublas_layout_hint == "default") {
        options.cublas_layout_hint = ppl::nn::llm::cuda::CUBLAS_LAYOUT_DEFAULT;
    } else if (g_cublas_layout_hint == "ampere") {
        options.cublas_layout_hint = ppl::nn::llm::cuda::CUBLAS_LAYOUT_AMPERE;
    } else {
        LOG(ERROR) << "unknown/unsupported --cublas-layout-hint option: " << g_cublas_layout_hint;
        return nullptr;
    }

    auto engine = std::unique_ptr<ppl::nn::Engine>(ppl::nn::llm::cuda::EngineFactory::Create(options));
    if (!engine) {
        LOG(ERROR) << "create cuda engine failed.";
        return nullptr;
    }

#ifdef PPLNN_CUDA_ENABLE_NCCL
    auto rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_SET_TP_NCCL_COMM, nccl_comm);
    if (rc != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "engine configure failed";
        return nullptr;
    }
#endif

    return engine.release();
}

static ppl::nn::Runtime* CreatePPLRuntime(ppl::nn::Engine* cuda_engine, const std::string& model_file) {
    auto builder = std::unique_ptr<ppl::nn::onnx::RuntimeBuilder>(ppl::nn::onnx::RuntimeBuilderFactory::Create());
    if (!builder) {
        LOG(ERROR) << "create onnx builder failed.";
        return nullptr;
    }

    auto rc = builder->LoadModel(model_file.c_str());
    if (rc != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "load model [" << model_file << "] failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    ppl::nn::onnx::RuntimeBuilder::Resources resources;
    resources.engines = &cuda_engine;
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

static void UpdateInputPrefill(int gen_len, ModelInput* model_input) {
    int batch_size = model_input->first_fill_len.size();
    model_input->decoding_batches = 0;

    model_input->seq_starts.reserve(batch_size + 1);
    model_input->seq_starts.push_back(0);

    model_input->kv_starts.reserve(batch_size + 1);
    model_input->kv_starts.push_back(0);

    model_input->start_pos.reserve(batch_size);

    model_input->cache_indices.reserve(batch_size);
    model_input->cache_indices.push_back(0);

    for (int i = 0; i < batch_size; ++i) {
        model_input->start_pos.push_back(0);
        model_input->seq_starts.push_back(model_input->seq_starts[i] + model_input->first_fill_len[i]);
        model_input->kv_starts.push_back(model_input->kv_starts[i] + model_input->first_fill_len[i]);
        model_input->max_seq_len = std::max(model_input->first_fill_len[i], model_input->max_seq_len);
        model_input->max_kv_len = std::max(model_input->first_fill_len[i], model_input->max_kv_len);

        if (i > 0) {
            model_input->cache_indices.push_back(model_input->cache_indices[i - 1] +
                                                 model_input->first_fill_len[i - 1] + gen_len - 1);
        }
    }
}

static void UpdateInputDecode(int step, const std::vector<int32_t>& gen_tokens, ModelInput* model_input) {
    int batch_size = model_input->first_fill_len.size();
    model_input->decoding_batches = batch_size;
    model_input->max_seq_len = 1;
    model_input->max_kv_len = model_input->max_kv_len + 1;

    model_input->token_ids.resize(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        model_input->token_ids[i] = gen_tokens.at(i);
        model_input->seq_starts[i + 1] = model_input->seq_starts[i] + 1;
        model_input->kv_starts[i + 1] = model_input->kv_starts[i] + model_input->first_fill_len[i] + step;
        if (step == 1) {
            model_input->start_pos[i] = model_input->first_fill_len[i];
        } else {
            model_input->start_pos[i]++;
        }
    }
}

static std::shared_ptr<ppl::llm::cuda::Sampler> CreateCudaSampler(ppl::nn::Runtime* runtime) {
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
        return std::shared_ptr<ppl::llm::cuda::Sampler>();
    }

    cudaStream_t stream;
    auto rc = dev->Configure(ppl::nn::llm::cuda::DEV_CONF_GET_STREAM, &stream);
    if (rc != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "Configure ppl::nn::llm::cuda::DEV_CONF_GET_STREAM failed: " << ppl::common::GetRetCodeStr(rc);
        return std::shared_ptr<ppl::llm::cuda::Sampler>();
    }

    return std::make_shared<ppl::llm::cuda::Sampler>(stream);
}

static bool CheckParameters(const ModelConfig& model_config) {
    if (model_config.auto_causal != true) {
        LOG(ERROR) << "only support auto_causal == true";
        return false;
    }

    if (model_config.cache_mode != 0) {
        LOG(ERROR) << "only support cache_mode == 0";
        return false;
    }

    if (model_config.cache_layout != 0 && model_config.cache_layout != 3) {
        LOG(ERROR) << "only support cache_layout == 0 || cache_layout == 3";
        return false;
    }

    if (model_config.cache_quant_bit != 8 && model_config.cache_quant_group != 8) {
        LOG(ERROR) << "only support cache_quant_bit == 8 and cache_quant_group == 8";
        return false;
    }

    if (model_config.dynamic_batching != true) {
        LOG(ERROR) << "only support dynamic_batching == true";
        return false;
    }

    return true;
}


class LLM {
public:
    LLM(const Config& config)
        : tensor_parallel_size_(config.tensor_parallel_size)
        , top_p_(config.top_p)
        , top_k_(config.top_k)
        , temperature_(config.temperature)
        , generation_len_(config.generation_len)
        , nccl_comm_list_(config.tensor_parallel_size)
        , engine_list_(config.tensor_parallel_size)
        , worker_thread_args_(config.tensor_parallel_size) {}

    ~LLM() {}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    void SetKernelProfiling(bool flag) {
        kernel_profiling_ = flag;
    }
#endif

    void Finalize() {
        gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
            sampler_.reset();
            worker_thread_args_[tid].runtime.reset();
            engine_list_[tid].reset();
            cudaFree(worker_thread_args_[tid].kv_cache_mem);
            cudaFree(worker_thread_args_[tid].kv_scale_mem);
            return ppl::common::RC_SUCCESS;
        });
#ifdef PPLNN_CUDA_ENABLE_NCCL
        FinalizeNccl(tensor_parallel_size_, nccl_comm_list_);
#endif
    }

    bool Init(const ModelConfig& model_config, const std::string& model_dir) {
        bool rc = CheckParameters(model_config);
        if (!rc) {
            LOG(ERROR) << "CheckParameters failed.";
            return false;
        }

        vocab_size_ = model_config.vocab_size;
        kv_cache_block_bytes_ = model_config.num_layers * 2 * model_config.num_kv_heads / tensor_parallel_size_ *
            model_config.hidden_dim / model_config.num_heads * sizeof(int8_t);
        kv_scale_block_bytes_ = model_config.num_layers * 2 * model_config.num_kv_heads / tensor_parallel_size_ *
            model_config.hidden_dim / model_config.num_heads / model_config.cache_quant_group * sizeof(int16_t);

#ifdef PPLNN_CUDA_ENABLE_NCCL
        rc = InitNccl(tensor_parallel_size_, &nccl_comm_list_);
        if (!rc) {
            LOG(ERROR) << "NCCL init failed.";
            return false;
        }
        LOG(INFO) << "Init Nccl successed";
#endif

        gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
            engine_list_[tid] = std::unique_ptr<ppl::nn::Engine>(CreateCudaEngine(nccl_comm_list_[tid], tid, model_config.quant_method));
            if (!engine_list_[tid]) {
                LOG(ERROR) << "create cuda engine [" << tid << "] failed.";
                return ppl::common::RC_OTHER_ERROR;
            }
            LOG(INFO) << "Create cuda engine [" << tid << "] success";

            const std::string model_path = model_dir + "/model_slice_" + std::to_string(tid) + "/model.onnx";
            worker_thread_args_[tid].runtime = std::unique_ptr<ppl::nn::Runtime>(CreatePPLRuntime(engine_list_[tid].get(), model_path));
            if (!worker_thread_args_[tid].runtime) {
                LOG(ERROR) << "create runtime [" << tid << "] failed.";
                return ppl::common::RC_OTHER_ERROR;
            }
            LOG(INFO) << "Create runtime [" << tid << "] success";

            if (tid == 0) {
                sampler_ = CreateCudaSampler(worker_thread_args_[0].runtime.get());
                if (!sampler_) {
                    LOG(ERROR) << "CreateCudaSampler failed";
                    return ppl::common::RC_OTHER_ERROR;
                }
                LOG(INFO) << "Create cuda sampler success";
            }

            return ppl::common::RC_SUCCESS;
        });

        return true;
    }

    bool PrepareInput(int batch_size, int kv_cache_tokens, const ModelConfig& model_config) {
        temperature_list_.resize(batch_size);
        for (size_t i = 0; i < temperature_list_.size(); ++i) {
            temperature_list_[i] = temperature_;
        }

        gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
            auto cu_ret = cudaMalloc(&worker_thread_args_[tid].kv_cache_mem, kv_cache_tokens * kv_cache_block_bytes_);
            if (cu_ret != cudaSuccess) {
                LOG(ERROR) << "alloc kv cache [" << kv_cache_tokens * kv_cache_block_bytes_
                           << "] failed: " << cudaGetErrorString(cu_ret);
                return ppl::common::RC_OTHER_ERROR;
            }
            cu_ret = cudaMalloc(&worker_thread_args_[tid].kv_scale_mem, kv_cache_tokens * kv_scale_block_bytes_);
            if (cu_ret != cudaSuccess) {
                cudaFree(worker_thread_args_[tid].kv_cache_mem);
                LOG(ERROR) << "alloc kv scale [" << kv_cache_tokens * kv_scale_block_bytes_
                           << "] failed: " << cudaGetErrorString(cu_ret);
                return ppl::common::RC_OTHER_ERROR;
            }

            // init tensor
            auto* arg = &worker_thread_args_[tid];
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

            arg->decoding_batches->SetDeviceContext(arg->runtime->GetHostDeviceContext());
            arg->max_seq_len->SetDeviceContext(arg->runtime->GetHostDeviceContext());
            arg->max_kv_len->SetDeviceContext(arg->runtime->GetHostDeviceContext());

            arg->kv_cache->SetBufferPtr(arg->kv_cache_mem);
            arg->kv_scale->SetBufferPtr(arg->kv_scale_mem);

            // set kv cache, kv scale shape
            if (model_config.cache_layout == 0) {
                arg->kv_cache->GetShape()->Reshape({(int64_t)kv_cache_tokens, model_config.num_layers, 2,
                                                    model_config.num_kv_heads / tensor_parallel_size_,
                                                    model_config.hidden_dim / model_config.num_heads});
                arg->kv_scale->GetShape()->Reshape(
                    {(int64_t)kv_cache_tokens, model_config.num_layers, 2,
                    model_config.num_kv_heads / tensor_parallel_size_,
                    model_config.hidden_dim / model_config.num_heads / model_config.cache_quant_group});
            } else if (model_config.cache_layout == 3) {
                arg->kv_cache->GetShape()->Reshape(
                    {model_config.num_layers, 2, model_config.num_kv_heads / tensor_parallel_size_,
                    (int64_t)kv_cache_tokens, model_config.hidden_dim / model_config.num_heads});
                arg->kv_scale->GetShape()->Reshape(
                    {model_config.num_layers, 2, model_config.num_kv_heads / tensor_parallel_size_,
                    (int64_t)kv_cache_tokens,
                    model_config.hidden_dim / model_config.num_heads / model_config.cache_quant_group});
            } else {
                LOG(ERROR) << "impossible status: cache_layout = [" << model_config.cache_layout << "]";
                return ppl::common::RC_OTHER_ERROR;
            }
            return ppl::common::RC_SUCCESS;
        });

        return true;
    }

    bool SetInputTensor(const ModelInput& model_input, int id, int step) {
        ppl::common::RetCode rc;
        // token ids
        // if (step < 2) {
        worker_thread_args_[id].token_ids->GetShape()->Reshape({int64_t(model_input.token_ids.size())});
        rc = worker_thread_args_[id].token_ids->CopyFromHostAsync(model_input.token_ids.data());
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "set token_ids [" << worker_thread_args_[id].token_ids->GetName()
                       << "] failed: " << ppl::common::GetRetCodeStr(rc);
            return false;
        }
        // }

        // kv_starts
        worker_thread_args_[id].kv_starts->GetShape()->Reshape({int64_t(model_input.kv_starts.size())});
        rc = worker_thread_args_[id].kv_starts->CopyFromHostAsync(model_input.kv_starts.data());
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "set kv_starts " << worker_thread_args_[id].kv_starts->GetName()
                       << " failed: " << ppl::common::GetRetCodeStr(rc);
            return false;
        }

        // start_pos
        worker_thread_args_[id].start_pos->GetShape()->Reshape({int64_t(model_input.start_pos.size())});
        rc = worker_thread_args_[id].start_pos->CopyFromHostAsync(model_input.start_pos.data());
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "set start_pos [" << worker_thread_args_[id].start_pos->GetName()
                       << "] failed: " << ppl::common::GetRetCodeStr(rc);
            return false;
        }

        // max_kv_len
        rc = worker_thread_args_[id].max_kv_len->CopyFromHostAsync(&model_input.max_kv_len);
        if (rc != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "set max_kv_len [" << worker_thread_args_[id].max_kv_len->GetName()
                       << "] failed: " << ppl::common::GetRetCodeStr(rc);
            return false;
        }

        // prefill
        if (step < 1) {
            // cache_indices
            worker_thread_args_[id].cache_indices->GetShape()->Reshape({int64_t(model_input.cache_indices.size())});
            rc = worker_thread_args_[id].cache_indices->CopyFromHostAsync(model_input.cache_indices.data());
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "set cache_indices [" << worker_thread_args_[id].cache_indices->GetName()
                           << "] failed: " << ppl::common::GetRetCodeStr(rc);
                return false;
            }
        }

        if (step < 2) {
            // seq_start
            // LOG(INFO) << "model_input.seq_starts: ";
            // PrintVector(model_input.seq_starts);
            worker_thread_args_[id].seq_starts->GetShape()->Reshape({int64_t(model_input.seq_starts.size())});
            rc = worker_thread_args_[id].seq_starts->CopyFromHostAsync(model_input.seq_starts.data());
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "set seq_starts [" << worker_thread_args_[id].seq_starts->GetName()
                           << "] failed: " << ppl::common::GetRetCodeStr(rc);
                return false;
            }

            // decoding batches
            rc = worker_thread_args_[id].decoding_batches->CopyFromHostAsync(&model_input.decoding_batches);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "set decoding_batches [" << worker_thread_args_[id].decoding_batches->GetName()
                           << "] failed: " << ppl::common::GetRetCodeStr(rc);
                return false;
            }

            // max_seq_len
            rc = worker_thread_args_[id].max_seq_len->CopyFromHostAsync(&model_input.max_seq_len);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "set max_seq_len [" << worker_thread_args_[id].max_seq_len->GetName()
                           << "] failed: " << ppl::common::GetRetCodeStr(rc);
                return false;
            }
        }

        // rc = worker_thread_args_[id].runtime->Synchronize();
        // if (rc != ppl::common::RC_SUCCESS) {
        //     LOG(ERROR) << "set input tensor synchronize fail";
        //     return false;
        // }
        return true;
    }

    void Generate(ModelInput* model_input, std::vector<std::vector<int32_t>>* output_tokens) {
        int batch_size = model_input->first_fill_len.size();

        double step_latency = 0;
        for (int step = 0; step < generation_len_; ++step) {
            {
                TimingGuard __timing__(&step_latency);
                if (step == 0) {
                    UpdateInputPrefill(generation_len_, model_input);
                } else {
                    UpdateInputDecode(step, output_tokens->at(step - 1), model_input);
                }

                gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
                    if (tid == 0 && kernel_profiling_ && (step == 0 || step == 1)) {
                        auto rc = worker_thread_args_[tid].runtime->Configure(ppl::nn::RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG, true);
                        if (rc != ppl::common::RC_SUCCESS) {
                            LOG(WARNING) << "enable kernel profiling failed: " << ppl::common::GetRetCodeStr(rc);
                        }
                    }
#endif
                    bool ret = SetInputTensor(*model_input, tid, step);
                    if (!ret) {
                        LOG(ERROR) << "SetInputTensor failed";
                        return ppl::common::RC_OTHER_ERROR;
                    }

                    auto rc = worker_thread_args_[tid].runtime->Run();
                    if (rc != ppl::common::RC_SUCCESS) {
                        LOG(ERROR) << "model run failed";
                        return ppl::common::RC_OTHER_ERROR;
                    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
                    if (tid == 0 && kernel_profiling_ && (step == 0 || step == generation_len_ - 1)) {
                        if (step == 0) {
                            auto rc = worker_thread_args_[tid].runtime->GetProfilingStatistics(&prefill_kernel_stat);
                            if (rc != ppl::common::RC_SUCCESS) {
                                LOG(WARNING) << "get prefill kernel profiling stats failed: " << ppl::common::GetRetCodeStr(rc);
                            }
                        } else {
                            auto rc = worker_thread_args_[tid].runtime->GetProfilingStatistics(&decode_kernel_stat);
                            if (rc != ppl::common::RC_SUCCESS) {
                                LOG(WARNING) << "get decode kernel profiling stats failed: " << ppl::common::GetRetCodeStr(rc);
                            }
                        }
                        auto rc = worker_thread_args_[tid].runtime->Configure(ppl::nn::RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG, false);
                        if (rc != ppl::common::RC_SUCCESS) {
                            LOG(WARNING) << "enable profiling failed: " << ppl::common::GetRetCodeStr(rc);
                        }
                    }
#endif
                    if (tid == 0) {
                        auto logits = worker_thread_args_[tid].logits;
                        auto rc = sampler_->SampleTopPTopK((float*)logits->GetBufferPtr(), temperature_list_.data(), batch_size,
                                                        vocab_size_, logits->GetShape()->GetDim(1), top_p_, top_k_, output_tokens->at(step).data());

                        if (rc != ppl::common::RC_SUCCESS) {
                            LOG(ERROR) << "SampleTopPTopK failed: " << ppl::common::GetRetCodeStr(rc);
                            return ppl::common::RC_OTHER_ERROR;
                        }
                    }
                    return ppl::common::RC_SUCCESS;
                });
            }

            profiling.step_latency[step] += step_latency;
        }
    }

private:
    int tensor_parallel_size_ = 0;

    float top_p_ = 0;
    float top_k_ = 1;
    float temperature_ = 1;
    int generation_len_ = 0;

    std::vector<ncclComm_t> nccl_comm_list_;
    std::vector<std::unique_ptr<ppl::nn::Engine>> engine_list_;
    std::vector<WorkerThreadArg> worker_thread_args_;

    std::vector<float> temperature_list_;
    std::shared_ptr<ppl::llm::cuda::Sampler> sampler_;
    int vocab_size_ = 0;
    uint64_t kv_cache_block_bytes_ = 0;
    uint64_t kv_scale_block_bytes_ = 0;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    bool kernel_profiling_ = false;
#endif
};

static void ParseConfig(Config* config) {
    config->model_type = g_flag_model_type;
    config->model_dir = g_flag_model_dir;
    config->model_param_path = g_flag_model_param_path;
    config->tensor_parallel_size = g_flag_tensor_parallel_size;
    config->top_p = g_flag_top_p;
    config->top_k = g_flag_top_k;
    config->temperature = g_flag_temperature;
    config->generation_len = g_flag_generation_len;
    config->benchmark_loops = g_flag_benchmark_loops;
    config->quant_method = g_flag_quant_method;

    LOG(INFO) << "config.model_type: " << config->model_type;
    LOG(INFO) << "config.model_dir: " << config->model_dir;
    LOG(INFO) << "config.model_param_path: " << config->model_param_path;

    LOG(INFO) << "config.tensor_parallel_size: " << config->tensor_parallel_size;

    LOG(INFO) << "config.top_k: " << config->top_k;
    LOG(INFO) << "config.top_p: " << config->top_p;
    LOG(INFO) << "config.temperature: " << config->temperature;
    LOG(INFO) << "config.generation_len: " << config->generation_len;

    LOG(INFO) << "config.benchmark_loops: " << config->benchmark_loops;

    LOG(INFO) << "config.quant_method: " << config->quant_method;
}

static bool ParseModelConfig(const std::string& model_param_path, ModelConfig* model_config) {
    std::ifstream ifs(model_param_path);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document document;
    if (document.ParseStream(isw) == false) {
        LOG(ERROR) << "ParseStream failed";
        return false;
    }
    document.ParseStream(isw);

    auto it = document.FindMember("num_heads");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [num_heads] failed";
        return false;
    }
    model_config->num_heads = it->value.GetInt();

    it = document.FindMember("num_kv_heads");
    if (it == document.MemberEnd()) {
        model_config->num_kv_heads = model_config->num_heads;
    } else {
        model_config->num_kv_heads = it->value.GetInt();
    }

    it = document.FindMember("num_layers");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [num_layers] failed";
        return false;
    }
    model_config->num_layers = it->value.GetInt();

    it = document.FindMember("hidden_dim");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [hidden_dim] failed";
        return false;
    }
    model_config->hidden_dim = it->value.GetInt();

    it = document.FindMember("intermediate_dim");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [intermediate_dim] failed";
        return false;
    }
    model_config->intermediate_dim = it->value.GetInt();

    it = document.FindMember("vocab_size");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [vocab_size] failed";
        return false;
    }
    model_config->vocab_size = it->value.GetInt();

    it = document.FindMember("cache_quant_bit");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_quant_bit] failed";
        return false;
    }
    model_config->cache_quant_bit = it->value.GetInt();

    it = document.FindMember("cache_quant_group");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_quant_group] failed";
        return false;
    }
    model_config->cache_quant_group = it->value.GetInt();

    it = document.FindMember("cache_layout");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_layout] failed";
        return false;
    }
    model_config->cache_layout = it->value.GetInt();

    it = document.FindMember("cache_mode");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_mode] failed";
        return false;
    }
    model_config->cache_mode = it->value.GetInt();

    it = document.FindMember("dynamic_batching");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [dynamic_batching] failed";
        return false;
    }
    model_config->dynamic_batching = it->value.GetBool();

    it = document.FindMember("auto_causal");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [auto_causal] failed";
        return false;
    }
    model_config->auto_causal = it->value.GetBool();

    LOG(INFO) << "model_config.num_layers: " << model_config->num_layers;
    LOG(INFO) << "model_config.num_heads: " << model_config->num_heads;
    LOG(INFO) << "model_config.num_kv_heads: " << model_config->num_kv_heads;
    LOG(INFO) << "model_config.hidden_dim: " << model_config->hidden_dim;
    LOG(INFO) << "model_config.intermediate_dim: " << model_config->intermediate_dim;
    LOG(INFO) << "model_config.vocab_size: " << model_config->vocab_size;

    LOG(INFO) << "model_config.cache_quant_bit: " << model_config->cache_quant_bit;
    LOG(INFO) << "model_config.cache_quant_group: " << model_config->cache_quant_group;
    LOG(INFO) << "model_config.cache_layout: " << model_config->cache_layout;
    LOG(INFO) << "model_config.cache_mode: " << model_config->cache_mode;

    LOG(INFO) << "model_config.dynamic_batching: " << model_config->dynamic_batching;
    LOG(INFO) << "model_config.auto_causal: " << model_config->auto_causal;

    return true;
}

static bool WriteOutput(const std::string& token_file, const std::vector<std::vector<int32_t>> &output_tokens) {
    std::ofstream fout(token_file, std::ios::out);
    if (!fout.is_open()) {
        LOG(ERROR) << "Error Openning " << token_file;
        return false;
    }

    for (size_t b = 0; b < output_tokens[0].size(); ++b) {
        for (size_t l = 0; l < output_tokens.size(); ++l) {
            fout << output_tokens[l][b];
            if (l + 1 < output_tokens.size())
                fout << ", ";
        }
        fout << std::endl;
    }
    return true;
}

static bool ParseInput(const std::string& token_file, ModelInput* model_input) {
    std::ifstream fin(token_file, std::ios::in);
    if (!fin.is_open()) {
        LOG(ERROR) << "Error Openning " << token_file;
        return false;
    }

    std::string line;
    uint32_t line_cnt = 0;
    while (std::getline(fin, line) && line_cnt < g_flag_batch_size) {
        std::stringstream line_stream(line);
        if (line.empty()) {
            continue;
        }
        std::string vals;
        model_input->first_fill_len.push_back(0);
        // each request
        while (std::getline(line_stream, vals, ',')) {
            model_input->token_ids.push_back(std::stoi(vals));
            ++(model_input->first_fill_len.back());
        }
        line_cnt++;
    }
    return true;
}

static void GenInput(ModelInput* model_input) {
    model_input->first_fill_len.assign(g_flag_batch_size, g_flag_input_len);
    model_input->token_ids.assign((int64_t)g_flag_batch_size * g_flag_input_len, 1);
}

int main(int argc, char* argv[]) {
    simple_flags::parse_args(argc, argv);

    if (g_flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    if (!simple_flags::get_unknown_flags().empty()) {
        std::string content;
        for (auto it : simple_flags::get_unknown_flags()) {
            content += "'" + it + "', ";
        }
        content.resize(content.size() - 2); // remove last ', '
        content.append(".");
        LOG(ERROR) << "unknown option(s): " << content.c_str();
        simple_flags::print_args_info();
        return -1;
    }

    LOG(INFO) << "ppl.nn version: [" << PPLNN_VERSION_MAJOR << "." << PPLNN_VERSION_MINOR << "." << PPLNN_VERSION_PATCH
              << "], commit: [" << PPLNN_COMMIT_STR << "]";

    Config config;
    ParseConfig(&config);

    ModelConfig model_config;
    if (!ParseModelConfig(config.model_param_path, &model_config)) {
        LOG(ERROR) << "PaseModelConfig failed, model_param_path: " << config.model_param_path;
        return -1;
    }
    model_config.quant_method = config.quant_method;

    gpu_thread_pool.Init(config.tensor_parallel_size);
    InitCudaThread();

    LOG(INFO) << "input_file: " << g_flag_input_file;
    LOG(INFO) << "input_length: " << g_flag_input_len;
    LOG(INFO) << "batch_size: " << g_flag_batch_size;

    ModelInput raw_model_input;
    if (g_flag_input_len == 0) {
        if (!ParseInput(g_flag_input_file, &raw_model_input)) {
            LOG(ERROR) << "ParseInput failed, input file: " << g_flag_input_file; 
            return -1;
        }
    } else {
        GenInput(&raw_model_input);
    }
    int64_t batch_size = raw_model_input.first_fill_len.size();
    int64_t total_input_length = 0;
    int64_t kv_cache_length = 0;
    for (auto input_length :  raw_model_input.first_fill_len) {
        total_input_length += input_length;
        kv_cache_length += input_length + config.generation_len - 1;
    }

    profiling.step_latency.resize(config.generation_len);

    LLM llm(config);
    bool ret = llm.Init(model_config, config.model_dir);
    if (!ret) {
        LOG(ERROR) << "Init failed";
        return -1;
    }

    ret = llm.PrepareInput(batch_size, kv_cache_length, model_config);
    if (!ret) {
        LOG(ERROR) << "PrepareInput failed";
        return -1;
    }
    LOG(INFO) << "PrepareInput success";

    std::vector<std::vector<int32_t>> output_tokens(config.generation_len, std::vector<int32_t>(batch_size));

    LOG(INFO) << "Request batch size: " << batch_size;
    LOG(INFO) << "Total input length: " << total_input_length;
    LOG(INFO) << "KV cache length: " << kv_cache_length;

    // warmup
    for (uint32_t i = 0; i < g_flag_warmup_loops; ++i) {
        LOG(INFO) << "Warmup " << i;
        ModelInput model_input = raw_model_input;
        double latency = 0;
        {
            TimingGuard __timeing__(&latency);
            llm.Generate(&model_input, &output_tokens);
        }
        LOG(INFO) << "Time " << latency << " ms";
    }

    profiling.Reset();
    for (int i = 0; i < config.benchmark_loops; ++i) {
        LOG(INFO) << "Benchmark " << i;
        ModelInput model_input = raw_model_input;
        double latency = 0;
        {
            TimingGuard __timeing__(&latency);
            llm.Generate(&model_input, &output_tokens);
        }
        profiling.total_latency += latency;
        LOG(INFO) << "Time " << latency << " ms";
    }
    size_t avail_bytes = 0, total = 0;
    gpu_thread_pool.Run([&](uint32_t nthr, uint32_t tid) {
        if (tid == 0)
            cudaMemGetInfo(&avail_bytes, &total);
        return ppl::common::RC_SUCCESS;
    });
    profiling.mem_usage = double(total - avail_bytes) / 1024 / 1024 / 1024;


    // profiling 结果
    double avg_prefill_latency = 0;
    double max_decode_latency = 0;
    double min_decode_latency = DBL_MAX;
    double avg_decode_latency = 0;
    double avg_step_latency = 0;
    for (size_t step = 0; step < profiling.step_latency.size(); ++step) {
        if (step > 0) {
            avg_decode_latency += profiling.step_latency[step];
            max_decode_latency = std::max(max_decode_latency, profiling.step_latency[step]);
            min_decode_latency = std::min(min_decode_latency, profiling.step_latency[step]);
        }
        avg_step_latency += profiling.step_latency[step];
    }

    int max_latency_step = std::max_element(profiling.step_latency.begin() + 1, profiling.step_latency.end()) -
        profiling.step_latency.begin();
    int min_latency_step = std::min_element(profiling.step_latency.begin() + 1, profiling.step_latency.end()) -
        profiling.step_latency.begin();

    avg_prefill_latency = profiling.step_latency[0] / config.benchmark_loops;
    avg_decode_latency = avg_decode_latency / (config.benchmark_loops * (config.generation_len - 1));
    min_decode_latency = min_decode_latency / config.benchmark_loops;
    max_decode_latency = max_decode_latency / config.benchmark_loops;
    avg_step_latency = avg_step_latency / (config.benchmark_loops * config.generation_len);
    double tokens_per_second = 1000 / avg_step_latency * batch_size;

    LOG(INFO) << "Memory usage(GB): " << profiling.mem_usage;
    LOG(INFO) << "Prefill latency(ms): " << avg_prefill_latency;
    LOG(INFO) << "Min decode latency(ms)[" << min_latency_step << "]: " << min_decode_latency;
    LOG(INFO) << "Max decode latency(ms)[" << max_latency_step << "]: " << max_decode_latency;
    LOG(INFO) << "Avg decode latency(ms): " << avg_decode_latency;
    LOG(INFO) << "Avg step latency(ms): " << avg_step_latency;
    LOG(INFO) << "Tokens per second: " << tokens_per_second;

    LOG(INFO) << "CSV format header:prefill(ms),decode(ms),avg(ms),tps(ms),mem(gib)";
    LOG(INFO) << "CSV format output:" << avg_prefill_latency << ","
        << avg_decode_latency << ","
        << avg_step_latency << ","
        << tokens_per_second << ","
        << profiling.mem_usage;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    if (g_flag_kernel_profiling) {
        LOG(INFO) << "Kernel profiling";
        ModelInput model_input = raw_model_input;
        llm.SetKernelProfiling(true);
        double latency = 0;
        {
            TimingGuard __timeing__(&latency);
            llm.Generate(&model_input, &output_tokens);
        }
        LOG(INFO) << "Time " << latency << " ms";
        LOG(INFO) << "----- Prefill statistics -----";
        PrintProfilingStatistics(prefill_kernel_stat, 1);
        LOG(INFO) << "----- Decode statistics -----";
        PrintProfilingStatistics(decode_kernel_stat, config.generation_len - 1);
    }
#endif

    if (!g_flag_output_file.empty()) {
        WriteOutput(g_flag_output_file, output_tokens);
    }

    llm.Finalize();

    FinalizeCudaThread();

    return 0;
}
