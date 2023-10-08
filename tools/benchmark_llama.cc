#include "sampler.h"
#include "simple_flags.h"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "nccl.h"
#include "ppl/common/log.h"
using namespace ppl::common;

#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/engines/llm_cuda/engine_factory.h"
#include "ppl/nn/models/onnx/runtime_builder_factory.h"
using namespace ppl::nn;

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <cuda_runtime.h>
#include <omp.h>
using namespace std;

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

Define_string_opt("--model_type", g_flag_model_type, "", "model type");
Define_string_opt("--model_dir", g_flag_model_dir, "", "model directory");
Define_string_opt("--model_param_path", g_flag_model_param_path, "", "path of model params");
Define_uint32_opt("--tensor_parallel_size", g_flag_tensor_parallel_size, 1, "tensor parallel size");
Define_float_opt("--top_p", g_flag_top_p, 0.0, "top p");
Define_uint32_opt("--top_k", g_flag_top_k, 1, "top k");
Define_float_opt("--temperature", g_flag_temperature, 1.0, "temperature");
Define_uint32_opt("--generation_len", g_flag_generation_len, 32, "generation length");
Define_uint32_opt("--warmup_loops", g_flag_warmup_loops, 2, "warm loops");
Define_uint32_opt("--benchmark_loops", g_flag_benchmark_loops, 4, "benchmark loops");
Define_string_opt("--input_file", g_flag_input_file, "", "input file of token ids");
Define_uint32_opt("--batch_size", g_flag_batch_size, UINT32_MAX, "batch size");
Define_string_opt("--quant-method", g_flag_quant_method, "none",
                        "llm cuda quantization mehtod, only accept "
                        "\"none\" and \"online_i8i8\", "
                        "default: \"none\"");

template <class T>
static void PrintVector(vector<T> vec) {
    for (auto& ele : vec) {
        std::cout << ele << ", ";
    }
    std::cout << std::endl;
}

class TimingGuard final {
public:
    TimingGuard(double* res) {
        diff_microsec_ = res;
        begin_ = std::chrono::high_resolution_clock::now();
    }
    ~TimingGuard() {
        auto end = std::chrono::high_resolution_clock::now();
        *diff_microsec_ = double(std::chrono::duration_cast<std::chrono::microseconds>(end - begin_).count()) / 1000.0;
    }

private:
    double* diff_microsec_;
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
    std::string input_file;

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
    std::unique_ptr<Runtime> runtime;

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

#define NCCL_CHECK(cmd, emsg)                                                \
    do {                                                                     \
        ncclResult_t e = (cmd);                                              \
        if (e != ncclSuccess) {                                              \
            LOG(ERROR) << "NCCL error(code:" << (int)e << ") on " << (emsg); \
            return RC_OTHER_ERROR;                                           \
        }                                                                    \
    } while (0);

static RetCode InitNccl(uint32_t tensor_parallel_size, std::vector<ncclComm_t>* nccl_comm_list) {
    nccl_comm_list->resize(tensor_parallel_size);
    std::vector<int> dev_list(tensor_parallel_size);
    for (size_t i = 0; i < tensor_parallel_size; ++i) {
        dev_list[i] = i;
    }
    NCCL_CHECK(ncclCommInitAll(nccl_comm_list->data(), tensor_parallel_size, dev_list.data()), "ncclCommInitAll");
    return RC_SUCCESS;
}

static Engine* CreateCudaEngine(ncclComm_t nccl_comm, int device_id) {
    ppl::nn::llm::cuda::EngineOptions options;
    options.device_id = device_id;
    options.mm_policy = ppl::nn::llm::cuda::MM_COMPACT;

    if (g_flag_quant_method == "none") {
        options.quant_method = llm::cuda::QUANT_METHOD_NONE;
    } else if (g_flag_quant_method == "online_i8i8") {
        options.quant_method = llm::cuda::QUANT_METHOD_ONLINE_I8I8;
    } else {
        LOG(ERROR) << "unknown/unsupported --quant-method option: " << g_flag_quant_method;
        return nullptr;
    }

    auto engine = unique_ptr<Engine>(ppl::nn::llm::cuda::EngineFactory::Create(options));
    if (!engine) {
        LOG(ERROR) << "create cuda engine failed.";
        return nullptr;
    }

    auto rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_SET_TP_NCCL_COMM, nccl_comm);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "engine configure failed";
        return nullptr;
    }

    return engine.release();
}

static Runtime* CreatePPLRuntime(Engine* cuda_engine, const string& model_file) {
    auto builder = unique_ptr<onnx::RuntimeBuilder>(onnx::RuntimeBuilderFactory::Create());
    if (!builder) {
        LOG(ERROR) << "create onnx builder failed.";
        return nullptr;
    }

    auto rc = builder->LoadModel(model_file.c_str());
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "load model [" << model_file << "] failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    onnx::RuntimeBuilder::Resources resources;
    resources.engines = &cuda_engine;
    resources.engine_num = 1;

    rc = builder->SetResources(resources);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set resources for builder failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    rc = builder->Preprocess();
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "builder preprocess failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    return builder->CreateRuntime();
}

static void UpdateInputPrefill(int gen_len, ModelInput* model_input) {
    int bs = model_input->first_fill_len.size();
    model_input->decoding_batches = 0;

    model_input->seq_starts.reserve(bs + 1);
    model_input->seq_starts.push_back(0);

    model_input->kv_starts.reserve(bs + 1);
    model_input->kv_starts.push_back(0);

    model_input->start_pos.reserve(bs);

    model_input->cache_indices.reserve(bs);
    model_input->cache_indices.push_back(0);

    for (int i = 0; i < bs; ++i) {
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
    int bs = model_input->first_fill_len.size();
    model_input->decoding_batches = bs;
    model_input->max_seq_len = 1;
    model_input->max_kv_len = model_input->max_kv_len + 1;

    model_input->token_ids.resize(bs);

    for (int i = 0; i < bs; ++i) {
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

static std::shared_ptr<ppl::llm::cuda::Sampler> CreateCudaSampler(Runtime* runtime) {
    DeviceContext::Type needed_type;
    *((int64_t*)needed_type.str) = 0;
    needed_type.str[0] = 'c';
    needed_type.str[1] = 'u';
    needed_type.str[2] = 'd';
    needed_type.str[3] = 'a';

    DeviceContext* dev = nullptr;
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
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "Configure ppl::nn::llm::cuda::DEV_CONF_GET_STREAM failed: " << GetRetCodeStr(rc);
        return std::shared_ptr<ppl::llm::cuda::Sampler>();
    }

    return std::make_shared<ppl::llm::cuda::Sampler>(stream);
}

static void PrintInputInfo(const Runtime* runtime) {
    LOG(INFO) << "----- input info -----";
    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto tensor = runtime->GetInputTensor(i);
        LOG(INFO) << "input[" << i << "]:";
        LOG(INFO) << "    name: " << tensor->GetName();

        string dims_str;
        auto shape = tensor->GetShape();
        for (uint32_t j = 0; j < shape->GetDimCount(); ++j) {
            dims_str += " " + ToString(shape->GetDim(j));
        }
        LOG(INFO) << "    dim(s):" << dims_str;

        LOG(INFO) << "    data type: " << GetDataTypeStr(shape->GetDataType());
        LOG(INFO) << "    data format: " << GetDataFormatStr(shape->GetDataFormat());
        LOG(INFO) << "    byte(s) excluding padding: " << shape->CalcBytesExcludingPadding();
        LOG(INFO) << "    buffer address: " << tensor->GetBufferPtr();

        const int64_t elem_count = tensor->GetShape()->CalcElementsExcludingPadding();
        if (tensor->GetShape()->GetDataType() == ppl::common::DATATYPE_INT64 && elem_count <= 16) {
            std::vector<int64_t> vals(elem_count, 0);
            if (ppl::common::RC_SUCCESS != tensor->CopyToHost(vals.data())) {
                LOG(ERROR) << "[" << tensor->GetName() << "] CopyToHost FAILED";
            } else {
                std::string val_str = "";
                for (uint32_t j = 0; j < elem_count; ++j) {
                    val_str += std::to_string(vals[j]) + " ";
                }
                LOG(INFO) << "    value(s): " << val_str;
            }
        }
    }

    LOG(INFO) << "----------------------";
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

    ~LLM() {
        sampler_.reset();

        for (int i = 0; i < tensor_parallel_size_; ++i) {
            cudaFree(worker_thread_args_[i].kv_cache_mem);
            cudaFree(worker_thread_args_[i].kv_scale_mem);

            worker_thread_args_[i].runtime.reset();
        }

        engine_list_.clear();

        for (int i = 0; i < tensor_parallel_size_; ++i) {
            auto e = ncclCommDestroy(nccl_comm_list_[i]);
            if (e != ncclSuccess) {
                LOG(ERROR) << "NCCL error(code:" << (int)e << ") on "
                           << "(ncclCommDestroy)";
            }
        }
    }

    bool Init(const ModelConfig& model_config, const std::string& model_dir) {
        vocab_size_ = model_config.vocab_size;
        kv_cache_block_bytes_ = model_config.num_layers * 2 * model_config.num_kv_heads / tensor_parallel_size_ *
            model_config.hidden_dim / model_config.num_heads * sizeof(int8_t);
        kv_scale_block_bytes_ = model_config.num_layers * 2 * model_config.num_kv_heads / tensor_parallel_size_ *
            model_config.hidden_dim / model_config.num_heads / model_config.cache_quant_group * sizeof(float16_t);

        auto rc = InitNccl(tensor_parallel_size_, &nccl_comm_list_);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "NCCL init failed.";
            exit(-1);
        }
        LOG(INFO) << "Init Nccl successed";

#pragma omp parallel num_threads(tensor_parallel_size_)
        {
            int id = omp_get_thread_num();
            engine_list_[id] = unique_ptr<Engine>(CreateCudaEngine(nccl_comm_list_[id], id));
            if (!engine_list_[id]) {
                LOG(ERROR) << "create cuda engine failed.";
                exit(-1);
            }
            LOG(INFO) << "Create cuda engine success";

            const std::string model_path = model_dir + "/model_slice_" + std::to_string(id) + "/model.onnx";
            worker_thread_args_[id].runtime = unique_ptr<Runtime>(CreatePPLRuntime(engine_list_[id].get(), model_path));
            if (!worker_thread_args_[id].runtime) {
                LOG(ERROR) << "create runtime failed.";
                exit(-1);
            }
            LOG(INFO) << "Create runtime success";
        }

        sampler_ = CreateCudaSampler(worker_thread_args_[0].runtime.get());
        if (!sampler_) {
            LOG(ERROR) << "CreateCudaSampler failed";
            return false;
        }
        LOG(INFO) << "create cuda sampler success";

        return true;
    }

    bool PrepareInput(int bs, int input_token_len) {
        temperature_list_.resize(bs);
        for (size_t i = 0; i < temperature_list_.size(); ++i) {
            temperature_list_[i] = temperature_;
        }

        uint64_t kv_cache_tokens = bs * (input_token_len + generation_len_ - 1);
        LOG(INFO) << "kv_cache_tokens: " << kv_cache_tokens;

#pragma omp parallel num_threads(tensor_parallel_size_)
        {
            int id = omp_get_thread_num();
            auto cu_ret = cudaMalloc(&worker_thread_args_[id].kv_cache_mem, kv_cache_tokens * kv_cache_block_bytes_);
            if (cu_ret != cudaSuccess) {
                LOG(ERROR) << "alloc kv cache [" << kv_cache_tokens * kv_cache_block_bytes_
                           << "] failed: " << cudaGetErrorString(cu_ret);
                exit(-1);
            }
            cu_ret = cudaMalloc(&worker_thread_args_[id].kv_scale_mem, kv_cache_tokens * kv_scale_block_bytes_);
            if (cu_ret != cudaSuccess) {
                cudaFree(worker_thread_args_[id].kv_cache_mem);
                LOG(ERROR) << "alloc kv scale [" << kv_cache_tokens * kv_scale_block_bytes_
                           << "] failed: " << cudaGetErrorString(cu_ret);
                exit(-1);
            }
        }

        // init tensor
        for (int i = 0; i < tensor_parallel_size_; ++i) {
            auto* arg = &worker_thread_args_[i];
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
        }
        return true;
    }

    bool SetInputTensor(const ModelInput& model_input, int id, int step) {
        RetCode rc;
        // token ids
        // if (step < 2) {
        worker_thread_args_[id].token_ids->GetShape()->Reshape({int64_t(model_input.token_ids.size())});
        rc = worker_thread_args_[id].token_ids->CopyFromHostAsync(model_input.token_ids.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set token_ids [" << worker_thread_args_[id].token_ids->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return false;
        }
        // }

        // kv_starts
        worker_thread_args_[id].kv_starts->GetShape()->Reshape({int64_t(model_input.kv_starts.size())});
        rc = worker_thread_args_[id].kv_starts->CopyFromHostAsync(model_input.kv_starts.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set kv_starts " << worker_thread_args_[id].kv_starts->GetName()
                       << " failed: " << GetRetCodeStr(rc);
            return false;
        }

        // start_pos
        worker_thread_args_[id].start_pos->GetShape()->Reshape({int64_t(model_input.start_pos.size())});
        rc = worker_thread_args_[id].start_pos->CopyFromHostAsync(model_input.start_pos.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set start_pos [" << worker_thread_args_[id].start_pos->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return false;
        }

        // max_kv_len
        rc = worker_thread_args_[id].max_kv_len->CopyFromHostAsync(&model_input.max_kv_len);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set max_kv_len [" << worker_thread_args_[id].max_kv_len->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return false;
        }

        // prefill
        if (step < 1) {
            // cache_indices
            worker_thread_args_[id].cache_indices->GetShape()->Reshape({int64_t(model_input.cache_indices.size())});
            rc = worker_thread_args_[id].cache_indices->CopyFromHostAsync(model_input.cache_indices.data());
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "set cache_indices [" << worker_thread_args_[id].cache_indices->GetName()
                           << "] failed: " << GetRetCodeStr(rc);
                return false;
            }
        }

        if (step < 2) {
            // seq_start
            // LOG(INFO) << "model_input.seq_starts: ";
            // PrintVector(model_input.seq_starts);
            worker_thread_args_[id].seq_starts->GetShape()->Reshape({int64_t(model_input.seq_starts.size())});
            rc = worker_thread_args_[id].seq_starts->CopyFromHostAsync(model_input.seq_starts.data());
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "set seq_starts [" << worker_thread_args_[id].seq_starts->GetName()
                           << "] failed: " << GetRetCodeStr(rc);
                return false;
            }

            // decoding batches
            rc = worker_thread_args_[id].decoding_batches->CopyFromHostAsync(&model_input.decoding_batches);
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "set decoding_batches [" << worker_thread_args_[id].decoding_batches->GetName()
                           << "] failed: " << GetRetCodeStr(rc);
                return false;
            }

            // max_seq_len
            rc = worker_thread_args_[id].max_seq_len->CopyFromHostAsync(&model_input.max_seq_len);
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "set max_seq_len [" << worker_thread_args_[id].max_seq_len->GetName()
                           << "] failed: " << GetRetCodeStr(rc);
                return false;
            }
        }

        // rc = worker_thread_args_[id].runtime->Synchronize();
        // if (rc != RC_SUCCESS) {
        //     LOG(ERROR) << "set input tensor synchronize fail";
        //     return false;
        // }
        return true;
    }

    void Generate(ModelInput* model_input, std::vector<std::vector<int32_t>>* output_tokens) {
        int bs = model_input->first_fill_len.size();

        double step_latency = 0;
        for (int step = 0; step < generation_len_; ++step) {
            {
                TimingGuard __timing__(&step_latency);
                if (step == 0) {
                    UpdateInputPrefill(generation_len_, model_input);
                } else {
                    UpdateInputDecode(step, output_tokens->at(step - 1), model_input);
                }

                {

                    #pragma omp parallel num_threads(tensor_parallel_size_)
                    {
                        int id = omp_get_thread_num();
                        bool ret = SetInputTensor(*model_input, id, step);
                        if (!ret) {
                            LOG(ERROR) << "SetInputTensor failed";
                            exit(-1);
                        }

                        auto rc = worker_thread_args_[id].runtime->Run();
                        if (rc != RC_SUCCESS) {
                            LOG(ERROR) << "model run failed";
                            exit(-1);
                        }
                    }
                }

                auto logits = worker_thread_args_[0].logits;
                auto rc = sampler_->SampleTopPTopK((float*)logits->GetBufferPtr(), temperature_list_.data(), bs,
                                                   vocab_size_, top_p_, top_k_, output_tokens->at(step).data());

                if (rc != RC_SUCCESS) {
                    LOG(ERROR) << "SampleTopPTopK failed: " << GetRetCodeStr(rc);
                    break;
                }
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
    config->input_file = g_flag_input_file;
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
    LOG(INFO) << "config.input_file: " << config->input_file;

    LOG(INFO) << "config.quant_method: " << config->quant_method;
}

bool ParseModelConfig(const std::string& model_param_path, ModelConfig* model_config) {
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

int main(int argc, char* argv[]) {
    ppl::common::GetCurrentLogger()->SetLogLevel(ppl::common::LOG_LEVEL_ERROR);
    simple_flags::parse_args(argc, argv);
    if (!simple_flags::get_unknown_flags().empty()) {
        string content;
        for (auto it : simple_flags::get_unknown_flags()) {
            content += "'" + it + "', ";
        }
        content.resize(content.size() - 2); // remove last ', '
        content.append(".");
        LOG(ERROR) << "unknown option(s): " << content.c_str();
        return -1;
    }

    LOG(INFO) << "ppl.nn version: [" << PPLNN_VERSION_MAJOR << "." << PPLNN_VERSION_MINOR << "." << PPLNN_VERSION_PATCH
              << "], commit: [" << PPLNN_COMMIT_STR << "]";

    LOG(INFO) << "g_flag_batch_size: " << g_flag_batch_size;

    Config config;
    ParseConfig(&config);

    ModelConfig model_config;
    if (!ParseModelConfig(config.model_param_path, &model_config)) {
        LOG(ERROR) << "PaseModelConfig failed, model_param_path: " << config.model_param_path;
        return -1;
    }

    ModelInput raw_model_input;
    ParseInput(config.input_file, &raw_model_input);
    LOG(INFO) << "batch size: " << raw_model_input.first_fill_len.size();
    int bs = raw_model_input.first_fill_len.size();
    int input_token_len = raw_model_input.token_ids.size() / bs;

    profiling.step_latency.resize(config.generation_len);

    LLM llm(config);
    bool ret = llm.Init(model_config, config.model_dir);
    if (!ret) {
        LOG(ERROR) << "llm init failed";
    }

    ret = llm.PrepareInput(bs, input_token_len);
    if (!ret) {
        LOG(ERROR) << "llm prepare input failed";
    }
    LOG(INFO) << "llm prepare input success";

    std::vector<std::vector<int32_t>> output_tokens(config.generation_len, std::vector<int32_t>(bs));

    // warmup
    for (uint32_t i = 0; i < g_flag_warmup_loops; ++i) {
        ModelInput model_input = raw_model_input;
        llm.Generate(&model_input, &output_tokens);
    }

    profiling.Reset();
    for (int i = 0; i < config.benchmark_loops; ++i) {
        ModelInput model_input = raw_model_input;
        double latency = 0;
        {
            TimingGuard __timeing__(&latency);
            llm.Generate(&model_input, &output_tokens);
        }
        profiling.total_latency += latency;
    }
    size_t avail_bytes = 0, total = 0;
    cudaMemGetInfo(&avail_bytes, &total);
    profiling.mem_usage = double((total - avail_bytes) >> 20) / 1024;

    // // get result
    // LOG(INFO) << "output: ";
    // for (int i = 0; i < bs; ++i) {
    //     for (int j = 0; j < config.generation_len; ++j) {
    //         std::cout << output_tokens[j][i] << ", ";
    //     }
    //     std::cout <<" --------------------------------- " << std::endl;
    // }

    // profiling 结果
    double avg_prefill_latency = 0;
    double max_decode_latency = 0;
    double min_decode_latency = 100000;
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

    fprintf(stderr,
            "[INFO] |- Request batch size: %d, Generation len: %d, Input tokens: %d, top_p: %.2f, top_k: %.2f, "
            "temperature: %.2f, warmup_loops: %d, benchmark_loops: %d\n",
            bs, config.generation_len, input_token_len, config.top_p, config.top_k, config.temperature,
            g_flag_warmup_loops, config.benchmark_loops);

    fprintf(stderr, "[PERF] |- GPU Memory usage: %.2f GB\n", profiling.mem_usage);

    // fprintf(stderr, "[PERF] |- SetInput Latency   | avg step: %.2f ms | total: %.2f ms\n",
    // profiling.set_intput_latency / (config.benchmark_loops * config.generation_len), profiling.set_intput_latency);

    fprintf(stderr,
            "[PERF] |- Prefill latency: %.2f ms |  min decode latency: %.2f ms at step [%d] | max decode latency: %.2f "
            "ms at step [%d] | avg decode latency: %.2f ms, avg step latency: %.2f ms\n",
            avg_prefill_latency, min_decode_latency, min_latency_step, max_decode_latency, max_latency_step,
            avg_decode_latency, avg_step_latency);

    // fprintf(stderr, "[PERF] |- Total Latency            | avg step: %.2f ms | avg generation : %.2f ms | total: %.2f
    // ms\n",
    //         profiling.total_latency / (config.benchmark_loops * config.generation_len), profiling.total_latency /
    //         config.benchmark_loops, profiling.total_latency);
    fprintf(stderr, "=======================================================================\n");
    return 0;
}
