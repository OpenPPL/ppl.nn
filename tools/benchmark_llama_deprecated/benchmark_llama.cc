#include "sampler.h"
#include "../simple_flags.h"

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

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/runtime_builder_factory.h"
#include "ppl/nn/models/pmx/load_model_options.h"
#include "ppl/nn/models/pmx/save_model_options.h"
#endif

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

#ifdef PPLNN_ENABLE_PMX_MODEL
Define_bool_opt("--use-pmx", g_flag_use_pmx, false, "use pmx model");
#endif

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
Define_uint32_opt("--micro-batch", g_flag_micro_batch, UINT32_MAX, "dummy");
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

static int64_t random_input[1024] = {4854, 28445, 26882, 19570, 28904, 7224, 11204, 12608, 23093, 5763, 17481, 3637, 4989, 8263, 18072, 7607, 10287, 6389, 30521, 19284, 1001, 30170, 16117, 11688, 3189, 4694, 18740, 6585, 3299, 289, 14008, 22789, 12043, 29885, 19050, 24321, 11134, 6291, 26101, 21448, 9998, 11708, 13471, 4035, 6285, 15050, 3445, 30546, 3335, 9024, 20135, 462, 27882, 29628, 2573, 29186, 24879, 16327, 13250, 2196, 4584, 14253, 24544, 14142, 21916, 26777, 22673, 23681, 29726, 4875, 15073, 25115, 29674, 19967, 14119, 18069, 23952, 4903, 14050, 7884, 25496, 25353, 8206, 17718, 24951, 22931, 25282, 27350, 7459, 15428, 13848, 17086, 30838, 6330, 19846, 21990, 12750, 18192, 23364, 31189, 2049, 5170, 18875, 1550, 24837, 20623, 5968, 21205, 12275, 11288, 31214, 17545, 25403, 22595, 26832, 27094, 4287, 2088, 14693, 30114, 11775, 16566, 1128, 9841, 6723, 4064, 19010, 10563, 16391, 22630, 25224, 4214, 10438, 4197, 20711, 25095, 8637, 1249, 21827, 15920, 1269, 24989, 18823, 10217, 4197, 18277, 3692, 3326, 16183, 12565, 11703, 20781, 26531, 9290, 11666, 18146, 20460, 3866, 30325, 23696, 14540, 15313, 17313, 11808, 24707, 7762, 7928, 31121, 188, 27724, 20011, 21316, 26679, 8934, 25191, 7640, 12644, 2745, 28379, 2915, 30257, 11475, 23502, 18365, 16392, 6913, 26862, 12704, 18085, 28552, 7072, 23477, 30879, 26014, 10777, 22887, 25528, 13986, 16807, 7838, 1914, 29227, 13069, 9977, 15107, 22174, 2453, 4482, 25644, 20425, 23556, 22172, 15768, 15790, 29825, 14381, 30648, 9594, 22624, 11919, 4756, 8095, 3566, 25349, 7798, 1451, 16108, 1740, 20877, 8163, 30604, 31876, 24077, 18241, 7281, 6266, 4243, 7069, 19769, 22766, 18629, 11727, 19192, 26391, 26689, 25834, 19592, 7891, 21956, 14238, 27197, 12860, 31620, 25199, 30635, 20908, 10656, 12847, 2502, 12412, 4969, 12149, 13885, 19198, 2346, 23433, 8594, 26669, 25496, 3386, 15291, 7447, 27139, 14139, 9704, 7289, 2297, 18465, 15065, 29629, 29297, 18111, 16321, 23181, 4635, 5194, 5680, 20010, 22590, 2653, 3869, 24767, 1965, 24028, 30772, 23175, 29866, 2205, 18108, 15062, 3118, 9045, 5723, 6415, 31082, 2188, 7311, 20256, 19578, 21254, 16531, 16726, 3079, 10648, 10834, 11582, 19042, 4120, 21394, 18674, 23845, 1607, 16299, 22337, 22147, 4969, 25872, 24250, 29371, 23383, 13664, 9146, 23049, 17562, 3404, 1871, 27293, 1761, 16423, 13860, 10916, 2501, 18750, 31245, 9438, 7113, 27553, 19404, 3935, 19308, 19074, 10950, 2523, 10560, 8343, 9880, 27166, 15279, 14267, 20852, 14966, 24011, 22818, 15692, 1707, 5708, 9276, 24446, 27951, 4064, 3860, 11723, 14799, 14288, 14789, 24125, 30444, 29224, 9204, 17018, 13849, 21455, 17831, 8628, 1219, 6999, 22257, 7093, 21735, 9971, 17377, 12209, 17336, 13298, 25329, 13935, 31161, 22448, 23774, 748, 20329, 534, 30021, 14973, 6819, 20014, 22457, 29490, 21, 16223, 5492, 12695, 17176, 11757, 21868, 9953, 11467, 19631, 8310, 22225, 21181, 2503, 31558, 3028, 16996, 22232, 3690, 21498, 3742, 5285, 7486, 30377, 28383, 24183, 25623, 19988, 15639, 30002, 31411, 10780, 17521, 20937, 15612, 20057, 8355, 8916, 974, 30669, 18007, 164, 24930, 5119, 31156, 5946, 7294, 12805, 8349, 24333, 25220, 22156, 17136, 30967, 22668, 18047, 23242, 31038, 16002, 6195, 7639, 3549, 26399, 24178, 2848, 5888, 12496, 7480, 23608, 479, 31809, 30003, 26686, 19203, 22386, 7131, 4202, 3938, 4982, 31438, 3689, 29917, 19597, 28127, 4193, 18764, 2921, 4958, 22711, 93, 9594, 2494, 25492, 29359, 1596, 19777, 16806, 31869, 30211, 18345, 25026, 7879, 31933, 3583, 24569, 13110, 26598, 28383, 18403, 31994, 26340, 16875, 7114, 7372, 21954, 27227, 9279, 9757, 29061, 8525, 13101, 7744, 14296, 3679, 20769, 681, 12047, 3626, 14519, 1882, 3318, 17983, 19078, 10225, 11902, 22704, 448, 17143, 4973, 4354, 8100, 16630, 21754, 17219, 21381, 17471, 15750, 21204, 16511, 13165, 15525, 21326, 30660, 17947, 13702, 3995, 4059, 20, 30822, 22434, 19823, 7723, 13703, 20727, 11601, 17352, 13278, 31426, 20254, 6780, 8720, 17786, 15357, 5186, 11210, 23357, 6095, 21162, 640, 17668, 26775, 15785, 24912, 3374, 16072, 1838, 10180, 10731, 21572, 29611, 19191, 515, 10627, 12119, 6484, 9732, 8013, 22587, 1849, 3148, 18262, 15175, 13366, 20509, 5587, 30812, 2584, 31511, 11407, 6734, 18259, 13605, 9521, 25685, 30029, 31019, 6722, 3166, 15975, 12804, 17449, 29155, 26789, 23069, 19316, 26635, 29030, 21767, 24352, 12835, 5827, 21404, 15769, 15340, 31644, 6557, 4483, 15009, 5492, 30064, 29790, 30548, 22490, 30943, 12428, 29600, 5910, 12041, 26366, 28920, 3731, 5983, 1577, 3275, 15440, 4307, 10031, 20999, 8512, 766, 8616, 23190, 2754, 17507, 8830, 28490, 19489, 30404, 18750, 19824, 9129, 13398, 28868, 9680, 14908, 1086, 25230, 3432, 18402, 21096, 26573, 13830, 10086, 30708, 29992, 2173, 22163, 1572, 7598, 26022, 20475, 29632, 13133, 21975, 13792, 29371, 18452, 17421, 27734, 5914, 7317, 21842, 10833, 9780, 19507, 456, 15224, 20667, 45, 25414, 17738, 527, 31635, 31812, 8268, 23148, 24295, 1167, 2536, 14759, 10377, 2069, 13663, 12073, 16907, 29637, 5153, 4634, 25994, 397, 31527, 1150, 18942, 28864, 25195, 20448, 6497, 16291, 25399, 6059, 20762, 10191, 9196, 5438, 30897, 9234, 21348, 15318, 10919, 8330, 1781, 4175, 22058, 12618, 23993, 27484, 19815, 13835, 14605, 30530, 12528, 15855, 15094, 25708, 16082, 14820, 19526, 7676, 9215, 19222, 21365, 20375, 8183, 7369, 7940, 17555, 24506, 8138, 3027, 10721, 17146, 18460, 12332, 5174, 12780, 25184, 2895, 19014, 7408, 19011, 1396, 4581, 23738, 18612, 18277, 2646, 27617, 17913, 14895, 11038, 22787, 23271, 4618, 29633, 28035, 25643, 6758, 29526, 2681, 2217, 22770, 1632, 20076, 30737, 4613, 6318, 19603, 24994, 2587, 24149, 7230, 29733, 21695, 12255, 22514, 26849, 5111, 17797, 24847, 16833, 12742, 12003, 5286, 17873, 10942, 23972, 21230, 6546, 14866, 18500, 15393, 22536, 8133, 25296, 22484, 19982, 13087, 29776, 23359, 10425, 22028, 11190, 16693, 2118, 23351, 27817, 21382, 1189, 25925, 19520, 27026, 2639, 15749, 18384, 29283, 29672, 21813, 19320, 31083, 23918, 26421, 11032, 25719, 19729, 30445, 14226, 8696, 29600, 9000, 15486, 29377, 1422, 12197, 6116, 3543, 21149, 28361, 6570, 26061, 3658, 21072, 2339, 19848, 17606, 2944, 24911, 6300, 13493, 16401, 19117, 31785, 22760, 24634, 26375, 7856, 20481, 25122, 14345, 16559, 6296, 27652, 13643, 15577, 21088, 1292, 6931, 31824, 15488, 25473, 19310, 20581, 21956, 9402, 4613, 1639, 840, 26369, 28685, 30877, 17166, 659, 28898, 11557, 19939, 31031, 18452, 29644, 19566, 12301, 472, 20018, 19573, 8257, 25520, 3814, 10656, 13039, 14661, 2207, 26849, 21633, 23418, 16230, 13791, 6774, 27429, 9088, 3167, 15050, 7711, 20597, 24940, 26294, 16510, 4960, 1806, 25994, 20792, 5446, 10808, 6183, 17514, 3541, 28826, 22857, 23680, 16870, 20164, 3110, 5153, 19392, 26894, 9187, 721, 27523, 7362, 1268, 15641, 15800, 11869, 10599, 12818, 13302, 19468, 26556, 29696, 30405, 9210, 1918, 13974, 17268, 19746, 13401, 17902, 9654, 26288, 17900, 23369, 3759, 2450, 30977, 30906, 17485, 17301, 26017, 14638};

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
    int64_t decoding_batches = 0;
    std::vector<int64_t> start_pos;
    int64_t max_seq_len = 0;
    int64_t max_kv_len = 0;

    void* kv_cache;
    void* kv_scale;

    std::vector<int64_t> first_fill_len;
};

struct WorkerThreadArg final {
    std::unique_ptr<ppl::nn::DeviceContext> host_device;
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

#ifdef PPLNN_ENABLE_PMX_MODEL
static ppl::nn::Runtime* CreatePMXPPLRuntime(ppl::nn::Engine* cuda_engine, const std::string& model_file) {
    auto builder = std::unique_ptr<ppl::nn::pmx::RuntimeBuilder>(ppl::nn::pmx::RuntimeBuilderFactory::Create());
    if (!builder) {
        LOG(ERROR) << "create PmxRuntimeBuilder failed.";
        return nullptr;
    }

    ppl::nn::pmx::RuntimeBuilder::Resources resources;
    resources.engines = &cuda_engine;
    resources.engine_num = 1;

    std::string external_data_dir_fix;
    ppl::nn::pmx::LoadModelOptions opt;
    auto status = builder->LoadModel(model_file.c_str(), resources, opt);
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
}
#endif //PPLNN_ENABLE_PMX_MODEL

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
            worker_thread_args_[tid].host_device.reset();
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

#ifdef PPLNN_ENABLE_PMX_MODEL
            if (g_flag_use_pmx) 
            {
                const std::string model_path = model_dir + "/model_slice_" + std::to_string(tid) + "/model.pmx";
                worker_thread_args_[tid].host_device.reset(ppl::nn::llm::cuda::EngineFactory::CreateHostDeviceContext(
                                                            ppl::nn::llm::cuda::HostDeviceOptions()));
                worker_thread_args_[tid].runtime = std::unique_ptr<ppl::nn::Runtime>(CreatePMXPPLRuntime(engine_list_[tid].get(), model_path));
            } 
            else
#endif
            {
                const std::string model_path = model_dir + "/model_slice_" + std::to_string(tid) + "/model.onnx";
                worker_thread_args_[tid].host_device.reset(ppl::nn::llm::cuda::EngineFactory::CreateHostDeviceContext(
                                                            ppl::nn::llm::cuda::HostDeviceOptions()));
                worker_thread_args_[tid].runtime = std::unique_ptr<ppl::nn::Runtime>(CreatePPLRuntime(engine_list_[tid].get(), model_path));
            }
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

            arg->decoding_batches->SetDeviceContext(arg->host_device.get());
            arg->max_seq_len->SetDeviceContext(arg->host_device.get());
            arg->max_kv_len->SetDeviceContext(arg->host_device.get());

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

static void GenInput(int vocab_size, ModelInput* model_input) {
    model_input->first_fill_len.assign(g_flag_batch_size, g_flag_input_len);
    model_input->token_ids.resize(g_flag_batch_size * g_flag_input_len);
    for(uint32_t i = 0; i < model_input->token_ids.size(); ++i) {
        model_input->token_ids[i] = random_input[i % 1024] % vocab_size;
    }
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
        GenInput(model_config.vocab_size, &raw_model_input);
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