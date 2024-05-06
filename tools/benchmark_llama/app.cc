#include "../simple_flags.h"
#include "ppl/common/log.h"

#include "text_generator.h"
#include "cuda/cuda_text_generator.h"


Define_bool_opt("--help", g_flag_help, false, "show these help information");
Define_string_opt("--model-dir", g_flag_model_dir, "", "model directory");
Define_string_opt("--model-param-path", g_flag_model_param_path, "", "path of model params");
Define_bool_opt("--use-pmx", g_flag_use_pmx, false, "use pmx model");

Define_uint32_opt("--batch-size", g_flag_batch_size, 0, "total batch size of benchmark dataset");
Define_uint32_opt("--micro-batch", g_flag_micro_batch, 0, "split benchmark dataset by micro batch, benchmark will be split into batch-size / micro-batch steps");

Define_uint32_opt("--input-len", g_flag_input_len, 0, "input length of request. default: 0(get length from input ids file)");
Define_uint32_opt("--generation-len", g_flag_generation_len, 0, "generation length. default: 0(get length from generation len file)");

Define_string_opt("--input-ids-file", g_flag_input_ids_file, "", "input file of requests' input token ids. no effect if --input-len is non-zero");
Define_string_opt("--generation-lens-file", g_flag_generation_lens_file, "", "input file of requests' generation len. no effect if --generation-len is non-zero");
Define_string_opt("--output-ids-file", g_flag_output_ids_file, "", "output file of requests' output token ids.");

Define_uint32_opt("--tensor-parallel-size", g_flag_tensor_parallel_size, 1, "tensor parallel size");

Define_uint32_opt("--warmup-loops", g_flag_warmup_loops, 2, "warm loops");
Define_uint32_opt("--benchmark-loops", g_flag_benchmark_loops, 4, "benchmark loops");

Define_string_opt("--quant-method", g_flag_quant_method, "none",
                        "llm cuda quantization mehtod, only accept "
                        "\"none\", \"online_i8i8\" and \"online_i4f16\", "
                        "default: \"none\"");
Define_string_opt("--cublas-layout-hint", g_cublas_layout_hint, "default",
                        "matrix layout hint for cublas(currently only effect int8 gemm), only accept "
                        "\"default\", \"ampere\". "
                        "default: \"default\"");

Define_bool_opt("--kernel-profiling", g_flag_kernel_profiling, true, "enable kernel profiling and print profiling info");


static bool WriteOutputToFile(const std::string& output_file, std::queue<Response> &responses) {
    std::ofstream fout(output_file, std::ios::out);
    if (!fout.is_open()) {
        LOG(ERROR) << "error openning " << output_file;
        return false;
    }

    while (!responses.empty()) {
        auto resp = responses.front();

        for (size_t t = 0; t < resp.token_ids.size(); ++t) {
            fout << resp.token_ids[t];
            if (t + 1 < resp.token_ids.size())
                fout << ", ";
        }
        fout << std::endl;

        responses.pop();
    }
    return true;
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

    LOG(INFO) << "================== Generate Parameters ==================";

    auto micro_batch = g_flag_micro_batch;
    if (micro_batch <= 0 || micro_batch > g_flag_batch_size) {
        LOG(WARNING) << "micro_batch <= 0 || micro_batch > batch_size, set it to batch_size";
        micro_batch = g_flag_batch_size;
    }

    LOG(INFO) << "input_ids_file = " << g_flag_input_ids_file;
    LOG(INFO) << "generation_lens_file = " << g_flag_generation_lens_file;
    LOG(INFO) << "output_ids_file = " << g_flag_output_ids_file;

    LOG(INFO) << "input_len = " << g_flag_input_len;
    LOG(INFO) << "generation_len = " << g_flag_generation_len;

    LOG(INFO) << "batch_size = " << g_flag_batch_size;
    LOG(INFO) << "micro_batch = " << micro_batch;

    LOG(INFO) << "================== Init TextGenerator ==================";
    // TODO: Move new CudaTextGenerator into #ifdef USE_LLM_CUDA
    std::unique_ptr<TextGenerator> generator(new CudaTextGenerator(
        {g_cublas_layout_hint}
    ));
    auto rc = generator->InitModel(
        "decoder_only",
        g_flag_model_dir,
        g_flag_model_param_path,
        g_flag_quant_method,
        g_flag_tensor_parallel_size,
        g_flag_use_pmx);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "generator InitModel failed";
        return -1;
    }

    RequestQueue req_queue;
    if (g_flag_input_ids_file.empty()) {
        LOG(INFO) << "================== Fixed Random RequestQueue ==================";
        if(!req_queue.GenerateRandomRequests(
            g_flag_batch_size,
            micro_batch,
            generator->GetModelConfig().vocab_size,
            g_flag_input_len,
            g_flag_generation_len))
        {
            LOG(ERROR) << "GenerateRandomRequests failed";
            return -1;
        }
    } else {
        LOG(INFO) << "================== Read File RequestQueue ==================";
        if (!req_queue.ReadRequestsFromFile(
            g_flag_batch_size,
            micro_batch,
            generator->GetModelConfig().vocab_size,
            g_flag_input_ids_file,
            g_flag_generation_lens_file,
            g_flag_generation_len))
        {
            LOG(ERROR) << "ReadRequestsFromFile failed";
            return -1;
        }
    }

    std::queue<Profiler> profilers;
    std::queue<Response> responses;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    Profiler kernel_profiler;
    kernel_profiler.collect_statistics = true;
#endif

    LOG(INFO) << "------------------ Pop Request ------------------";
    auto req = req_queue.PopRequests(micro_batch);

    if (req.size() && g_flag_warmup_loops > 0) {
        LOG(INFO) << "================== Warm Up ==================";
        std::vector<Response> resp;
        Profiler prof;

        rc = generator->PrepareGeneration(req, &resp);
        if (ppl::common::RC_SUCCESS != rc) {
            LOG(ERROR) << "generator PrepareGeneration failed";
            return -1;
        }

        for (uint32_t i = 0; i < g_flag_warmup_loops; ++i) {
            rc = generator->Generate(req, &resp, &prof);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "generator Generate failed";
                return -1;
            }
        }
    }

    LOG(INFO) << "================== Benchmark ==================";
    for (; req.size(); req = req_queue.PopRequests(micro_batch)) {
        std::vector<Response> resp;
        Profiler prof;

        rc = generator->PrepareGeneration(req, &resp);
        if (ppl::common::RC_SUCCESS != rc) {
            LOG(ERROR) << "generator PrepareGeneration failed";
            return -1;
        }

        for (uint32_t i = 0; i < g_flag_benchmark_loops; ++i) {
            rc = generator->Generate(req, &resp, &prof);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "generator Generate failed";
                return -1;
            }
        }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        if (g_flag_kernel_profiling) {
            LOG(INFO) << "----- Kernel Profiling -----";
            rc = generator->Generate(req, &resp, &kernel_profiler);
            if (ppl::common::RC_SUCCESS != rc) {
                LOG(ERROR) << "generator Generate failed";
                return -1;
            }
        }
#endif

        profilers.push(prof);
        for (auto r : resp)
            responses.push(r);

        LOG(INFO) << "------------------ Pop Request ------------------";
    }

    LOG(INFO) << "================== Finalize TextGenerator ==================";
    rc = generator->FinalizeModel();
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "generator FinalizeModel failed";
        return -1;
    }

    if (!g_flag_output_ids_file.empty()) {
        LOG(INFO) << "================== Write Outputs ==================";
        WriteOutputToFile(g_flag_output_ids_file, responses);
    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    LOG(INFO) << "================== Kernel Profiling Summary ==================";
    kernel_profiler.PrintProfilingStatistics();
#endif

    LOG(INFO) << "================== TextGenerator Profiling Summary ==================";
    double total_prefill_latency = 0;
    double total_generate_latency = 0;
    double max_mem_usage = 0;
    int64_t total_input_tokens = 0;
    int64_t total_ouput_tokens = 0;
    int64_t total_request_count = 0;
    int64_t total_run_count = 0;
    int64_t total_step_count = 0;
    while (!profilers.empty()) {
        auto prof = profilers.front();
        total_input_tokens += prof.total_input_tokens;
        total_ouput_tokens += prof.total_output_tokens;
        total_request_count += prof.total_request_count;
        total_run_count += prof.total_run_count;
        total_step_count += prof.total_step_count;
        total_prefill_latency += prof.total_prefill_latency;
        total_generate_latency += prof.total_generate_latency;
        max_mem_usage = std::max(max_mem_usage, prof.max_mem_usage);
        profilers.pop();
    }
    int64_t total_decode_tokens = total_ouput_tokens - total_request_count;
    double total_decode_latency = total_generate_latency - total_prefill_latency;
    double avg_prefill_latency = total_prefill_latency / total_run_count;
    double avg_generate_latency = total_generate_latency / total_run_count;
    double avg_decode_latency = total_decode_tokens == 0 ? 0.0 : total_decode_latency / (total_step_count - total_run_count);
    double avg_step_latency = total_generate_latency / total_step_count;

    double out_tokens_per_second = total_ouput_tokens / (total_generate_latency / 1000);
    double in_out_tokens_per_second = (total_input_tokens + total_ouput_tokens) / (total_generate_latency / 1000);
    double prefill_tokens_per_second = total_input_tokens / (total_prefill_latency / 1000);
    double decode_tokens_per_second = total_decode_tokens == 0 ? 0.0 : (total_decode_tokens / (total_decode_latency / 1000));

    LOG(INFO) << "Avg Generate latency(ms): " << avg_generate_latency;
    LOG(INFO) << "Avg Prefill  latency(ms): " << avg_prefill_latency;
    LOG(INFO) << "Avg Decode   latency(ms): " << avg_decode_latency;
    LOG(INFO) << "Avg Step     latency(ms): " << avg_step_latency;
    LOG(INFO) << "Prefill         tokens/s: " << prefill_tokens_per_second;
    LOG(INFO) << "Decode          tokens/s: " << decode_tokens_per_second;
    LOG(INFO) << "Output          tokens/s: " << out_tokens_per_second;
    LOG(INFO) << "In-Out          tokens/s: " << in_out_tokens_per_second;
    LOG(INFO) << "Max Memory    usage(GiB): " << max_mem_usage;

    LOG(INFO) << "CSV format header:generate(ms),prefill(ms),decode(ms),step(ms),prefill_tps,decode_tps,o_tps,io_tps,mem(gib)";
    LOG(INFO) << "CSV format output:"
        << avg_generate_latency << ","
        << avg_prefill_latency << ","
        << avg_decode_latency << ","
        << avg_step_latency << ","
        << prefill_tokens_per_second << ","
        << decode_tokens_per_second << ","
        << out_tokens_per_second << ","
        << in_out_tokens_per_second << ","
        << max_mem_usage;

    return 0;
}
