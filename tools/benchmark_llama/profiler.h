#pragma once

#include <vector>
#include <chrono>
#include <map>

#include "ppl/common/log.h"
#include "ppl/nn/runtime/runtime.h"

struct Profiler {
    int64_t total_run_count = 0;
    int64_t total_step_count = 0;
    double max_mem_usage = 0; // GiB

    double total_prefill_latency = 0;
    double total_generate_latency = 0;
    int64_t total_input_tokens = 0;
    int64_t total_output_tokens = 0;
    int64_t total_request_count = 0;
    

    bool collect_statistics = false;
    ppl::nn::ProfilingStatistics prefill_statistics;
    ppl::nn::ProfilingStatistics decode_statistics;

    void Reset() {
        total_run_count = 0;
        max_mem_usage = 0;

        total_input_tokens = 0;
        total_output_tokens = 0;
        total_prefill_latency = 0;
        total_generate_latency = 0;

        prefill_statistics.prof_info.clear();
        decode_statistics.prof_info.clear();
    }

    void PrintProfilingStatistics() {
        LOG(INFO) << "----- Prefill statistics -----";
        PrintProfilingStatistics(prefill_statistics, total_run_count);
        LOG(INFO) << "----- Decode statistics -----";
        PrintProfilingStatistics(decode_statistics, total_step_count - total_run_count);
    }

private:
    static void PrintProfilingStatistics(const ppl::nn::ProfilingStatistics& stat, int32_t run_count) {
        std::map<std::string, std::pair<double, double>> type_stat;
        std::map<std::string, int32_t> type_count;
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
};

class Timer final {
public:
    void Tic() {
        begin_ = std::chrono::high_resolution_clock::now();
    }
    void Toc() {
        auto end = std::chrono::high_resolution_clock::now();
        diff_millisec_ = double(std::chrono::duration_cast<std::chrono::microseconds>(end - begin_).count()) / 1000.0;
    }
    double GetMilliSecond() {
        return diff_millisec_;
    }

private:
    double diff_millisec_;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_;
};
