#pragma once

#include "request.h"
#include "model_confg.h"

#include "ppl/common/log.h"

#include <vector>
#include <stdint.h>

struct GenerationState {
    struct ModelIO {
        std::vector<int64_t> token_ids;
        std::vector<int64_t> seq_starts;
        std::vector<int64_t> kv_starts;
        std::vector<int64_t> cache_indices;
        std::vector<int64_t> start_pos;

        std::vector<int64_t> seq_lengths;
        std::vector<int64_t> kv_lengths;

        int64_t decoding_batches = 0;
        int64_t max_seq_len = 0;
        int64_t max_kv_len = 0;

        std::vector<int32_t> output;
        std::vector<int64_t> batch_slots;
    } model_io;

    int64_t total_batch_size;
    int64_t total_input_length = 0;
    int64_t total_output_length = 0;
    int64_t total_cache_length = 0;
    int64_t total_cache_pages = 0;
    int64_t max_cache_pages = 0;
    int64_t max_steps = 0;

    int64_t current_step = 0;
    int64_t current_batch_size = 0;

    ppl::common::RetCode PrepareGeneration(
        const std::vector<Request>& requests,
        const ModelConfig &model_config,
        std::vector<Response> *response);

    void FirstStep(
        const std::vector<Request>& requests,
        const ModelConfig &model_config,
        std::vector<Response> *response);

    bool NextStep(
        const std::vector<Request>& requests,
        const ModelConfig &model_config,
        std::vector<Response> *response);
};
