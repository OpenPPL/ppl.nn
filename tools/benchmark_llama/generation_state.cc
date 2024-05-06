#include "generation_state.h"

#include <set>

ppl::common::RetCode GenerationState::PrepareGeneration(
    const std::vector<Request>& requests,
    const ModelConfig &model_config,
    std::vector<Response> *response)
{
    total_batch_size = requests.size();
    total_input_length = 0;
    total_output_length = 0;
    total_cache_length = 0;
    total_cache_pages = 0;
    max_cache_pages = 0;
    max_steps = 0;

    response->resize(total_batch_size);
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &req = requests[i];
        auto &resp = response->at(i);

        if (req.generation_len <= 0) {
            LOG(ERROR) << "generation_len of request must not be 0";
            return ppl::common::RC_INVALID_VALUE;
        }

        resp.token_ids.resize(req.generation_len);
        total_input_length += req.token_ids.size();
        total_output_length += req.generation_len;

        const int32_t cache_length = req.token_ids.size() + req.generation_len - 1;
        max_steps = std::max(max_steps, req.generation_len);
        if (model_config.cache_mode == 0) {
            total_cache_length += cache_length;
        } else if (model_config.cache_mode == 1) {
            // Align cache to page size
            const int64_t cache_pages = (cache_length + model_config.page_size - 1) / model_config.page_size;
            total_cache_length += cache_pages * model_config.page_size;
            total_cache_pages += cache_pages;
            max_cache_pages = std::max(max_cache_pages, cache_pages);
        } else {
            LOG(ERROR) << "unknown cache_mode: " << model_config.cache_mode;
            return ppl::common::RC_INVALID_VALUE;
        }
    }

    LOG(INFO) << "request_number = " << requests.size()
        << ", total_input_length = " << total_input_length
        << ", max_steps = " << max_steps
        << ", total_cache_length = " << total_cache_length;
    if (model_config.cache_mode == 1) {
        LOG(INFO) << "total_cache_pages = " << total_cache_pages
            << ", max_cache_pages_per_request = " << max_cache_pages;
    }

    return ppl::common::RC_SUCCESS;
}

void GenerationState::FirstStep(
    const std::vector<Request>& requests,
    const ModelConfig &model_config,
    std::vector<Response> *response)
{
    current_step = 0;
    current_batch_size = total_batch_size;

    model_io.batch_slots.resize(current_batch_size);
    model_io.output.resize(total_batch_size);

    model_io.decoding_batches = 0;
    model_io.seq_starts.clear();
    model_io.seq_starts.reserve(current_batch_size + 1);
    model_io.seq_starts.push_back(0);
    model_io.kv_starts.clear();
    model_io.kv_starts.reserve(current_batch_size + 1);
    model_io.kv_starts.push_back(0);

    model_io.start_pos.resize(current_batch_size);
    model_io.kv_lengths.resize(current_batch_size);
    model_io.seq_lengths.resize(current_batch_size);

    model_io.token_ids.resize(total_input_length);
    if (model_config.cache_mode == 0) {
        model_io.cache_indices.resize(current_batch_size);
    } else if (model_config.cache_mode == 1) {
        model_io.cache_indices.resize(current_batch_size * max_cache_pages);
    }

    int64_t current_cache_index = 0;
    int64_t current_input_index = 0;
    for (int64_t i = 0; i < current_batch_size; ++i) {
        auto &req = requests[i];
        const int64_t input_length = req.token_ids.size();
        const int64_t cache_length = input_length + req.generation_len  - 1;

        model_io.batch_slots[i] = i;
        model_io.seq_lengths[i] = input_length;
        model_io.kv_lengths[i] = input_length;
        model_io.start_pos[i] = 0;

        model_io.seq_starts.push_back(model_io.seq_starts[i] + model_io.seq_lengths[i]);
        model_io.kv_starts.push_back(model_io.kv_starts[i] + model_io.kv_lengths[i]);
        model_io.max_seq_len = std::max(model_io.seq_lengths[i], model_io.max_seq_len);
        model_io.max_kv_len = std::max(model_io.kv_lengths[i], model_io.max_kv_len);

        if (model_config.cache_mode == 0) {
            model_io.cache_indices[i] = current_cache_index;
            current_cache_index += cache_length;
        }
        if (model_config.cache_mode == 1) {
            const int64_t cache_pages = (cache_length + model_config.page_size - 1) / model_config.page_size;
            for (int64_t p = 0; p < cache_pages; ++p) {
                model_io.cache_indices[i * max_cache_pages + p] = current_cache_index;
                current_cache_index += model_config.page_size;
            }
        }
        std::copy(
            req.token_ids.begin(),
            req.token_ids.end(),
            model_io.token_ids.begin() + current_input_index);
        
        current_input_index += input_length;
    }
}

bool GenerationState::NextStep(
    const std::vector<Request>& requests,
    const ModelConfig &model_config,
    std::vector<Response> *response)
{
    // Rearrange output datas
    int64_t finished_batches = 0;
    model_io.token_ids.resize(current_batch_size);
    for (int64_t i = 0; i < current_batch_size; ++i) {
        auto &req = requests[model_io.batch_slots[i]];
        auto &resp = response->at(model_io.batch_slots[i]);

        model_io.token_ids[i] = model_io.output[i];
        resp.token_ids.at(current_step) = model_io.output[i];

        if (current_step + 1 >= req.generation_len) {
            finished_batches++;
            // Mark finished batch's slot to -1.
            // It is safety because it will not be used any more.
            model_io.batch_slots[i] = -1;
        }
    }

    // Do not continue when we meet max_steps
    if (current_step + 1 >= max_steps) {
        return false;
    }
    current_step = current_step + 1;

    // Remove finished batch. And prepare next step's inputs.
    // Using fast-slow indexing method.
    for (int64_t fast_i = 0, slow_i = 0; fast_i < current_batch_size; fast_i++) {
        if (model_io.batch_slots[fast_i] == -1)
            continue;

        model_io.start_pos[slow_i] = model_io.kv_lengths[fast_i];
        model_io.kv_lengths[slow_i] = model_io.kv_lengths[fast_i] + 1;
        model_io.seq_lengths[slow_i] = 1;

        model_io.seq_starts[slow_i + 1] = model_io.seq_starts[slow_i] + model_io.seq_lengths[slow_i];
        model_io.kv_starts[slow_i + 1] = model_io.kv_starts[slow_i] + model_io.kv_lengths[slow_i];
        model_io.max_kv_len = std::max(model_io.kv_lengths[slow_i], model_io.max_kv_len);

        // Do not copy these data if position not changed.
        if (fast_i > slow_i) {
            model_io.batch_slots[slow_i] = model_io.batch_slots[fast_i];
            model_io.token_ids[slow_i] = model_io.token_ids[fast_i];

            if (model_config.cache_mode == 0) {
                model_io.cache_indices[slow_i] = model_io.cache_indices[fast_i];
            }
            if (model_config.cache_mode == 1) {
                std::copy(
                    model_io.cache_indices.begin() + (fast_i + 0) * max_cache_pages,
                    model_io.cache_indices.begin() + (fast_i + 1) * max_cache_pages,
                    model_io.cache_indices.begin() + (slow_i + 0) * max_cache_pages);
            }
        }
        slow_i++;
    }
    current_batch_size -= finished_batches;
    model_io.decoding_batches = current_batch_size;
    model_io.max_seq_len = 1;

    return true;
}
