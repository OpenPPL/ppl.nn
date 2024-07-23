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

#include <string>
#include <string.h>
#include <chrono>
#include <string>
#include <memory>
#include <random>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <algorithm>
using namespace std;

#include "ppl/common/str_utils.h"
#include "ppl/common/mmap.h"
using namespace ppl::common;

#include "ppl/nn/runtime/options.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/utils/file_data_stream.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::nn;

#ifdef PPLNN_ENABLE_ONNX_MODEL
#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/runtime_builder_factory.h"
#include "ppl/nn/models/pmx/load_model_options.h"
#include "ppl/nn/models/pmx/save_model_options.h"
#endif

/* -------------------------------------------------------------------------- */

#include "simple_flags.h"

Define_bool_opt("--help", g_flag_help, false, "show these help information");
Define_bool_opt("--version", g_flag_version, false, "show version info");

#ifdef PPLNN_ENABLE_ONNX_MODEL
Define_string_opt("--onnx-model", g_flag_onnx_model, "", "onnx model file");
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
Define_string_opt("--pmx-model", g_flag_pmx_model, "", "pmx model file");
Define_string_opt("--pmx-external-data-dir", g_flag_pmx_external_data_dir, "", "dir that contains external data");
Define_string_opt("--pmx-external-data-file", g_flag_pmx_external_data_file, "", "data file that contains all external data");
Define_string_opt("--export-pmx-model", g_flag_export_pmx_model, "", "dump model to <filename> in pmx format");
Define_string_opt("--save-pmx-model", g_flag_save_pmx_model, "", "deprecated. use `--export-pmx-model` instead.");
#endif

Define_string_opt(
    "--mm-policy", g_flag_mm_policy, "mem",
    "\"mem\"(default) => less memory usage; \"perf\" => better performance; \"plain\": plain implementation");
Define_bool_opt("--no-run", g_flag_no_run, false, "do not evaluate the model");

Define_bool_opt("--enable-profiling", g_flag_enable_profiling, false, "enable profiling and print profiling info");
Define_float_opt("--min-profiling-seconds", g_flag_min_profiling_seconds, 1.0f,
                 "min execute time by seconds for profiling");
Define_uint32_opt("--min-profiling-iterations", g_flag_min_profiling_iterations, 1, "declare profiling iteration");
Define_uint32_opt("--warmup-iterations", g_flag_warmup_iterations, 1, "declare profiling warmup iteration");
Define_bool_opt("--perf-with-io", g_flag_perf_with_io, false, "profiling with io copy");

Define_string_opt("--input", g_flag_input, "", "binary input file containing all tensors' data");
Define_string_opt("--inputs", g_flag_inputs, "", "binary input files separated by comma");
Define_string_opt("--reshaped-inputs", g_flag_reshaped_inputs, "",
                  "binary input files separated by comma."
                  " file name format: 'name-dims-datatype.dat'. for example:"
                  " input1-1_1_1_1-fp32.dat,input2-1_1_1_1-fp16.dat,input3-1_1-int8.dat");
Define_string_opt("--in-shapes", g_flag_input_shapes, "",
                  "shapes of input tensors."
                  " dims are separated by underline, inputs are separated by comma. example:"
                  " 1_3_128_128,2_3_400_640,3_3_768_1024. empty fields between commas are scalars.");

Define_bool_opt("--save-input", g_flag_save_input, false, "save input tensors in one file in NDARRAY format");
Define_bool_opt("--save-inputs", g_flag_save_inputs, false, "save separated input tensors in NDARRAY format");
Define_bool_opt("--save-outputs", g_flag_save_outputs, false, "save separated output tensors in NDARRAY format");
Define_string_opt("--save-data-dir", g_flag_save_data_dir, ".",
                  "directory to save input/output data if '--save-*' options are enabled.");

/* -------------------------------------------------------------------------- */

static vector<int64_t> GenerateRandomDims(uint32_t dim_count) {
    static constexpr uint32_t max_dim = 640;
    static constexpr uint32_t min_dim = 128;
    srand(time(nullptr));

    vector<int64_t> dims(dim_count);
    for (uint32_t i = 0; i < dim_count; ++i) {
        dims[i] = rand() % (max_dim - min_dim + 1) + min_dim;
    }
    return dims;
}

static const char* MemMem(const char* haystack, unsigned int haystack_len, const char* needle,
                          unsigned int needle_len) {
    if (!haystack || haystack_len == 0 || !needle || needle_len == 0) {
        return nullptr;
    }

    for (auto h = haystack; haystack_len >= needle_len; ++h, --haystack_len) {
        if (memcmp(h, needle, needle_len) == 0) {
            return h;
        }
    }
    return nullptr;
}

static void SplitString(const char* str, unsigned int len, const char* delim, unsigned int delim_len,
                        const function<bool(const char* s, unsigned int l)>& f) {
    const char* end = str + len;

    while (str < end) {
        auto cursor = MemMem(str, len, delim, delim_len);
        if (!cursor) {
            f(str, end - str);
            return;
        }

        if (!f(str, cursor - str)) {
            return;
        }

        cursor += delim_len;
        str = cursor;
        len = end - cursor;
    }

    f("", 0); // the last empty field
}

static bool ParseInputShapes(const string& shape_str, vector<vector<int64_t>>* input_shapes) {
    vector<string> input_shape_list;
    SplitString(shape_str.data(), shape_str.size(), ",", 1, [&input_shape_list](const char* s, unsigned int l) -> bool {
        if (l > 0) {
            input_shape_list.emplace_back(s, l);
        } else {
            input_shape_list.push_back(string());
        }
        return true;
    });

    for (auto x = input_shape_list.begin(); x != input_shape_list.end(); ++x) {
        vector<int64_t> shape;

        // empty shape means scalar
        if (!x->empty()) {
            bool ok = true;
            SplitString(x->data(), x->size(), "_", 1, [&ok, &shape](const char* s, unsigned int l) -> bool {
                if (l > 0) {
                    int64_t dim = atol(string(s, l).c_str());
                    shape.push_back(dim);
                    return true;
                }
                LOG(ERROR) << "illegal dim format.";
                ok = false;
                return false;
            });
            if (!ok) {
                return false;
            }
        }

        input_shapes->push_back(shape);
    }

    return true;
}

/* -------------------------------------------------------------------------- */

#ifdef PPLNN_USE_CUDA

Define_bool_opt("--use-cuda", g_flag_use_cuda, false, "use cuda engine");

Define_bool_opt("--quick-select", g_flag_quick_select, false, "quick select algorithms for conv and gemm kernel");
Define_uint32_opt("--device-id", g_flag_device_id, 0, "declare device id for cuda");
Define_string_opt("--kernel-type", g_flag_kernel_type, "",
                  "set kernel type for cuda inferencing. valid values: int8/16/32/64,float16/32");

Define_string_opt("--export-algo-file", g_flag_export_algo_file, "",
                  "Export the selected best algo info into the json file.");
Define_string_opt("--import-algo-file", g_flag_import_algo_file, "",
                  "The objects in the json file declare best algo info for certain conv input shape");

Define_string_opt("--quant-file", g_flag_quant_file, "", "a json file containing quantization information");

Define_bool_opt("--enable-cuda-graph", g_flag_enable_cuda_graph, false, "use cuda graph");

#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/engines/cuda/options.h"
#include "ppl/nn/engines/cuda/utils.h"
#include "ppl/nn/utils/array.h"

static void CudaSaveAlgoInfo(const char* data, uint64_t bytes, void* arg) {
    auto fname = (const char*)arg;
    auto fp = fopen(fname, "w");
    if (!fp) {
        LOG(ERROR) << "open [" << fname << "] for exporting algo info failed.";
        return;
    }

    auto ret = fwrite(data, bytes, 1, fp);
    if (ret != 1) {
        LOG(ERROR) << "write algo info to [" << fname << "] failed.";
    }

    fclose(fp);
}

static bool RegisterCudaEngine(vector<unique_ptr<Engine>>* engines) {
    cuda::EngineOptions options;
    options.device_id = g_flag_device_id;

    if (g_flag_mm_policy == "perf") {
        options.mm_policy = cuda::MM_BEST_FIT;
    } else if (g_flag_mm_policy == "mem") {
        options.mm_policy = cuda::MM_COMPACT;
    } else if (g_flag_mm_policy == "plain") {
        options.mm_policy = cuda::MM_PLAIN;
    } else {
        LOG(ERROR) << "unknown --mm-policy option: " << g_flag_mm_policy;
        return false;
    }

    if (g_flag_enable_cuda_graph) {
        options.enable_cuda_graph = true;
    }

    auto cuda_engine = cuda::EngineFactory::Create(options);
    if (!cuda_engine) {
        return false;
    }

    cuda_engine->Configure(cuda::ENGINE_CONF_USE_DEFAULT_ALGORITHMS, g_flag_quick_select);

    if (!g_flag_kernel_type.empty()) {
        string kernel_type_str(g_flag_kernel_type);
        std::transform(g_flag_kernel_type.begin(), g_flag_kernel_type.end(), kernel_type_str.begin(), ::toupper);

        datatype_t kernel_type = DATATYPE_UNKNOWN;
        for (datatype_t i = DATATYPE_UNKNOWN; i < DATATYPE_MAX; i++) {
            if (GetDataTypeStr(i) == kernel_type_str) {
                kernel_type = i;
                break;
            }
        }

        if (kernel_type != DATATYPE_UNKNOWN) {
            cuda_engine->Configure(cuda::ENGINE_CONF_SET_KERNEL_TYPE, kernel_type);
        } else {
            LOG(ERROR) << "invalid kernel type[" << g_flag_kernel_type << "]. valid values: int8/16/32/64,float16/32.";
        }
    }

    if (!g_flag_quant_file.empty()) {
        Mmap fm;
        auto status = fm.Init(g_flag_quant_file.c_str(), Mmap::READ);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "mapping file[" << g_flag_quant_file << "] failed.";
            return false;
        }
        cuda_engine->Configure(cuda::ENGINE_CONF_SET_QUANT_INFO, fm.GetData(), fm.GetSize());
    }

    if (!g_flag_export_algo_file.empty()) {
        cuda_engine->Configure(cuda::ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER, CudaSaveAlgoInfo,
                               g_flag_export_algo_file.c_str());
    }

    if (!g_flag_import_algo_file.empty()) {
        Mmap fm;
        auto rc = fm.Init(g_flag_import_algo_file.c_str(), Mmap::READ);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "mapping algo file[" << g_flag_import_algo_file << "] failed.";
            return false;
        }

        cuda_engine->Configure(cuda::ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER, fm.GetData(), fm.GetSize());
    }

    // pass input shapes to cuda engine for further optimizations
    if (!g_flag_input_shapes.empty()) {
        vector<vector<int64_t>> input_shapes;
        if (!ParseInputShapes(g_flag_input_shapes, &input_shapes)) {
            LOG(ERROR) << "ParseInputShapes failed.";
            return false;
        }

        vector<utils::Array<int64_t>> dims(input_shapes.size());
        for (uint32_t i = 0; i < input_shapes.size(); ++i) {
            auto& arr = dims[i];
            arr.base = input_shapes[i].data();
            arr.size = input_shapes[i].size();
        }
        cuda_engine->Configure(cuda::ENGINE_CONF_SET_INPUT_DIMS, dims.data(), dims.size());
    }

    engines->emplace_back(unique_ptr<Engine>(cuda_engine));
    LOG(INFO) << "***** register CudaEngine *****";
    return true;
}

#endif

#ifdef PPLNN_USE_X86

Define_bool_opt("--use-x86", g_flag_use_x86, false, "use x86 engine");

Define_bool_opt("--disable-avx512", g_flag_disable_avx512, false, "disable avx512 feature");
Define_bool_opt("--disable-avx-fma3", g_flag_disable_avx_fma3, false, "disable avx, fma3 and avx512 feature");
Define_bool_opt("--core-binding", g_flag_core_binding, false, "core binding");
Define_int32_opt("--num-threads", g_flag_num_threads, 0, "override the environment variable OMP_NUM_THREADS");

Define_bool_opt("--disable-graph-fusion", g_flag_disable_graph_fusion, false, "disable graph kernel fusion rules");
Define_bool_opt("--enable-tensor-debug", g_flag_enable_tensor_debug, false, "dump tensors' data");
Define_string_opt("--debug-data-dir", g_flag_debug_data_dir, ".", "directory to save dumped tensors' data");

#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/x86/options.h"
#include "ppl/nn/engines/x86/threading.h"
#include "ppl/kernel/x86/common/threading_tools.h"

static bool RegisterX86Engine(vector<unique_ptr<Engine>>* engines) {
    x86::EngineOptions options;
    if (g_flag_mm_policy == "perf") {
        options.mm_policy = x86::MM_MRU;
    } else if (g_flag_mm_policy == "mem") {
        options.mm_policy = x86::MM_COMPACT;
    } else if (g_flag_mm_policy == "plain") {
        options.mm_policy = x86::MM_PLAIN;
    } else {
        LOG(ERROR) << "unknown --mm-policy option: " << g_flag_mm_policy;
        return false;
    }

    options.disable_avx512 = g_flag_disable_avx512;
    options.disable_avx_fma3 = g_flag_disable_avx_fma3;

    auto x86_engine = x86::EngineFactory::Create(options);

    auto rc = x86_engine->Configure(x86::ENGINE_CONF_GRAPH_FUSION, g_flag_disable_graph_fusion ? 0 : 1);
    if (RC_SUCCESS != rc) {
        LOG(ERROR) << "x86_engine Configure ENGINE_CONF_GRAPH_FUSION failed: " << GetRetCodeStr(rc);
        return false;
    }
    rc = x86_engine->Configure(x86::ENGINE_CONF_TENSOR_DEBUG, g_flag_enable_tensor_debug ? 1 : 0);
    if (RC_SUCCESS != rc) {
        LOG(ERROR) << "x86_engine Configure ENGINE_CONF_TENSOR_DEBUG failed: " << GetRetCodeStr(rc);
        return false;
    }
    rc = x86_engine->Configure(x86::ENGINE_CONF_DEBUG_DATA_DIR, g_flag_debug_data_dir.c_str());
    if (RC_SUCCESS != rc) {
        LOG(ERROR) << "x86_engine Configure ENGINE_CONF_DEBUG_DATA_DIR failed: " << GetRetCodeStr(rc);
        return false;
    }

    if (g_flag_num_threads) {
        ppl::nn::x86::SetGlobalOmpNumThreads(g_flag_num_threads);
        LOG(INFO) << "set omp_num_threads to: " << g_flag_num_threads;
    }

    if (g_flag_core_binding) {
        ppl::kernel::x86::set_omp_core_binding(nullptr, 0, 1);
    }

    engines->emplace_back(unique_ptr<Engine>(x86_engine));
    LOG(INFO) << "***** register X86Engine *****";
    return true;
}

#endif

#ifdef PPLNN_USE_RISCV

Define_bool_opt("--use-riscv", g_flag_use_riscv, false, "use riscv engine");
Define_bool_opt("--use-fp16", g_flag_use_fp16, false, "infer with riscv fp16 (use fp32 by default)");
Define_int32_opt(
    "--wg-level", g_flag_wg_level, 1,
    "select winograd level[0-4]. 0: wingorad off. 1: turn on winograd and automatically select block size. 2: use "
    "winograd block 2 if possible. 3: use winograd block 4 if possible. 4: use winograd block 6 if possible");
Define_int32_opt("--tuning-level", g_flag_tuning_level, 0, "select conv algo dynamic tuning level[0-1]. 0: off. 1: on");

#include "ppl/nn/engines/riscv/engine_factory.h"
#include "ppl/nn/engines/riscv/options.h"
#include "ppl/nn/engines/riscv/engine_options.h"

static bool RegisterRiscvEngine(vector<unique_ptr<Engine>>* engines) {
    riscv::EngineOptions options;
    options.tune_param_flag = false;

    if (g_flag_mm_policy == "perf") {
        options.mm_policy = riscv::MM_MRU;
    } else if (g_flag_mm_policy == "mem") {
        options.mm_policy = riscv::MM_COMPACT;
    } else if (g_flag_mm_policy == "plain") {
        options.mm_policy = riscv::MM_PLAIN;
    } else {
        LOG(ERROR) << "unknown --mm-policy option: " << g_flag_mm_policy;
        return false;
    }

    if (g_flag_use_fp16) {
        options.forward_precision = DATATYPE_FLOAT16;
    } else {
        options.forward_precision = DATATYPE_FLOAT32;
    }
    options.dynamic_tuning_level = g_flag_tuning_level;
    options.winograd_level = g_flag_wg_level;

    auto riscv_engine = riscv::EngineFactory::Create(options);
    // configure engine
    engines->emplace_back(unique_ptr<Engine>(riscv_engine));
    LOG(INFO) << "***** register RiscvEngine *****";
    return true;
}

#endif

#ifdef PPLNN_USE_ARM

Define_bool_opt("--use-arm", g_flag_use_arm, false, "use arm engine");
Define_bool_opt("--use-fp16", g_flag_use_fp16, false, "infer with armv8.2 fp16");
Define_int32_opt("--wg-level", g_flag_wg_level, 3,
                 "select winograd level[0-3]. 0: wingorad off. 1: turn on winograd and automatically select block "
                 "size. 2: use winograd block 2 if possible. 3: use winograd block 4 if possible");
Define_int32_opt("--tuning-level", g_flag_tuning_level, 1, "select conv algo dynamic tuning level[0-1]. 0: off. 1: on");
Define_int32_opt("--numa-node-id", g_flag_numa_node_id, -1,
                 "bind arm engine to specified numa node, range [0, numa_max_node), -1 means not bind");

#include "ppl/nn/engines/arm/engine_factory.h"

static bool RegisterArmEngine(vector<unique_ptr<Engine>>* engines) {
    arm::EngineOptions options;
    if (g_flag_mm_policy == "perf") {
        options.mm_policy = arm::MM_MRU;
    } else if (g_flag_mm_policy == "mem") {
        options.mm_policy = arm::MM_COMPACT;
    } else if (g_flag_mm_policy == "plain") {
        options.mm_policy = arm::MM_PLAIN;
    } else {
        LOG(ERROR) << "unknown --mm-policy option: " << g_flag_mm_policy;
        return false;
    }

    if (g_flag_use_fp16) {
        options.forward_precision = DATATYPE_FLOAT16;
    } else {
        options.forward_precision = DATATYPE_FLOAT32;
    }
    options.graph_optimization_level = arm::OPT_ENABLE_ALL;
    options.winograd_level = g_flag_wg_level;
    options.dynamic_tuning_level = g_flag_tuning_level;
    options.numa_node_id = g_flag_numa_node_id;

    auto arm_engine = arm::EngineFactory::Create(options);
    // configure engine
    engines->emplace_back(unique_ptr<Engine>(arm_engine));
    LOG(INFO) << "***** register ArmEngine *****";
    return true;
}
#endif

static bool RegisterEngines(vector<unique_ptr<Engine>>* engines) {
#ifdef PPLNN_USE_X86
    if (g_flag_use_x86) {
        bool ok = RegisterX86Engine(engines);
        if (!ok) {
            LOG(ERROR) << "RegisterX86Engine failed.";
            return false;
        }
    }
#endif

#ifdef PPLNN_USE_CUDA
    if (g_flag_use_cuda) {
        bool ok = RegisterCudaEngine(engines);
        if (!ok) {
            LOG(ERROR) << "RegisterCudaEngine failed.";
            return false;
        }
    }
#endif

#ifdef PPLNN_USE_RISCV
    if (g_flag_use_riscv) {
        bool ok = RegisterRiscvEngine(engines);
        if (!ok) {
            LOG(ERROR) << "RegisterCudaEngine failed.";
            return false;
        }
    }
#endif

#ifdef PPLNN_USE_ARM
    if (g_flag_use_arm) {
        bool ok = RegisterArmEngine(engines);
        if (!ok) {
            LOG(ERROR) << "RegisterArmEngine failed.";
            return false;
        }
    }
#endif

    if (engines->empty()) {
        LOG(ERROR) << "no engine is registered. run `./pplnn --help` to see supported engines marked with '--use-*', "
                   << "or see documents listed in README.md for building instructions.";
        return false;
    }

    return true;
}

/* -------------------------------------------------------------------------- */

static string GetDimsStr(const Tensor* tensor) {
    auto shape = tensor->GetShape();
    if (shape->GetRealDimCount() == 0) {
        return string();
    }

    string res = ToString(shape->GetDim(0));
    for (uint32_t i = 1; i < shape->GetDimCount(); ++i) {
        res += "_" + ToString(shape->GetDim(i));
    }

    return res;
}

static bool SetRandomInputs(const vector<vector<int64_t>>& input_shapes, Runtime* runtime, vector<string>* input_data) {
    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto shape = t->GetShape();

        if (input_shapes.empty()) {
            auto dim_count = shape->GetRealDimCount();
            if (dim_count == 0) {
                continue;
            }

            auto dims = GenerateRandomDims(dim_count);

            if (shape->GetDim(0) == INVALID_DIM_VALUE) {
                shape->SetDim(0, 1);
            }
            for (uint32_t j = 1; j < dim_count; ++j) {
                if (shape->GetDim(j) == INVALID_DIM_VALUE) {
                    shape->SetDim(j, dims[j]);
                }
            }
        } else {
            shape->Reshape(input_shapes[c]);
        }

        auto nr_element = shape->CalcBytesIncludingPadding() / sizeof(float);
        vector<float> buffer(nr_element);

        std::default_random_engine eng;
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (uint32_t i = 0; i < nr_element; ++i) {
            buffer[i] = dis(eng);
        }

        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto status = t->ConvertFromHost(buffer.data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set tensor[" << t->GetName() << "] content failed: " << GetRetCodeStr(status);
            return false;
        }

        input_data->emplace_back(string((const char*)buffer.data(), buffer.size() * sizeof(float)));
    }

    return true;
}

static string GetBasename(const string& path) {
    string last_entry;
    SplitString(path.data(), path.size(), "/", 1, [&last_entry](const char* s, unsigned int l) -> bool {
        if (l > 0) {
            last_entry.assign(s, l);
        }
        return true;
    });
    return last_entry;
}

static bool SetInputsAllInOne(const string& input_file, const vector<vector<int64_t>>& input_shapes, Runtime* runtime,
                              vector<string>* input_data) {
    Mmap fm;
    auto status = fm.Init(input_file.c_str(), Mmap::READ);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "mapping file[" << input_file << "] failed.";
        return false;
    }

    auto data = fm.GetData();
    const string file_name = GetBasename(input_file);
    uint64_t expect_input_size = 0;
    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);

        if (!input_shapes.empty()) {
            t->GetShape()->Reshape(input_shapes[c]);
        }

        expect_input_size += t->GetShape()->CalcBytesExcludingPadding();
    }
    if (fm.GetSize() < expect_input_size) {
        LOG(ERROR) << "input file[" << file_name << "] size(" << fm.GetSize() << ") is less than expected size("
                   << expect_input_size << ")";
        return false;
    }
    if (fm.GetSize() > expect_input_size) {
        LOG(WARNING) << "input file[" << file_name << "] size(" << fm.GetSize() << ") is bigger than expected size("
                     << expect_input_size << ")";
    }

    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);

        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto status = t->ConvertFromHost(data, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert tensor[" << t->GetName() << "] content failed: " << GetRetCodeStr(status);
            return false;
        }

        const uint64_t content_size = src_desc.CalcBytesIncludingPadding();
        input_data->emplace_back(string(data, content_size));
        data += content_size;
    }

    return true;
}

static const pair<string, datatype_t> g_str2datatype[] = {
    {"fp64", DATATYPE_FLOAT64}, {"fp32", DATATYPE_FLOAT32}, {"fp16", DATATYPE_FLOAT16}, {"int32", DATATYPE_INT32},
    {"int64", DATATYPE_INT64},  {"bool", DATATYPE_BOOL},    {"", DATATYPE_UNKNOWN},
};

static datatype_t FindDataTypeByStr(const string& str) {
    for (int i = 0; !g_str2datatype[i].first.empty(); ++i) {
        if (str == g_str2datatype[i].first) {
            return g_str2datatype[i].second;
        }
    }
    return DATATYPE_UNKNOWN;
}

static const char* FindDataTypeStr(datatype_t dt) {
    for (int i = 0; !g_str2datatype[i].first.empty(); ++i) {
        if (g_str2datatype[i].second == dt) {
            return g_str2datatype[i].first.c_str();
        }
    }
    return nullptr;
}

static bool SetInputsOneByOne(const string& input_files_str, const vector<vector<int64_t>>& input_shapes,
                              Runtime* runtime, vector<string>* input_data) {
    vector<string> files;
    SplitString(input_files_str.data(), input_files_str.size(), ",", 1,
                [&files](const char* s, unsigned int l) -> bool {
                    if (l > 0) {
                        files.push_back(string(s, l));
                    }
                    return true;
                });
    if (files.size() != runtime->GetInputCount()) {
        LOG(ERROR) << "input file num[" << files.size() << "] != input count[" << runtime->GetInputCount() << "]";
        return false;
    }

    for (uint32_t i = 0; i < files.size(); ++i) {
        const string& file_full_path = files[i];
        const string file_name = GetBasename(file_full_path);

        Mmap fm;
        auto status = fm.Init(file_full_path.c_str(), Mmap::READ);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "mapping file[" << file_full_path << "] failed.";
            return false;
        }

        auto t = runtime->GetInputTensor(i);

        if (!input_shapes.empty()) {
            t->GetShape()->Reshape(input_shapes[i]);
        }

        ppl::nn::TensorShape src_desc = *t->GetShape();
        auto tensor_size = src_desc.CalcBytesIncludingPadding();
        if (fm.GetSize() < tensor_size) {
            LOG(ERROR) << "input file[" << file_name << "] size(" << fm.GetSize() << ") is less than tensor["
                       << t->GetName() << "] size(" << tensor_size << ")";
            return false;
        }
        if (fm.GetSize() > tensor_size) {
            LOG(WARNING) << "input file[" << file_name << "] size(" << fm.GetSize() << ") is bigger than tensor["
                         << t->GetName() << "] size(" << tensor_size << ")";
        }
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        status = t->ConvertFromHost(fm.GetData(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set input[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        input_data->emplace_back(string(fm.GetData(), fm.GetSize()));
    }

    return true;
}

static bool SetReshapedInputsOneByOne(const string& input_files_str, Runtime* runtime, vector<string>* input_data) {
    vector<string> files;
    SplitString(input_files_str.data(), input_files_str.size(), ",", 1,
                [&files](const char* s, unsigned int l) -> bool {
                    if (l > 0) {
                        files.push_back(string(s, l));
                    }
                    return true;
                });
    if (files.size() != runtime->GetInputCount()) {
        LOG(ERROR) << "input file num[" << files.size() << "] != graph input num[" << runtime->GetInputCount() << "]";
        return false;
    }

    for (uint32_t c = 0; c < files.size(); ++c) {
        const string& file_full_path = files[c];
        const string file_name = GetBasename(file_full_path);

        vector<string> conponents;
        SplitString(file_name.data(), file_name.size(), "-", 1, [&conponents](const char* s, unsigned int l) -> bool {
            if (l > 0) {
                conponents.push_back(string(s, l));
            } else {
                conponents.push_back(string());
            }
            return true;
        });
        if (conponents.size() != 3) {
            LOG(ERROR) << "illegal file name format: " << file_name;
            return false;
        }

        string data_type_str;
        SplitString(conponents[2].data(), conponents[2].size(), ".", 1,
                    [&data_type_str](const char* s, unsigned int l) -> bool {
                        if (l > 0) {
                            data_type_str.assign(s, l);
                        }
                        return false;
                    });

        const string& dims_str = conponents[1];
        vector<int64_t> dims;
        if (!dims_str.empty()) {
            SplitString(dims_str.data(), dims_str.size(), "_", 1, [&dims](const char* s, unsigned int l) -> bool {
                if (l > 0) {
                    int64_t dim = atol(string(s, l).c_str());
                    dims.push_back(dim);
                    return true;
                }
                LOG(ERROR) << "illegal dim format.";
                return false;
            });
        }

        auto data_type = FindDataTypeByStr(data_type_str);
        if (data_type == DATATYPE_UNKNOWN) {
            LOG(ERROR) << "cannot find data type[" << data_type_str << "] in input file[" << file_name << "]";
            return false;
        }

        ppl::nn::TensorShape input_shape;
        input_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        input_shape.SetDataType(data_type);
        input_shape.Reshape(dims.data(), dims.size());

        Mmap fm;
        auto status = fm.Init(file_full_path.c_str(), Mmap::READ);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "mapping file[" << file_full_path << "] failed.";
            return false;
        }

        auto t = runtime->GetInputTensor(c);
        *t->GetShape() = input_shape;

        ppl::nn::TensorShape src_desc = *t->GetShape();
        auto tensor_size = src_desc.CalcBytesIncludingPadding();
        if (fm.GetSize() < tensor_size) {
            LOG(ERROR) << "input file[" << file_name << "] size(" << fm.GetSize() << ") is less than tensor["
                       << t->GetName() << "] size(" << tensor_size << ")";
            return false;
        }
        if (fm.GetSize() > tensor_size) {
            LOG(WARNING) << "input file[" << file_name << "] size(" << fm.GetSize() << ") is bigger than tensor["
                         << t->GetName() << "] size(" << tensor_size << ")";
        }
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        status = t->ConvertFromHost(fm.GetData(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set input[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        input_data->emplace_back(string(fm.GetData(), fm.GetSize()));
    }

    return true;
}

static bool SaveInputsOneByOne(const Runtime* runtime) {
    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto shape = t->GetShape();

        auto bytes = shape->CalcBytesIncludingPadding();
        vector<char> buffer(bytes);

        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto status = t->ConvertToHost(buffer.data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data failed: " << GetRetCodeStr(status);
            return false;
        }

        const char* data_type_str = FindDataTypeStr(shape->GetDataType());
        if (!data_type_str) {
            LOG(ERROR) << "unsupported data type[" << GetDataTypeStr(shape->GetDataType()) << "]";
            return false;
        }

        char name_prefix[32];
        sprintf(name_prefix, "pplnn_input_%05u_", c);
        const string in_file_name = g_flag_save_data_dir + "/" + string(name_prefix) + t->GetName() + "-" +
            GetDimsStr(t) + "-" + string(data_type_str) + ".dat";
        ofstream ofs(in_file_name, ios_base::out | ios_base::binary | ios_base::trunc);
        if (!ofs.is_open()) {
            LOG(ERROR) << "save input file[" << in_file_name << "] failed.";
            return false;
        }

        ofs.write(buffer.data(), bytes);
    }

    return true;
}

static bool SaveInputsAllInOne(const Runtime* runtime) {
    const string in_file_name = g_flag_save_data_dir + "/pplnn_input.dat";
    ofstream ofs(in_file_name, ios_base::out | ios_base::binary | ios_base::trunc);
    if (!ofs.is_open()) {
        LOG(ERROR) << "open file[" << in_file_name << "] failed.";
        return false;
    }

    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto bytes = t->GetShape()->CalcBytesIncludingPadding();
        vector<char> buffer(bytes);

        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto status = t->ConvertToHost((void*)buffer.data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data failed: " << GetRetCodeStr(status);
            return false;
        }

        ofs.write(buffer.data(), bytes);
    }

    return true;
}

static bool SaveOutputsOneByOne(const Runtime* runtime) {
    for (uint32_t c = 0; c < runtime->GetOutputCount(); ++c) {
        auto t = runtime->GetOutputTensor(c);

        ppl::nn::TensorShape dst_desc = *t->GetShape();
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        // convert fp16 to fp32
        if (dst_desc.GetDataType() == DATATYPE_FLOAT16) {
            dst_desc.SetDataType(DATATYPE_FLOAT32);
        }

        auto bytes = dst_desc.CalcBytesIncludingPadding();
        vector<char> buffer(bytes);
        auto status = t->ConvertToHost(buffer.data(), dst_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data of tensor[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        const string out_file_name = g_flag_save_data_dir + "/pplnn_output-" + t->GetName() + ".dat";
        ofstream ofs(out_file_name, ios_base::out | ios_base::binary | ios_base::trunc);
        if (!ofs.is_open()) {
            LOG(ERROR) << "open output file[" << out_file_name << "]";
            return false;
        }

        ofs.write(buffer.data(), bytes);
    }

    return true;
}

static void PrintInputInfo(const Runtime* runtime) {
    cout << "----- input info -----" << endl;
    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto tensor = runtime->GetInputTensor(i);
        cout << "input[" << i << "]:" << endl;
        cout << "    name: " << tensor->GetName() << endl;

        string dims_str;
        auto shape = tensor->GetShape();
        for (uint32_t j = 0; j < shape->GetDimCount(); ++j) {
            dims_str += " " + ToString(shape->GetDim(j));
        }
        cout << "    dim(s):" << dims_str << endl;

        cout << "    data type: " << GetDataTypeStr(shape->GetDataType()) << endl;
        cout << "    data format: " << GetDataFormatStr(shape->GetDataFormat()) << endl;
        cout << "    byte(s) excluding padding: " << shape->CalcBytesExcludingPadding() << endl;
    }

    cout << "----------------------" << endl;
}

static void PrintOutputInfo(const Runtime* runtime) {
    cout << "----- output info -----" << endl;
    for (uint32_t i = 0; i < runtime->GetOutputCount(); ++i) {
        auto tensor = runtime->GetOutputTensor(i);
        cout << "output[" << i << "]:" << endl;
        cout << "    name: " << tensor->GetName() << endl;

        string dims_str;
        auto shape = tensor->GetShape();
        for (uint32_t j = 0; j < shape->GetDimCount(); ++j) {
            dims_str += " " + ToString(shape->GetDim(j));
        }
        cout << "    dim(s):" << dims_str << endl;

        cout << "    data type: " << GetDataTypeStr(shape->GetDataType()) << endl;
        cout << "    data format: " << GetDataFormatStr(shape->GetDataFormat()) << endl;
        cout << "    byte(s) excluding padding: " << shape->CalcBytesExcludingPadding() << endl;

        datatype_t saved_data_type = shape->GetDataType();
        if (saved_data_type == DATATYPE_FLOAT16) {
            saved_data_type = DATATYPE_FLOAT32;
        }
        cout << "    saved data type: " << GetDataTypeStr(saved_data_type) << endl;
    }

    cout << "----------------------" << endl;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
static void PrintProfilingStatistics(const ProfilingStatistics& stat, double run_dur, int32_t run_count) {
    std::map<std::string, std::pair<double, double>> type_stat;
    std::map<std::string, int> type_count;
    char float_buf_0[128];
    char float_buf_1[128];
    LOG(INFO) << "----- OP statistics by Node -----";
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
        sprintf(float_buf_0, "%8.4f", avg_time);
        string temp = x->name;
        temp.insert(temp.length(), temp.length() > 50 ? 0 : 50 - temp.length(), ' ');
        LOG(INFO) << "NAME: [" << temp << "], "
                  << "AVG_TIME: [" << float_buf_0 << "], "
                  << "EXEC_COUNT: [" << x->exec_count << "]";
    }
    LOG(INFO) << "----- OP statistics by OpType -----";
    double tot_kernel_time = 0;
    for (auto it = type_stat.begin(); it != type_stat.end(); ++it) {
        tot_kernel_time += it->second.second;
    }
    for (auto it = type_stat.begin(); it != type_stat.end(); ++it) {
        sprintf(float_buf_0, "%8.4f", it->second.first);
        sprintf(float_buf_1, "%8.4f", it->second.second / tot_kernel_time * 100);
        string temp = it->first;
        temp.insert(temp.length(), temp.length() > 20 ? 0 : 20 - temp.length(), ' ');
        LOG(INFO) << "TYPE: [" << temp << "], AVG_TIME: [" << float_buf_0 << "], Percentage: [" << float_buf_1
                  << "], excute times [" << type_count[it->first] << "]";
    }

    LOG(INFO) << "----- TOTAL statistics -----";
    sprintf(float_buf_0, "%8.4f", tot_kernel_time / run_count);
    sprintf(float_buf_1, "%8.4f", run_dur / run_count);
    LOG(INFO) << "RUN_COUNT: [" << run_count << "]";
    LOG(INFO) << "AVG_KERNEL_TIME: [" << float_buf_0 << "]";
    LOG(INFO) << "AVG_RUN_TIME: [" << float_buf_1 << "]";
    sprintf(float_buf_0, "%8.4f", tot_kernel_time);
    sprintf(float_buf_1, "%8.4f", run_dur);
    LOG(INFO) << "TOT_KERNEL_TIME: [" << float_buf_0 << "]";
    LOG(INFO) << "TOT_RUN_TIME: [" << float_buf_1 << "]";
    sprintf(float_buf_0, "%8.4f%%", (run_dur - tot_kernel_time) / run_dur * 100);
    LOG(INFO) << "SCHED_LOST: [" << float_buf_0 << "]";
}
#endif

static bool SetInputs(const vector<string>& input_data, Runtime* runtime) {
    if (input_data.size() != runtime->GetInputCount()) {
        LOG(ERROR) << "number of input data [" << input_data.size() << "] != runtime input count ["
                   << runtime->GetInputCount() << "]";
        return false;
    }

    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto t = runtime->GetInputTensor(i);
        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto status = t->ConvertFromHost(input_data[i].data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set input [" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }
    }

    return true;
}

static bool GetOutputs(const Runtime* runtime) {
    for (uint32_t c = 0; c < runtime->GetOutputCount(); ++c) {
        auto t = runtime->GetOutputTensor(c);

        ppl::nn::TensorShape dst_desc = *t->GetShape();
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        if (dst_desc.GetDataType() == DATATYPE_FLOAT16) {
            dst_desc.SetDataType(DATATYPE_FLOAT32);
        }
        auto bytes = dst_desc.CalcBytesIncludingPadding();
        vector<char> buffer(bytes);
        auto status = t->ConvertToHost(buffer.data(), dst_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data of tensor[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }
    }

    return true;
}

static bool Profiling(const vector<string>& input_data, Runtime* runtime) {
    if (g_flag_warmup_iterations > 0) {
        LOG(INFO) << "Warm up start for " << g_flag_warmup_iterations << " times.";
        for (uint32_t i = 0; i < g_flag_warmup_iterations; ++i) {
            runtime->Run();
        }
        LOG(INFO) << "Warm up end.";
    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    auto status = runtime->Configure(RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG, true);
    if (status != RC_SUCCESS) {
        LOG(WARNING) << "enable profiling failed: " << GetRetCodeStr(status);
    }
#endif
    LOG(INFO) << "Profiling start";

    double run_dur = 0;
    uint32_t run_count = 0;
    while (run_dur < g_flag_min_profiling_seconds * 1000 || run_count < g_flag_min_profiling_iterations) {
        auto run_begin_ts = std::chrono::system_clock::now();
        if (g_flag_perf_with_io) {
            SetInputs(input_data, runtime);
        }
        runtime->Run();
        if (g_flag_perf_with_io) {
            GetOutputs(runtime);
        }
        auto run_end_ts = std::chrono::system_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
        run_dur += (double)diff.count() / 1000;
        run_count += 1;
    }

    LOG(INFO) << "Total duration: " << run_dur << " ms";

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    ProfilingStatistics stat;
    status = runtime->GetProfilingStatistics(&stat);
    if (status != RC_SUCCESS) {
        LOG(WARNING) << "Get profiling statistics failed: " << GetRetCodeStr(status);
    }
    PrintProfilingStatistics(stat, run_dur, run_count);
#else
    LOG(INFO) << "Average run costs: " << (run_dur / run_count) << " ms.";
#endif

    LOG(INFO) << "Profiling End";
    return true;
}

static uint32_t CalcModelNum() {
    uint32_t counter = 0;
#ifdef PPLNN_ENABLE_ONNX_MODEL
    if (!g_flag_onnx_model.empty()) {
        ++counter;
    }
#endif
#ifdef PPLNN_ENABLE_PMX_MODEL
    if (!g_flag_pmx_model.empty()) {
        ++counter;
    }
#endif
    return counter;
}

int main(int argc, char* argv[]) {
    RetCode status;

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

    if (g_flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    cout << "ppl.nn version: [" << PPLNN_VERSION_MAJOR << "." << PPLNN_VERSION_MINOR << "." << PPLNN_VERSION_PATCH
         << "], commit: [" << PPLNN_COMMIT_STR << "]" << endl;

    if (g_flag_version) {
        return 0;
    }

#ifdef PPLNN_ENABLE_PMX_MODEL
    if (!g_flag_save_pmx_model.empty()) {
        LOG(ERROR) << "`--save-pmx-model` is deprecated. use `--export-pmx-model` instead.";
        return -1;
    }
#endif

    auto nr_model = CalcModelNum();
    if (nr_model == 0) {
        LOG(ERROR) << "please specify a model.";
        return -1;
    }
    if (nr_model > 1) {
        LOG(ERROR) << "multiple model options are specified.";
        return -1;
    }

    auto prepare_begin_ts = std::chrono::system_clock::now();

    vector<unique_ptr<Engine>> engines;
    if (!RegisterEngines(&engines)) {
        LOG(ERROR) << "RegisterEngines failed.";
        return -1;
    }

    unique_ptr<Runtime> runtime;

    if (false) {
    }
#ifdef PPLNN_ENABLE_ONNX_MODEL
    else if (!g_flag_onnx_model.empty()) {
        auto builder = unique_ptr<onnx::RuntimeBuilder>(onnx::RuntimeBuilderFactory::Create());
        if (!builder) {
            LOG(ERROR) << "create RuntimeBuilder failed.";
            return -1;
        }

        status = builder->LoadModel(g_flag_onnx_model.c_str());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "create OnnxRuntimeBuilder failed: " << GetRetCodeStr(status);
            return -1;
        }

        vector<Engine*> engine_ptrs(engines.size());
        for (uint32_t i = 0; i < engines.size(); ++i) {
            engine_ptrs[i] = engines[i].get();
        }

        onnx::RuntimeBuilder::Resources resources;
        resources.engines = engine_ptrs.data();
        resources.engine_num = engine_ptrs.size();

        status = builder->SetResources(resources);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "onnx RuntimeBuilder SetResources failed: " << GetRetCodeStr(status);
            return -1;
        }

        status = builder->Preprocess();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "onnx preprocess failed: " << GetRetCodeStr(status);
            return -1;
        }

#ifdef PPLNN_ENABLE_PMX_MODEL
        if (!g_flag_export_pmx_model.empty()) {
            utils::FileDataStream fds;
            auto status = fds.Init(g_flag_export_pmx_model.c_str());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "open pmx output file failed: " << GetRetCodeStr(status);
                return -1;
            }
            pmx::SaveModelOptions opt;
            if (!g_flag_pmx_external_data_dir.empty() && !g_flag_pmx_external_data_file.empty()) {
                LOG(ERROR) << "only one of `--pmx-external-data-dir` and `pmx-external-data-file` can be set.";
                return -1;
            }
            if (!g_flag_pmx_external_data_dir.empty()) {
                opt.external_data_dir = g_flag_pmx_external_data_dir.c_str();
            } else if (!g_flag_pmx_external_data_file.empty()) {
                opt.external_data_file = g_flag_pmx_external_data_file.c_str();
            }
            status = builder->Serialize("pmx", &opt, &fds);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "save ppl model failed: " << GetRetCodeStr(status);
                return -1;
            }
        }
#endif

        runtime.reset(builder->CreateRuntime());
    }
#endif
#ifdef PPLNN_ENABLE_PMX_MODEL
    else if (!g_flag_pmx_model.empty()) {
        auto builder = unique_ptr<pmx::RuntimeBuilder>(pmx::RuntimeBuilderFactory::Create());
        if (!builder) {
            LOG(ERROR) << "create PmxRuntimeBuilder failed.";
            return -1;
        }

        vector<Engine*> engine_ptrs(engines.size());
        for (uint32_t i = 0; i < engines.size(); ++i) {
            engine_ptrs[i] = engines[i].get();
        }

        pmx::RuntimeBuilder::Resources resources;
        resources.engines = engine_ptrs.data();
        resources.engine_num = engine_ptrs.size();

        Mmap constant_file_buf;
        pmx::LoadModelOptions opt;
        if (!g_flag_pmx_external_data_dir.empty() && !g_flag_pmx_external_data_file.empty()) {
            LOG(ERROR) << "only one of `--pmx-external-data-dir` and `pmx-external-data-file` can be set.";
            return -1;
        }
        if (!g_flag_pmx_external_data_dir.empty()) {
            opt.external_data_dir = g_flag_pmx_external_data_dir.c_str();
        } else if (!g_flag_pmx_external_data_file.empty()) {
            auto rc = constant_file_buf.Init(g_flag_pmx_external_data_file.c_str(), Mmap::READ);
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "mmap weight file [" << g_flag_pmx_external_data_file << "] failed.";
                return -1;
            }
            opt.external_buffer = constant_file_buf.GetData();
            opt.external_buffer_size = constant_file_buf.GetSize();
        }

        auto status = builder->LoadModel(g_flag_pmx_model.c_str(), resources, opt);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "PmxRuntimeBuilder LoadModel failed: " << GetRetCodeStr(status);
            return -1;
        }

        status = builder->Preprocess();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "pmx preprocess failed: " << GetRetCodeStr(status);
            return -1;
        }

        if (!g_flag_export_pmx_model.empty()) {
            utils::FileDataStream fds;
            auto status = fds.Init(g_flag_export_pmx_model.c_str());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "open pmx output file failed: " << GetRetCodeStr(status);
                return -1;
            }

            pmx::SaveModelOptions opt;
            if (!g_flag_pmx_external_data_dir.empty() && !g_flag_pmx_external_data_file.empty()) {
                LOG(ERROR) << "only one of `--pmx-external-data-dir` and `pmx-external-data-file` can be set.";
                return -1;
            }
            if (!g_flag_pmx_external_data_dir.empty()) {
                opt.external_data_dir = g_flag_pmx_external_data_dir.c_str();
            } else if (!g_flag_pmx_external_data_file.empty()) {
                opt.external_data_file = g_flag_pmx_external_data_file.c_str();
            }

            status = builder->Serialize("pmx", &opt, &fds);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "save ppl model failed: " << GetRetCodeStr(status);
                return -1;
            }
        }

        runtime.reset(builder->CreateRuntime());
    }
#endif

    if (!runtime) {
        LOG(ERROR) << "CreateRuntime failed.";
        return -1;
    }

#if PPLNN_USE_CUDA
    if (g_flag_enable_cuda_graph) {
        Engine* cuda_engine = nullptr;
        for (auto& engine : engines) {
            if (!strcmp(engine->GetName(), "cuda")) {
                cuda_engine = engine.get();
                break;
            }
        }
        auto status = cuda::SetGraphScheduler(runtime.get(), cuda_engine);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Set Graph Scheduler failed.";
            return -1;
        }
    }
#endif

    vector<vector<int64_t>> input_shapes;
    if (!g_flag_input_shapes.empty()) {
        if (!ParseInputShapes(g_flag_input_shapes, &input_shapes)) {
            LOG(ERROR) << "ParseInputShapes failed.";
            return -1;
        }
        if (input_shapes.size() != runtime->GetInputCount()) {
            LOG(ERROR) << "the number of input shapes [" << input_shapes.size() << "] != required input count ["
                       << runtime->GetInputCount() << "]";
            return -1;
        }
    }

    vector<string> input_data; // store input data for profiling
    if (!g_flag_input.empty()) {
        if (!SetInputsAllInOne(g_flag_input, input_shapes, runtime.get(), &input_data)) {
            LOG(ERROR) << "SetInputsAllInOne failed.";
            return -1;
        }
    } else if (!g_flag_inputs.empty()) {
        if (!SetInputsOneByOne(g_flag_inputs, input_shapes, runtime.get(), &input_data)) {
            LOG(ERROR) << "SetInputsOneByOne failed.";
            return -1;
        }
    } else if (!g_flag_reshaped_inputs.empty()) {
        if (!SetReshapedInputsOneByOne(g_flag_reshaped_inputs, runtime.get(), &input_data)) {
            LOG(ERROR) << "SetReshapedInputsOneByOne failed.";
            return -1;
        }
    } else {
        if (!SetRandomInputs(input_shapes, runtime.get(), &input_data)) {
            LOG(ERROR) << "SetRandomInputs failed.";
            return -1;
        }
    }

    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto in = runtime->GetInputTensor(i);
        auto shape = in->GetShape();
        if (shape->CalcElementsIncludingPadding() == 0) {
            // We should allow empty input
            LOG(WARNING) << "input tensor[" << in->GetName() << "] is empty.";
        }
    }

    if (g_flag_save_input) {
        if (!SaveInputsAllInOne(runtime.get())) {
            return -1;
        }
    } else if (g_flag_save_inputs) {
        if (!SaveInputsOneByOne(runtime.get())) {
            return -1;
        }
    }

    PrintInputInfo(runtime.get());

    if (g_flag_no_run) {
        return 0;
    }

    auto prepare_end_ts = std::chrono::system_clock::now();
    auto prepare_diff = std::chrono::duration_cast<std::chrono::microseconds>(prepare_end_ts - prepare_begin_ts);
    LOG(INFO) << "Prepare costs: " << (float)prepare_diff.count() / 1000 << " ms.";

    auto run_begin_ts = std::chrono::system_clock::now();
    status = runtime->Run();
    auto run_end_ts = std::chrono::system_clock::now();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Run() failed: " << GetRetCodeStr(status);
        return -1;
    }

    PrintOutputInfo(runtime.get());

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    LOG(INFO) << "Run() costs: " << (float)diff.count() / 1000 << " ms.";

    if (g_flag_save_outputs) {
        if (!SaveOutputsOneByOne(runtime.get())) {
            return -1;
        }
    }

    LOG(INFO) << "Run ok";

    if (g_flag_enable_profiling) {
        if (!Profiling(input_data, runtime.get())) {
            LOG(ERROR) << "Profiling() failed.";
            return -1;
        }
    }

    return 0;
}
