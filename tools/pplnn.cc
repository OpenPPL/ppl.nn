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

#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/version.h"
#include "ppl/common/file_mapping.h"
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
using namespace ppl::nn;
using namespace ppl::common;
using namespace std;

/* -------------------------------------------------------------------------- */

#include "simple_flags.h"

Define_bool_opt("--help", g_flag_help, false, "show these help information");
Define_bool_opt("--version", g_flag_version, false, "show version info");

Define_string_opt("--onnx-model", g_flag_onnx_model, "", "onnx model file");

Define_string_opt("--mm-policy", g_flag_mm_policy, "mem",
                  "\"perf\" => better performance, or \"mem\" => less memory usage");

Define_bool_opt("--enable-profiling", g_flag_enable_profiling, false, "enable profiling and print profiling info");
Define_float_opt("--min-profiling-time", g_flag_min_profiling_time, 1.0f, "min execute time by seconds for profiling");
Define_uint32_opt("--warmuptimes", g_flag_warmup_times, 0, "declare warmup times");

Define_string_opt("--input", g_flag_input, "", "binary input file containing all tensors' data");
Define_string_opt("--inputs", g_flag_inputs, "", "binary input files separated by comma");
Define_string_opt("--reshaped-inputs", g_flag_reshaped_inputs, "",
                  "binary input files separated by comma."
                  " file name format: 'name-dims-datatype.dat'. for example:"
                  " input1-1_1_1_1-fp32.dat,input2-1_1_1_1-fp16.dat,input3-1_1-int8.dat");
Define_string_opt("--in-shapes", g_flag_input_shapes, "",
                  "shapes of input tensors."
                  " dims are separated by underline, inputs are separated by comma. example:"
                  " 1_3_128_128,2_3_400_640,3_3_768_1024");

Define_bool_opt("--save-input", g_flag_save_input, false, "save input tensors in one file in NDARRAY format");
Define_bool_opt("--save-inputs", g_flag_save_inputs, false, "save separated input tensors in NDARRAY format");
Define_bool_opt("--save-outputs", g_flag_save_outputs, false, "save separated output tensors in NDARRAY format");
Define_string_opt("--save-data-dir", g_flag_save_data_dir, ".",
                  "directory to save input/output data if '--save-*' options are enabled.");

/* -------------------------------------------------------------------------- */

template <typename T>
string ToString(T v) {
    stringstream ss;
    ss << v;
    return ss.str();
}

#ifdef PPLNN_USE_CUDA

Define_bool_opt("--use-cuda", g_flag_use_cuda, false, "use cuda engine");

Define_string_opt("--output-format", g_flag_output_format, "", "declare the output format");
Define_string_opt("--output-type", g_flag_output_type, "", "declare the output type");
Define_string_opt("--dims", g_flag_compiler_dims, "",
                  "declare init input dims for algo selection (split with comma)."
                  " for example: 1_3_224_224,1_3_128_640");
Define_bool_opt("--quick-select", g_flag_quick_select, false, "quick select algorithms for conv and gemm kernel");
Define_uint32_opt("--device-id", g_flag_device_id, 0, "declare device id for cuda");

Define_string_opt("--algo-info", g_algo_info, "", "declare best algo index for certain conv input shape");

#include "ppl/nn/engines/cuda/engine_factory.h"
#include "ppl/nn/engines/cuda/cuda_options.h"

static inline bool RegisterCudaEngine(vector<unique_ptr<Engine>>* engines) {
    CudaEngineOptions options;
    options.device_id = g_flag_device_id;

    if (g_flag_mm_policy == "perf") {
        options.mm_policy = CUDA_MM_BEST_FIT;
    } else if (g_flag_mm_policy == "mem") {
        options.mm_policy = CUDA_MM_COMPACT;
    }

    auto cuda_engine = CudaEngineFactory::Create(options);
    if (!cuda_engine) {
        return false;
    }

    cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_OUTPUT_FORMAT, g_flag_output_format.c_str());
    cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_OUTPUT_TYPE, g_flag_output_type.c_str());
    cuda_engine->Configure(ppl::nn::CUDA_CONF_USE_DEFAULT_ALGORITHMS, g_flag_quick_select);
    cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_ALGORITHM, g_algo_info.c_str());

    if (!g_flag_compiler_dims.empty()) {
        cuda_engine->Configure(ppl::nn::CUDA_CONF_SET_COMPILER_INPUT_SHAPE, g_flag_compiler_dims.c_str());
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

#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/x86/x86_options.h"
#include "ppl/kernel/x86/common/threading_tools.h"
static inline bool RegisterX86Engine(vector<unique_ptr<Engine>>* engines) {
    X86EngineOptions options;
    if (g_flag_mm_policy == "perf") {
        options.mm_policy = X86_MM_MRU;
    } else if (g_flag_mm_policy == "mem") {
        options.mm_policy = X86_MM_COMPACT;
    }

    auto x86_engine = X86EngineFactory::Create(options);
    if (g_flag_disable_avx512) {
        x86_engine->Configure(ppl::nn::X86_CONF_DISABLE_AVX512);
    }
    if (g_flag_disable_avx_fma3) {
        x86_engine->Configure(ppl::nn::X86_CONF_DISABLE_AVX_FMA3);
    }
    if (g_flag_core_binding) {
        ppl::kernel::x86::set_omp_core_binding(nullptr, 0, 1);
    }
    // configure engine
    engines->emplace_back(unique_ptr<Engine>(x86_engine));
    LOG(INFO) << "***** register X86Engine *****";
    return true;
}

#endif

static inline bool RegisterEngines(vector<unique_ptr<Engine>>* engines) {
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

    if (engines->empty()) {
        LOG(ERROR) << "no engine is registered. run `./pplnn --help` to see supported engines marked with '--use-*', "
                      "or see documents listed in README.md for building instructions.";
        return false;
    }

    return true;
}

/* -------------------------------------------------------------------------- */

static string GetDimsStr(const Tensor* tensor) {
    auto& shape = tensor->GetShape();
    if (shape.GetRealDimCount() == 0) {
        return string();
    }

    string res = ToString(shape.GetDim(0));
    for (uint32_t i = 1; i < shape.GetDimCount(); ++i) {
        res += "_" + ToString(shape.GetDim(i));
    }

    return res;
}

static const char* MemMem(const char* haystack, unsigned int haystack_len,
                          const char* needle, unsigned int needle_len)
{
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

static void GenerateRandomDims(TensorShape* shape) {
    static const uint32_t max_dim = 640;
    static const uint32_t min_dim = 128;
    srand(time(nullptr));

    auto dimcount = shape->GetRealDimCount();
    for (uint32_t i = 2; i < dimcount; ++i) {
        if (shape->GetDim(i) == 1) {
            auto value = rand() % (max_dim - min_dim + 1) + min_dim;
            shape->SetDim(i, value);
        }
    }
}

static bool SetRandomInputs(const vector<vector<int64_t>>& input_shapes, Runtime* runtime) {
    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto& shape = t->GetShape();

        if (input_shapes.empty()) {
            GenerateRandomDims(&shape);
        } else {
            shape.Reshape(input_shapes[c]);
        }

        auto nr_element = shape.GetBytesIncludingPadding() / sizeof(float);
        vector<float> buffer(nr_element);

        std::default_random_engine eng;
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (uint32_t i = 0; i < nr_element; ++i) {
            buffer[i] = dis(eng);
        }

        auto status = t->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        TensorShape src_desc = t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        status = t->ConvertFromHost(buffer.data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set tensor[" << t->GetName() << "] content failed: " << GetRetCodeStr(status);
            return false;
        }
    }

    return true;
}

static bool SetInputsAllInOne(const string& input_file, const vector<vector<int64_t>>& input_shapes, Runtime* runtime) {
    FileMapping fm;
    auto status = fm.Init(input_file.c_str());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "mapping file[" << input_file << "] failed: " << GetRetCodeStr(status);
        return false;
    }

    auto data = fm.Data();
    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);

        if (!input_shapes.empty()) {
            t->GetShape().Reshape(input_shapes[c]);
        }

        auto status = t->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        TensorShape src_desc = t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        status = t->ConvertFromHost(data, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert tensor[" << t->GetName() << "] content failed: " << GetRetCodeStr(status);
            return false;
        }

        data += src_desc.GetBytesIncludingPadding();
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
                              Runtime* runtime) {
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
        const string& file_name = files[i];

        FileMapping fm;
        auto status = fm.Init(file_name.c_str());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "mapping file[" << file_name << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        auto t = runtime->GetInputTensor(i);

        if (!input_shapes.empty()) {
            t->GetShape().Reshape(input_shapes[i]);
        }

        status = t->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        TensorShape src_desc = t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        status = t->ConvertFromHost(fm.Data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set input[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }
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

static bool SetReshapedInputsOneByOne(const string& input_files_str, Runtime* runtime) {
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
        SplitString(dims_str.data(), dims_str.size(), "_", 1, [&dims](const char* s, unsigned int l) -> bool {
            if (l > 0) {
                int64_t dim = atol(string(s, l).c_str());
                dims.push_back(dim);
                return true;
            }
            LOG(ERROR) << "illegal dim format.";
            return false;
        });

        auto data_type = FindDataTypeByStr(data_type_str);
        if (data_type == DATATYPE_UNKNOWN) {
            LOG(ERROR) << "cannot find data type[" << data_type_str << "] in input file[" << file_name << "]";
            return false;
        }

        TensorShape input_shape;
        input_shape.SetDataFormat(DATAFORMAT_NDARRAY);
        input_shape.SetDataType(data_type);
        input_shape.Reshape(dims.data(), dims.size());

        FileMapping fm;
        auto status = fm.Init(file_full_path.c_str());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "mapping file[" << file_full_path << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        auto t = runtime->GetInputTensor(c);
        t->GetShape() = input_shape;

        status = t->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        TensorShape src_desc = t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        status = t->ConvertFromHost(fm.Data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set input[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }
    }

    return true;
}

static bool SaveInputsOneByOne(const Runtime* runtime) {
    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto& shape = t->GetShape();

        auto bytes = shape.GetBytesIncludingPadding();
        vector<char> buffer(bytes);

        TensorShape src_desc = t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto status = t->ConvertToHost(buffer.data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data failed: " << GetRetCodeStr(status);
            return false;
        }

        const char* data_type_str = FindDataTypeStr(shape.GetDataType());
        if (!data_type_str) {
            LOG(ERROR) << "unsupported data type[" << shape.GetDataType();
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
        auto bytes = t->GetShape().GetBytesIncludingPadding();
        vector<char> buffer(bytes);

        TensorShape src_desc = t->GetShape();
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

        TensorShape dst_desc = t->GetShape();
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto bytes = dst_desc.GetBytesIncludingPadding();
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

static void PrintInputOutputInfo(const Runtime* runtime) {
    LOG(INFO) << "----- input info -----";
    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto tensor = runtime->GetInputTensor(i);
        LOG(INFO) << "input[" << i << "]:";
        LOG(INFO) << "    name: " << tensor->GetName();

        string dims_str;
        auto& shape = tensor->GetShape();
        for (uint32_t j = 0; j < shape.GetDimCount(); ++j) {
            dims_str += " " + ToString(shape.GetDim(j));
        }
        LOG(INFO) << "    dim(s):" << dims_str;

        LOG(INFO) << "    DataType: " << GetDataTypeStr(shape.GetDataType());
        LOG(INFO) << "    DataFormat: " << GetDataFormatStr(shape.GetDataFormat());
        LOG(INFO) << "    NumBytesIncludePadding: " << shape.GetBytesIncludingPadding();
        LOG(INFO) << "    NumBytesExcludePadding: " << shape.GetBytesExcludingPadding();
    }

    LOG(INFO) << "----- output info -----";
    for (uint32_t i = 0; i < runtime->GetOutputCount(); ++i) {
        auto tensor = runtime->GetOutputTensor(i);
        LOG(INFO) << "output[" << i << "]:";
        LOG(INFO) << "    name: " << tensor->GetName();

        string dims_str;
        auto& shape = tensor->GetShape();
        for (uint32_t j = 0; j < shape.GetDimCount(); ++j) {
            dims_str += " " + ToString(shape.GetDim(j));
        }
        LOG(INFO) << "    dim(s):" << dims_str;

        LOG(INFO) << "    DataType: " << GetDataTypeStr(shape.GetDataType());
        LOG(INFO) << "    DataFormat: " << GetDataFormatStr(shape.GetDataFormat());
        LOG(INFO) << "    NumBytesIncludePadding: " << shape.GetBytesIncludingPadding();
        LOG(INFO) << "    NumBytesExcludePadding: " << shape.GetBytesExcludingPadding();
    }

    LOG(INFO) << "----------------------";
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

static bool ParseInputShapes(const string& shape_str, vector<vector<int64_t>>* input_shapes) {
    bool ok = true;

    vector<string> input_shape_list;
    SplitString(shape_str.data(), shape_str.size(), ",", 1,
                [&ok, &input_shape_list](const char* s, unsigned int l) -> bool {
                    if (l > 0) {
                        input_shape_list.emplace_back(s, l);
                        return true;
                    }
                    LOG(ERROR) << "empty shape in option '--input-shapes'";
                    ok = false;
                    return false;
                });
    if (!ok) {
        return false;
    }

    for (auto x = input_shape_list.begin(); x != input_shape_list.end(); ++x) {
        ok = true;
        vector<int64_t> shape;
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

        input_shapes->push_back(shape);
    }

    return true;
}

int main(int argc, char* argv[]) {
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
    if (g_flag_version) {
        cout << GetVersionString() << endl;
        return 0;
    }

    LOG(INFO) << "ppl.nn version: " << GetVersionString();

    vector<unique_ptr<Engine>> engines;
    if (!RegisterEngines(&engines)) {
        LOG(ERROR) << "RegisterEngines failed.";
        return -1;
    }

    unique_ptr<Runtime> runtime;

    if (!g_flag_onnx_model.empty()) {
        vector<Engine*> engine_ptrs(engines.size());
        for (uint32_t i = 0; i < engines.size(); ++i) {
            engine_ptrs[i] = engines[i].get();
        }
        auto builder = unique_ptr<RuntimeBuilder>(
            OnnxRuntimeBuilderFactory::Create(g_flag_onnx_model.c_str(), engine_ptrs.data(), engine_ptrs.size()));
        if (!builder) {
            LOG(ERROR) << "create RuntimeBuilder failed.";
            return -1;
        }

        runtime.reset(builder->CreateRuntime());
    }

    if (!runtime) {
        LOG(ERROR) << "CreateRuntime failed.";
        return -1;
    }

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

    if (!g_flag_input.empty()) {
        if (!SetInputsAllInOne(g_flag_input, input_shapes, runtime.get())) {
            LOG(ERROR) << "SetInputsAllInOne failed.";
            return -1;
        }
    } else if (!g_flag_inputs.empty()) {
        if (!SetInputsOneByOne(g_flag_inputs, input_shapes, runtime.get())) {
            LOG(ERROR) << "SetInputsOneByOne failed.";
            return -1;
        }
    } else if (!g_flag_reshaped_inputs.empty()) {
        if (!SetReshapedInputsOneByOne(g_flag_reshaped_inputs, runtime.get())) {
            LOG(ERROR) << "SetReshapedInputsOneByOne failed.";
            return -1;
        }
    } else {
        if (!SetRandomInputs(input_shapes, runtime.get())) {
            LOG(ERROR) << "SetRandomInputs failed.";
            return -1;
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

    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto in = runtime->GetInputTensor(i);
        auto& shape = in->GetShape();
        if (shape.GetElementsIncludingPadding() == 0) {
            LOG(ERROR) << "input tensor[" << in->GetName() << "] is empty.";
            return -1;
        }
    }

    auto run_begin_ts = std::chrono::system_clock::now();
    auto status = runtime->Run();
    if (status == RC_SUCCESS) {
        status = runtime->Sync();
    }
    auto run_end_ts = std::chrono::system_clock::now();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Run() failed: " << GetRetCodeStr(status);
        return -1;
    }

    PrintInputOutputInfo(runtime.get());

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    LOG(INFO) << "Run() costs: " << (float)diff.count() / 1000 << " ms.";

    if (g_flag_save_outputs) {
        if (!SaveOutputsOneByOne(runtime.get())) {
            return -1;
        }
    }

    LOG(INFO) << "Run ok";

    if (g_flag_enable_profiling) {
        if (g_flag_warmup_times > 0) {
            LOG(INFO) << "Warm up start for " << g_flag_warmup_times << " times.";
            for (uint32_t i = 0; i < g_flag_warmup_times; ++i) {
                auto status = runtime->Run();
                if (status == RC_SUCCESS) {
                    status = runtime->Sync();
                }
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
        int32_t run_count = 0;
        while (run_dur < g_flag_min_profiling_time * 1000) {
            run_begin_ts = std::chrono::system_clock::now();
            auto status = runtime->Run();
            if (status == RC_SUCCESS) {
                status = runtime->Sync();
            }
            run_end_ts = std::chrono::system_clock::now();
            diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
            run_dur += (double)diff.count() / 1000;
            run_count += 1;
        }

        LOG(INFO) << "Duration: " << run_dur << " ms";

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        ProfilingStatistics stat;
        status = runtime->GetProfilingStatistics(&stat);
        if (status != RC_SUCCESS) {
            LOG(WARNING) << "Get profiling statistics failed: " << GetRetCodeStr(status);
        }
        PrintProfilingStatistics(stat, run_dur, run_count);
#else
        LOG(INFO) << "Average run cost: " << (run_dur / run_count) << " ms.";
#endif

        LOG(INFO) << "Profiling End";
    }

    return 0;
}
