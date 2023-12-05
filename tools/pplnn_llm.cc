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
#include "ppl/common/destructor.h"
using namespace ppl::common;

#include "ppl/nn/runtime/options.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/utils/file_data_stream.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::nn;

#ifdef PPLNN_ENABLE_ONNX_MODEL
#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#endif

/* -------------------------------------------------------------------------- */

#include "simple_flags.h"

Define_bool_opt("--help", g_flag_help, false, "show these help information");
Define_bool_opt("--version", g_flag_version, false, "show version info");

#ifdef PPLNN_ENABLE_ONNX_MODEL
Define_string_opt("--onnx-model", g_flag_onnx_model, "", "onnx model file");
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

Define_string_opt("--input-files", g_flag_input_files, "", "binary input files separated by comma");
Define_string_opt("--shaped-input-files", g_flag_shaped_input_files, "",
                  "binary input files separated by comma."
                  " file name format: 'name-dims-datatype.dat'. for example:"
                  " input1-1_1_1_1-fp32.dat,input2-1_1_1_1-fp16.dat,input3-1_1-int8.dat");
Define_string_opt("--inupt-shapes", g_flag_input_shapes, "",
                  "shapes of input tensors."
                  " dims are separated by underline, inputs are separated by comma. example:"
                  " 1_3_128_128,2_3_400_640,3_3_768_1024. empty fields between commas are scalars.");

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

static int mpi_world_size = 1;
static int mpi_local_rank = 0;
static int mpi_initialized = 0;

#ifdef PPLNN_ENABLE_MPI_TOOLS
#include <mpi.h>

#define MPI_CHECK(cmd, emsg) \
do {\
    int e = (cmd);\
    if (e != MPI_SUCCESS) {\
        LOG(ERROR) << "MPI error(code:" << e << ") on " << (emsg);\
        return RC_OTHER_ERROR;\
    }\
} while (0);

static RetCode InitMPI(int *argc, char ***argv) {
    MPI_CHECK(MPI_Init(argc, argv), "MPI_Init");

    MPI_CHECK(MPI_Initialized(&mpi_initialized), "MPI_Initialized");

    if (!mpi_initialized) {
        LOG(ERROR) << "MPI is not initialized.";
        return RC_OTHER_ERROR;
    }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_local_rank), "MPI_Comm_rank");
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size), "MPI_Comm_size");

    if (mpi_local_rank != 0)
        ppl::common::GetCurrentLogger()->SetLogLevel(ppl::common::LOG_LEVEL_WARNING);
    LOG(INFO) << "MPI world size: " << mpi_world_size;

    return RC_SUCCESS;
}

static void FinalizeMPI() {
    MPI_Finalize();
}

#endif

/* -------------------------------------------------------------------------- */

#ifdef PPLNN_USE_LLM_CUDA

Define_bool_opt("--use-llm-cuda", g_flag_use_llm_cuda, false, "use llm cuda engine");

Define_uint32_opt("--device-id", g_flag_device_id, 0, "declare device id for llm cuda");

Define_string_opt("--in-devices", g_flag_in_devices, "",
                        "specify device of each input separated by comma, "
                        "only accept \"host\" and \"device\", "
                        "all tensor is set to \"device\" by default");

Define_string_opt("--quant-method", g_flag_quant_method, "none",
                        "llm cuda quantization mehtod, only accept "
                        "\"none\", \"online_i8i8\" and \"online_i4f16\", "
                        "default: \"none\"");

Define_string_opt("--cublas-layout-hint", g_cublas_layout_hint, "default",
                        "matrix layout hint for cublas(currently only effect int8 gemm), only accept "
                        "\"default\", \"ampere\". "
                        "default: \"default\"");

#include <cuda_runtime.h>

#ifdef PPLNN_CUDA_ENABLE_NCCL
#include <nccl.h>

#define NCCL_CHECK(cmd, emsg) \
do {\
    ncclResult_t e = (cmd);\
    if (e != ncclSuccess) {\
        LOG(ERROR) << "NCCL error(code:" << (int)e << ") on " << (emsg);\
        return RC_OTHER_ERROR;\
    }\
} while (0);

static ncclComm_t g_nccl_comm = nullptr;

static RetCode InitNccl() {
    if (mpi_world_size == 1) {
        return RC_SUCCESS;
    }
    if (!mpi_initialized) {
        LOG(ERROR) << "MPI is not initialized for NCCL initialization.";
        return RC_OTHER_ERROR;
    }

#ifdef PPLNN_ENABLE_MPI_TOOLS
    std::vector<uint32_t> devices(mpi_world_size, 0);
    MPI_CHECK(MPI_Allgather(
        &g_flag_device_id, 1, MPI_UINT32_T,
        devices.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD), "MPI_Gather");
        std::sort(devices.begin(), devices.end());
    if (std::unique(devices.begin(), devices.end()) != devices.end()) {
        if (mpi_local_rank == 0) {
            LOG(WARNING) << "CUDA device-id are not uniqueded, please check '--device-id' option.";
            LOG(WARNING) << "Fallback CUDA device-id to MPI rank.";
        }
        g_flag_device_id = mpi_local_rank;
    }
#endif
    cudaSetDevice(g_flag_device_id);

    ncclUniqueId uid;
    NCCL_CHECK(ncclGetUniqueId(&uid), "ncclGetUniqueId");

#ifdef PPLNN_ENABLE_MPI_TOOLS
    MPI_CHECK(MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD), "MPI_Bcast");
#endif
    NCCL_CHECK(ncclCommInitRank(&g_nccl_comm, mpi_world_size, uid, mpi_local_rank), "ncclCommInitRank");

    return RC_SUCCESS;
}

static void FinalizeNccl() {
    if (g_nccl_comm != nullptr)
        ncclCommDestroy(g_nccl_comm);
}

#endif

#include "ppl/nn/engines/llm_cuda/engine_factory.h"
#include "ppl/nn/engines/llm_cuda/options.h"
#include "ppl/nn/utils/array.h"

static bool RegisterLlmCudaEngine(vector<unique_ptr<Engine>>* engines) {
    llm::cuda::EngineOptions options;
    options.device_id = g_flag_device_id;

    if (g_flag_mm_policy == "mem") {
        options.mm_policy = llm::cuda::MM_COMPACT;
    } else if (g_flag_mm_policy == "plain") {
        options.mm_policy = llm::cuda::MM_PLAIN;
    } else {
        LOG(ERROR) << "unknown/unsupported --mm-policy option: " << g_flag_mm_policy;
        return false;
    }

    if (g_flag_quant_method == "none") {
        options.quant_method = llm::cuda::QUANT_METHOD_NONE;
    } else if (g_flag_quant_method == "online_i8i8") {
        options.quant_method = llm::cuda::QUANT_METHOD_ONLINE_I8I8;
    } else if (g_flag_quant_method == "online_i4f16") {
        options.quant_method = llm::cuda::QUANT_METHOD_ONLINE_I4F16;
    } else  {
        LOG(ERROR) << "unknown/unsupported --quant-method option: " << g_flag_quant_method;
        return false;
    }

    if (g_cublas_layout_hint == "default") {
        options.cublas_layout_hint = llm::cuda::CUBLAS_LAYOUT_DEFAULT;
    } else if (g_cublas_layout_hint == "ampere") {
        options.cublas_layout_hint = llm::cuda::CUBLAS_LAYOUT_AMPERE;
    } else {
        LOG(ERROR) << "unknown/unsupported --cublas-layout-hint option: " << g_cublas_layout_hint;
        return false;
    }

    auto llm_cuda_engine = llm::cuda::EngineFactory::Create(options);
    if (!llm_cuda_engine) {
        LOG(ERROR) << "create LlmCudaEngine failed.";
        return false;
    }

#ifdef PPLNN_CUDA_ENABLE_NCCL
    if (g_nccl_comm != nullptr) {
        auto status = llm_cuda_engine->Configure(llm::cuda::ENGINE_CONF_SET_TP_NCCL_COMM, g_nccl_comm);
        if (RC_SUCCESS != status) {
            LOG(ERROR) << "configure SET_TP_NCCL_COMM failed: " << ppl::common::GetRetCodeStr(status);
            return false;
        }
    }
#endif

    engines->emplace_back(unique_ptr<Engine>(llm_cuda_engine));
    LOG(INFO) << "***** register LlmCudaEngine *****";
    return true;
}

static bool SetInputDeviceOneByOne(const string& input_device_str, Runtime* runtime) {
    vector<string> devices;
    SplitString(input_device_str.data(), input_device_str.size(), ",", 1,
                [&devices](const char* s, unsigned int l) -> bool {
                    if (l > 0) {
                        devices.push_back(string(s, l));
                    }
                    return true;
                });
    if (devices.size() != runtime->GetInputCount()) {
        LOG(ERROR) << "input device num[" << devices.size() << "] != input count[" << runtime->GetInputCount() << "]";
        return false;
    }

    for (uint32_t i = 0; i < devices.size(); ++i) {
        auto t = runtime->GetInputTensor(i);
        if (devices[i] == "host") {
            LOG(INFO) << "Set [" << t->GetName() << "] to Host";
            t->SetDeviceContext(runtime->GetHostDeviceContext());
        } else if (devices[i] == "device") {
            LOG(INFO) << "Keep [" << t->GetName() << "] on Device";
        } else {
            LOG(ERROR) << "Get unknown device name [" << devices[i] << "]";
            return false;
        }
    }

    return true;
}

#endif


static bool RegisterEngines(vector<unique_ptr<Engine>>* engines) {
#ifdef PPLNN_USE_LLM_CUDA
    if (g_flag_use_llm_cuda) {
        bool ok = RegisterLlmCudaEngine(engines);
        if (!ok) {
            LOG(ERROR) << "RegisterLlmCudaEngine failed.";
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

static const pair<string, datatype_t> g_str2datatype[] = {
    {"fp64", DATATYPE_FLOAT64}, {"fp32", DATATYPE_FLOAT32}, {"fp16", DATATYPE_FLOAT16}, {"int32", DATATYPE_INT32},
    {"int64", DATATYPE_INT64},  {"int8", DATATYPE_INT8},    {"bool", DATATYPE_BOOL},    {"", DATATYPE_UNKNOWN},
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
        auto tensor_size = src_desc.CalcBytesExcludingPadding();
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

        LOG(INFO) << "Read [" << file_full_path.c_str() << "] to Input(" << i << ")[" << t->GetName() << "]";

        input_data->emplace_back(string(fm.GetData(), fm.GetSize()));
    }

    return true;
}

static bool SetShapedInputsOneByOne(const string& input_files_str, Runtime* runtime, vector<string>* input_data) {
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

        auto t = runtime->GetInputTensor(c);
        *t->GetShape() = input_shape;

        if (input_shape.IsEmpty()) {
            input_data->emplace_back(string());
            continue;
        }

        Mmap fm;
        auto status = fm.Init(file_full_path.c_str(), Mmap::READ);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "mapping file[" << file_full_path << "] failed.";
            return false;
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

        LOG(INFO) << "Read [" << file_full_path.c_str() << "] to Input(" << c << ")[" << t->GetName() << "]";

        input_data->emplace_back(string(fm.GetData(), fm.GetSize()));
    }

    return true;
}

static bool SaveInputsOneByOne(const Runtime* runtime, const std::string &tag = "") {
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
        if (tag.empty())
            sprintf(name_prefix, "pplnn_input_%05u_", c);
        else
            sprintf(name_prefix, "pplnn_input_%s_%05u_", tag.c_str(), c);
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

static bool SaveOutputsOneByOne(const Runtime* runtime) {
    for (uint32_t c = 0; c < runtime->GetOutputCount(); ++c) {
        auto t = runtime->GetOutputTensor(c);

        ppl::nn::TensorShape dst_desc = *t->GetShape();
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);

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
        if (tensor->GetShape()->GetDataType() == ppl::common::DATATYPE_INT64 && elem_count <= 10) {
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

static void PrintOutputInfo(const Runtime* runtime) {
    LOG(INFO) << "----- output info -----";
    for (uint32_t i = 0; i < runtime->GetOutputCount(); ++i) {
        auto tensor = runtime->GetOutputTensor(i);
        LOG(INFO) << "output[" << i << "]:";
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

        datatype_t saved_data_type = shape->GetDataType();
        if (saved_data_type == DATATYPE_FLOAT16) {
            saved_data_type = DATATYPE_FLOAT32;
        }
        LOG(INFO) << "    saved data type: " << GetDataTypeStr(saved_data_type);
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
        auto run_begin_ts = std::chrono::high_resolution_clock::now();
        runtime->Run();
        auto run_end_ts = std::chrono::high_resolution_clock::now();
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

    LOG(INFO) << "ppl.nn version: [" << PPLNN_VERSION_MAJOR << "." << PPLNN_VERSION_MINOR << "." << PPLNN_VERSION_PATCH
         << "], commit: [" << PPLNN_COMMIT_STR << "]";

    if (g_flag_version) {
        return 0;
    }

#ifdef PPLNN_ENABLE_MPI_TOOLS
    status = InitMPI(&argc, &argv);
    if (RC_SUCCESS != status) {
        return -1;
    }
    ppl::common::Destructor __mpi_guard([]() -> void {
        FinalizeMPI();
    });
#endif

#if defined(PPLNN_USE_LLM_CUDA) && defined(PPLNN_CUDA_ENABLE_NCCL)
    if (g_flag_use_llm_cuda) {
        status = InitNccl();
        if (RC_SUCCESS != status) {
            return -1;
        }
    }
    ppl::common::Destructor __nccl_guard([]() -> void {
        FinalizeNccl();
    });
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

    auto prepare_begin_ts = std::chrono::high_resolution_clock::now();
    vector<unique_ptr<Engine>> engines;
    if (!RegisterEngines(&engines)) {
        LOG(ERROR) << "RegisterEngines failed.";
        return -1;
    }
    auto prepare_end_ts = std::chrono::high_resolution_clock::now();
    auto prepare_diff = std::chrono::duration_cast<std::chrono::microseconds>(prepare_end_ts - prepare_begin_ts);
    LOG(INFO) << "RegisterEngines costs: " << (float)prepare_diff.count() / 1000 << " ms.";

#ifdef PPLNN_USE_LLM_CUDA
    {
        size_t free_bytes, total_bytes;
        auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMemGetInfo failed: " << cudaGetErrorString(err);
        } else {
            double free  = free_bytes / 1024.0 / 1024.0;
            double total = total_bytes / 1024.0 / 1024.0;
            double used  = total - free;
            LOG(INFO) << "Init Engine GPU Mem Usage: " << used / 1024.0 << " GiB";
        }
    }
#endif

    unique_ptr<Runtime> runtime;

    if (false) {
    }
#ifdef PPLNN_ENABLE_ONNX_MODEL
    else if (!g_flag_onnx_model.empty()) {
        prepare_begin_ts = std::chrono::high_resolution_clock::now();
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
        prepare_end_ts = std::chrono::high_resolution_clock::now();
        prepare_diff = std::chrono::duration_cast<std::chrono::microseconds>(prepare_end_ts - prepare_begin_ts);
        LOG(INFO) << "RuntimeBuilder LoadModel costs: " << (float)prepare_diff.count() / 1000 << " ms.";

        prepare_begin_ts = std::chrono::high_resolution_clock::now();
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
        prepare_end_ts = std::chrono::high_resolution_clock::now();
        prepare_diff = std::chrono::duration_cast<std::chrono::microseconds>(prepare_end_ts - prepare_begin_ts);
        LOG(INFO) << "RuntimeBuilder Preprocess costs: " << (float)prepare_diff.count() / 1000 << " ms.";

        prepare_begin_ts = std::chrono::high_resolution_clock::now();
        runtime.reset(builder->CreateRuntime());
        prepare_end_ts = std::chrono::high_resolution_clock::now();
        prepare_diff = std::chrono::duration_cast<std::chrono::microseconds>(prepare_end_ts - prepare_begin_ts);
        LOG(INFO) << "RuntimeBuilder CreateRuntime costs: " << (float)prepare_diff.count() / 1000 << " ms.";
    }
#endif

#ifdef PPLNN_USE_LLM_CUDA
    {
        size_t free_bytes, total_bytes;
        auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMemGetInfo failed: " << cudaGetErrorString(err);
        } else {
            double free  = free_bytes / 1024.0 / 1024.0;
            double total = total_bytes / 1024.0 / 1024.0;
            double used  = total - free;
            LOG(INFO) << "Load Model GPU Mem Usage: " << used / 1024.0 << " GiB";
        }
    }
#endif

    if (!runtime) {
        LOG(ERROR) << "CreateRuntime failed.";
        return -1;
    }

#ifdef PPLNN_USE_LLM_CUDA
    if (!g_flag_in_devices.empty()) {
        if(!SetInputDeviceOneByOne(g_flag_in_devices, runtime.get())) {
            LOG(ERROR) << "SetInputDeviceOneByOne failed.";
            return -1;
        }
    }
#endif


    prepare_begin_ts = std::chrono::high_resolution_clock::now();
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
    if (!g_flag_input_files.empty()) {
        if (!SetInputsOneByOne(g_flag_input_files, input_shapes, runtime.get(), &input_data)) {
            LOG(ERROR) << "SetInputsOneByOne failed.";
            return -1;
        }
    } else if (!g_flag_shaped_input_files.empty()) {
        if (!SetShapedInputsOneByOne(g_flag_shaped_input_files, runtime.get(), &input_data)) {
            LOG(ERROR) << "SetShapedInputsOneByOne failed.";
            return -1;
        }
    } else {
        if (!SetRandomInputs(input_shapes, runtime.get(), &input_data)) {
            LOG(ERROR) << "SetRandomInputs failed.";
            return -1;
        }
    }
    input_data.clear();

    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto in = runtime->GetInputTensor(i);
        auto shape = in->GetShape();
        if (shape->CalcElementsIncludingPadding() == 0) {
            // We should allow empty input
            LOG(WARNING) << "input tensor[" << in->GetName() << "] is empty.";
        }
    }

    if (g_flag_save_inputs) {
        if (!SaveInputsOneByOne(runtime.get())) {
            return -1;
        }
    }

    PrintInputInfo(runtime.get());

    prepare_end_ts = std::chrono::high_resolution_clock::now();
    prepare_diff = std::chrono::duration_cast<std::chrono::microseconds>(prepare_end_ts - prepare_begin_ts);
    LOG(INFO) << "Load/Save Input costs: " << (float)prepare_diff.count() / 1000 << " ms.";

    if (g_flag_no_run) {
        return 0;
    }

    auto run_begin_ts = std::chrono::high_resolution_clock::now();
    status = runtime->Run();
    auto run_end_ts = std::chrono::high_resolution_clock::now();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Run() failed: " << GetRetCodeStr(status);
        return -1;
    }

    PrintOutputInfo(runtime.get());

    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    LOG(INFO) << "Run() costs: " << (float)diff.count() / 1000 << " ms.";

    if (g_flag_save_inputs) {
        if (!SaveInputsOneByOne(runtime.get(), "afterrun")) {
            return -1;
        }
    }

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

#ifdef PPLNN_USE_LLM_CUDA
    {
        size_t free_bytes, total_bytes;
        auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMemGetInfo failed: " << cudaGetErrorString(err);
        } else {
            double free  = free_bytes / 1024.0 / 1024.0;
            double total = total_bytes / 1024.0 / 1024.0;
            double used  = total - free;
            LOG(INFO) << "Run Model GPU Mem Usage: " << used / 1024.0 << " GiB";
        }
    }
#endif

    return 0;
}
