# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import logging
import argparse
import random
import numpy as np
from pyppl import nn as pplnn
from pyppl import common as pplcommon

# ---------------------------------------------------------------------------- #

g_supported_devices = ["x86", "cuda"]

g_pplnntype2numpytype = {
    pplcommon.DATATYPE_INT8 : np.int8,
    pplcommon.DATATYPE_INT16 : np.int16,
    pplcommon.DATATYPE_INT32 : np.int32,
    pplcommon.DATATYPE_INT64 : np.int64,
    pplcommon.DATATYPE_UINT8 : np.uint8,
    pplcommon.DATATYPE_UINT16 : np.uint16,
    pplcommon.DATATYPE_UINT32 : np.uint32,
    pplcommon.DATATYPE_UINT64 : np.uint64,
    pplcommon.DATATYPE_FLOAT16 : np.float16,
    pplcommon.DATATYPE_FLOAT32 : np.float32,
    pplcommon.DATATYPE_FLOAT64 : np.float64,
    pplcommon.DATATYPE_BOOL : bool,
}

g_data_type_str = {
    pplcommon.DATATYPE_INT8 : "int8",
    pplcommon.DATATYPE_INT16 : "int16",
    pplcommon.DATATYPE_INT32 : "int32",
    pplcommon.DATATYPE_INT64 : "int64",
    pplcommon.DATATYPE_UINT8 : "uint8",
    pplcommon.DATATYPE_UINT16 : "uint16",
    pplcommon.DATATYPE_UINT32 : "uint32",
    pplcommon.DATATYPE_UINT64 : "uint64",
    pplcommon.DATATYPE_FLOAT16 : "fp16",
    pplcommon.DATATYPE_FLOAT32 : "fp32",
    pplcommon.DATATYPE_FLOAT64 : "fp64",
    pplcommon.DATATYPE_BOOL : "bool",
    pplcommon.DATATYPE_UNKNOWN : "unknown",
}

# ---------------------------------------------------------------------------- #

def ParseCommandLineArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", dest = "display_version", action = "store_true",
                        default = False, required = False)

    for dev in g_supported_devices:
        parser.add_argument("--use-" + dev, dest = "use_" + dev, action = "store_true",
                            default = False, required = False)
        if dev == "x86":
            parser.add_argument("--disable-avx512", dest = "disable_avx512", action = "store_true",
                                default = False, required = False)
            parser.add_argument("--disable-avx-fma3", dest = "disable_avx_fma3", action = "store_true",
                                default = False, required = False)
        elif dev == "cuda":
            parser.add_argument("--quick-select", dest = "quick_select", action = "store_true",
                                default = False, required = False)
            parser.add_argument("--device-id", type = int, dest = "device_id",
                                default = 0, required = False, help = "specify which device is used.")
            parser.add_argument("--import-algo-file", type = str, default = "", required = False,
                                help = "a json file containing op implementations info")
            parser.add_argument("--export-algo-file", type = str, default = "", required = False,
                                help = "a json file used to store op implementations info")

    parser.add_argument("--onnx-model", type = str, default = "", required = False,
                        help = "onnx model file")

    parser.add_argument("--mm-policy", type = str, default = "perf", required = False,
                        help = "\"perf\" => better performance, or \"mem\" => less memory usage")

    parser.add_argument("--in-shapes", type = str, dest = "in_shapes",
                        default = "", required = False, help = "shapes of input tensors."
                        " dims are separated by underline, inputs are separated by comma. example:"
                        " 1_3_128_128,2_3_400_640,3_3_768_1024")
    parser.add_argument("--inputs", type = str, dest = "inputs",
                        default = "", required = False, help = "input files separated by comma.")
    parser.add_argument("--reshaped-inputs", type = str, dest = "reshaped_inputs",
                        default = "", required = False, help = "binary input files separated by comma."
                        " file name format: 'name-dims-datatype.dat'. for example:"
                        " input1-1_1_1_1-fp32.dat,input2-1_1_1_1-fp16.dat,input3-1_1-int8.dat")

    parser.add_argument("--save-input", dest = "save_input", action = "store_true",
                        default = False, required = False,
                        help = "save all input tensors in NDARRAY format in one file named 'pplnn_inputs.dat'")
    parser.add_argument("--save-inputs", dest = "save_inputs", action = "store_true",
                        default = False, required = False,
                        help = "save separated input tensors in NDARRAY format")
    parser.add_argument("--save-outputs", dest = "save_outputs", action = "store_true",
                        default = False, required = False,
                        help = "save separated output tensors in NDARRAY format")
    parser.add_argument("--save-data-dir", type = str, dest = "save_data_dir",
                        default = ".", required = False,
                        help = "directory to save input/output data if '--save-*' options are enabled.")

    return parser.parse_args()

# ---------------------------------------------------------------------------- #

def ParseInShapes(in_shapes_str):
    ret = []
    shape_strs = list(filter(None, in_shapes_str.split(",")))
    for s in shape_strs:
        dims = [int(d) for d in s.split("_")]
        ret.append(dims)
    return ret

# ---------------------------------------------------------------------------- #

def CreateX86Engine(args):
    x86_options = pplnn.X86EngineOptions()
    if args.mm_policy == "perf":
        x86_options.mm_policy = pplnn.X86_MM_MRU
    elif args.mm_policy == "mem":
        x86_options.mm_policy = pplnn.X86_MM_COMPACT

    x86_engine = pplnn.X86EngineFactory.Create(x86_options)
    if not x86_engine:
        logging.error("create x86 engine failed.")
        sys.exit(-1)

    if args.disable_avx512:
        status = x86_engine.Configure(pplnn.X86_CONF_DISABLE_AVX512)
        if status != pplcommon.RC_SUCCESS:
            logging.error("x86 engine Configure() failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.disable_avx_fma3:
        status = x86_engine.Configure(pplnn.X86_CONF_DISABLE_AVX_FMA3)
        if status != pplcommon.RC_SUCCESS:
            logging.error("x86 engine Configure() failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    return x86_engine

def CreateCudaEngine(args):
    cuda_options = pplnn.CudaEngineOptions()
    cuda_options.device_id = args.device_id
    if args.mm_policy == "perf":
        cuda_options.mm_policy = pplnn.CUDA_MM_BEST_FIT
    elif args.mm_policy == "mem":
        cuda_options.mm_policy = pplnn.CUDA_MM_COMPACT

    cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
    if not cuda_engine:
        logging.error("create cuda engine failed.")
        sys.exit(-1)

    if args.quick_select:
        status = cuda_engine.Configure(pplnn.CUDA_CONF_USE_DEFAULT_ALGORITHMS)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(CUDA_CONF_USE_DEFAULT_ALGORITHMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.in_shapes:
        shapes = ParseInShapes(args.in_shapes)
        status = cuda_engine.Configure(pplnn.CUDA_CONF_SET_INPUT_DIMS, shapes)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(CUDA_CONF_SET_INPUT_DIMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.export_algo_file:
        status = cuda_engine.Configure(pplnn.CUDA_CONF_EXPORT_ALGORITHMS, args.export_algo_file)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(CUDA_CONF_EXPORT_ALGORITHMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.import_algo_file:
        # import and export from the same file
        if args.import_algo_file == args.export_algo_file:
            # try to create this file first
            f = open(args.export_algo_file, "a")
            f.close()

        status = cuda_engine.Configure(pplnn.CUDA_CONF_IMPORT_ALGORITHMS, args.import_algo_file)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(CUDA_CONF_IMPORT_ALGORITHMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    return cuda_engine

def RegisterEngines(args):
    engines = []
    if args.use_x86:
        x86_engine = CreateX86Engine(args)
        engines.append(pplnn.Engine(x86_engine))

    if args.use_cuda:
        cuda_engine = CreateCudaEngine(args)
        engines.append(pplnn.Engine(cuda_engine))

    return engines

# ---------------------------------------------------------------------------- #

def SetInputsOneByOne(inputs, in_shapes, runtime):
    input_files = list(filter(None, inputs.split(",")))
    file_num = len(input_files)
    if file_num != runtime.GetInputCount():
        logging.error("input file num[" + str(file_num) + "] != graph input num[" +
                      runtime.GetInputCount() + "]")
        sys.exit(-1)

    for i in range(file_num):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        np_data_type = g_pplnntype2numpytype[shape.GetDataType()]

        dims = []
        if in_shapes:
            dims = in_shapes[i]
        else:
            dims = shape.GetDims()

        in_data = np.fromfile(input_files[i], dtype=np_data_type).reshape(dims)
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            logging.error("copy data to tensor[" + tensor.GetName() + "] failed: " +
                          pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

# ---------------------------------------------------------------------------- #

def SetReshapedInputsOneByOne(reshaped_inputs, runtime):
    input_files = list(filter(None, reshaped_inputs.split(",")))
    file_num = len(input_files)
    if file_num != runtime.GetInputCount():
        logging.error("input file num[" + str(file_num) + "] != graph input num[" +
                      runtime.GetInputCount() + "]")
        sys.exit(-1)

    for i in range(file_num):
        input_file_name = os.path.basename(input_files[i])
        file_name_components = input_file_name.split("-")
        if len(file_name_components) != 3:
            logging.error("invalid input filename[" + input_files[i] + "] in '--reshaped-inputs'.")
            sys.exit(-1)

        input_shape_str_list = file_name_components[1].split("_")
        input_shape = [int(s) for s in input_shape_str_list]

        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        np_data_type = g_pplnntype2numpytype[shape.GetDataType()]
        in_data = np.fromfile(input_files[i], dtype=np_data_type).reshape(input_shape)
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            logging.error("copy data to tensor[" + tensor.GetName() + "] failed: " +
                          pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

# ---------------------------------------------------------------------------- #

def SetRandomInputs(in_shapes, runtime):
    def GenerateRandomDims(shape):
        dims = shape.GetDims()
        dim_count = len(dims)
        for i in range(2, dim_count):
            if dims[i] == 1:
                dims[i] = random.randint(128, 641)
                if dims[i] % 2 != 0:
                    dims[i] = dims[i] + 1
        return dims

    rng = np.random.default_rng()
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        data_type = shape.GetDataType()

        np_data_type = g_pplnntype2numpytype[data_type]
        if data_type in (pplcommon.DATATYPE_FLOAT16, pplcommon.DATATYPE_FLOAT32, pplcommon.DATATYPE_FLOAT64):
            lower_bound = -1.0
            upper_bound = 1.0
        else:
            info = np.iinfo(np_data_type)
            lower_bound = info.min
            upper_bound = info.max

        dims = []
        if in_shapes:
            dims = in_shapes[i]
        else:
            dims = GenerateRandomDims(shape)

        in_data = (upper_bound - lower_bound) * rng.random(dims, dtype = np_data_type) * lower_bound
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            logging.error("copy data to tensor[" + tensor.GetName() + "] failed: " +
                          pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

# ---------------------------------------------------------------------------- #

def GenDimsStr(dims):
    if not dims:
        return ""

    s = str(dims[0])
    for i in range(1, len(dims)):
        s = s + "_" + str(dims[i])
    return s

# ---------------------------------------------------------------------------- #

def CalcElementCount(dims):
    count = 1
    for d in dims:
        count = count * d
    return count

def SaveInputsOneByOne(save_data_dir, runtime):
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        dims = shape.GetDims()
        out_file_name = save_data_dir + "/pplnn_input_" + str(i) + "_" + tensor.GetName() + "-" + GenDimsStr(dims) + "-" + g_data_type_str[shape.GetDataType()] + ".dat"
        element_count = CalcElementCount(dims)
        if element_count > 0:
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                logging.error("copy data from tensor[" + tensor.GetName() + "] failed.")
                sys.exit(-1)

            in_data = np.array(tensor_data, copy=False)
            in_data.tofile(out_file_name)
        else:
            open(out_file_name, 'a').close()

# ---------------------------------------------------------------------------- #

def SaveInputsAllInOne(save_data_dir, runtime):
    out_file_name = save_data_dir + "/pplnn_inputs.dat"
    fd = open(out_file_name, mode="wb+")
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        dims = tensor.GetShape().GetDims()
        element_count = CalcElementCount(dims)
        if element_count > 0:
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                logging.error("copy data from tensor[" + tensor.GetName() + "] failed.")
                sys.exit(-1)

            in_data = np.array(tensor_data, copy=False)
            fd.write(in_data.tobytes())
    fd.close()

# ---------------------------------------------------------------------------- #

def SaveOutputsOneByOne(save_data_dir, runtime):
    for i in range(runtime.GetOutputCount()):
        tensor = runtime.GetOutputTensor(i)
        out_file_name = save_data_dir + "/pplnn_output-" + tensor.GetName() + ".dat"
        dims = tensor.GetShape().GetDims()
        element_count = CalcElementCount(dims)
        if element_count > 0:
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                logging.error("copy data from tensor[" + tensor.GetName() + "] failed.")
                sys.exit(-1)

            out_data = np.array(tensor_data, copy=False)
            out_data.tofile(out_file_name)
        else:
            open(out_file_name, 'a').close()

# ---------------------------------------------------------------------------- #

def CalcBytes(dims, item_size):
    return item_size * CalcElementCount(dims)

def PrintInputOutputInfo(runtime):
    logging.info("----- input info -----")
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        dims = shape.GetDims()
        logging.info("input[" + str(i) + "]")
        logging.info("    name: " + tensor.GetName())
        logging.info("    dim(s): " + str(dims))
        logging.info("    type: " + pplcommon.GetDataTypeStr(shape.GetDataType()))
        logging.info("    format: " + pplcommon.GetDataFormatStr(shape.GetDataFormat()))
        logging.info("    byte(s) excluding padding: " + str(CalcBytes(dims, pplcommon.GetSizeOfDataType(shape.GetDataType()))))

    logging.info("----- output info -----")
    for i in range(runtime.GetOutputCount()):
        tensor = runtime.GetOutputTensor(i)
        shape = tensor.GetShape()
        dims = shape.GetDims()
        logging.info("output[" + str(i) + "]")
        logging.info("    name: " + tensor.GetName())
        logging.info("    dim(s): " + str(dims))
        logging.info("    type: " + pplcommon.GetDataTypeStr(shape.GetDataType()))
        logging.info("    format: " + pplcommon.GetDataFormatStr(shape.GetDataFormat()))
        logging.info("    byte(s) excluding padding: " + str(CalcBytes(dims, pplcommon.GetSizeOfDataType(shape.GetDataType()))))

# ---------------------------------------------------------------------------- #

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = ParseCommandLineArgs()

    if args.display_version:
        logging.info("PPLNN version: " + pplnn.GetVersionString())
        sys.exit(0)

    if not args.onnx_model:
        logging.error("'--onnx-model' is not specified.")
        sys.exit(-1)

    engines = RegisterEngines(args)
    if not engines:
        logging.error("no engine is specified. run '" + sys.argv[0] + " -h' to see supported device types marked with '--use-*'.")
        sys.exit(-1)

    runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(args.onnx_model, engines)
    if not runtime_builder:
        logging.error("create RuntimeBuilder failed.")
        sys.exit(-1)

    runtime = runtime_builder.CreateRuntime()
    if not runtime:
        logging.error("create Runtime instance failed.")
        sys.exit(-1)

    in_shapes = ParseInShapes(args.in_shapes)

    if args.inputs:
        SetInputsOneByOne(args.inputs, in_shapes, runtime)
    elif args.reshaped_inputs:
        SetReshapedInputsOneByOne(args.reshaped_inputs, runtime)
    else:
        SetRandomInputs(in_shapes, runtime)

    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        if CalcElementCount(shape.GetDims()) == 0:
            logging.error("input tensor[" + tensor.GetName() + "] is empty.")
            sys.exit(-1)

    if args.save_input:
        SaveInputsAllInOne(args.save_data_dir, runtime)
    if args.save_inputs:
        SaveInputsOneByOne(args.save_data_dir, runtime)

    status = runtime.Run()
    if status != pplcommon.RC_SUCCESS:
        logging.error("Run() failed: " + pplcommon.GetRetCodeStr(status))
        sys.exit(-1)

    PrintInputOutputInfo(runtime)

    if args.save_outputs:
        SaveOutputsOneByOne(args.save_data_dir, runtime)

    logging.info("Run ok")
