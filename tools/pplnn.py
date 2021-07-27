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
import pypplnn as pplnn
import pypplcommon as pplcommon

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

    parser.add_argument("--onnx-model", type = str, default = "", required = False,
                        help = "onnx model file")

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

def RegisterEngines(args):
    engines = []
    if args.use_x86:
        x86_engine = pplnn.X86EngineFactory.Create()
        if not x86_engine:
            logging.error("create x86 engine failed.")
            sys.exit(-1)
        if args.disable_avx512:
            x86_engine.Configure(pplnn.X86_CONF_DISABLE_AVX512)

        engines.append(pplnn.Engine(x86_engine))
    if args.use_cuda:
        cuda_options = pplnn.CudaEngineOptions()
        cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
        if not cuda_engine:
            logging.error("create cuda engine failed.")
            sys.exit(-1)
        engines.append(pplnn.Engine(cuda_engine))

    return engines

# ---------------------------------------------------------------------------- #

def ParseInShapes(in_shapes_str):
    ret = []
    shape_strs = list(filter(None, in_shapes_str.split(",")))
    for s in shape_strs:
        dims = [int(d) for d in s.split("_")]
        ret.append(dims)
    return ret

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
        in_data = np.fromfile(input_files[i], dtype=shape.GetDataType())
        tensor.CopyFromHost(in_data)

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
        in_data = np.fromfile(input_files[i], dtype = np_data_type)
        in_data = in_data.reshape(input_shape)
        status = tensor.CopyFromHost(in_data)
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
        if not in_shapes:
            dims = GenerateRandomDims(shape)
        else:
            dims = in_shapes[i]

        in_data = (upper_bound - lower_bound) * rng.random(dims, dtype=g_pplnntype2numpytype[shape.GetDataType()]) * lower_bound
        status = tensor.CopyFromHost(in_data)
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

def SaveInputsOneByOne(save_data_dir, runtime):
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shape = tensor.GetShape()
        tensor_data = tensor.CopyToHost()
        in_data = np.array(tensor_data, copy=False)
        in_data.tofile(save_data_dir + "/pplnn_input_" + str(i) + "_" +
                       tensor.GetName() + "-" + GenDimsStr(shape.GetDims()) + "-" +
                       g_data_type_str[shape.GetDataType()] + ".dat")

# ---------------------------------------------------------------------------- #

def SaveInputsAllInOne(save_data_dir, runtime):
    out_file_name = save_data_dir + "/pplnn_inputs.dat"
    fd = open(out_file_name, mode="wb+")
    for i in range(runtime.GetInputCount()):
        tensor = runtime.GetInputTensor(i)
        shsape = tensor.GetShape()
        tensor_data = tensor.CopyToHost()
        in_data = np.array(tensor_data, copy=False)
        fd.write(in_data.tobytes())
    fd.close()

# ---------------------------------------------------------------------------- #

def SaveOutputsOneByOne(save_data_dir, runtime):
    for i in range(runtime.GetOutputCount()):
        tensor = runtime.GetOutputTensor(i)
        shape = tensor.GetShape()
        tensor_data = tensor.CopyToHost()
        out_data = np.array(tensor_data, copy=False)
        out_data.tofile(save_data_dir + "/pplnn_output-" + tensor.GetName() + ".dat")

# ---------------------------------------------------------------------------- #

def CalcBytes(dims, item_size):
    nbytes = item_size
    for d in dims:
        nbytes = nbytes * d
    return nbytes

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

    args = ParseCommandLineArgs();

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
        logging.error("create OnnxRuntimeBuilder failed.")
        sys.exit(-1)

    runtime_options = pplnn.RuntimeOptions()
    runtime = runtime_builder.CreateRuntime(runtime_options)
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

    if args.save_input:
        SaveInputsAllInOne(args.save_data_dir, runtime)
    if args.save_inputs:
        SaveInputsOneByOne(args.save_data_dir, runtime)

    status = runtime.Run()
    if status != pplcommon.RC_SUCCESS:
        logging.error("Run() failed: " + pplcommon.GetRetCodeStr(status))
        sys.exit(-1)

    status = runtime.Sync()
    if status != pplcommon.RC_SUCCESS:
        logging.error("Run() failed: " + pplcommon.GetRetCodeStr(status))
        sys.exit(-1)

    PrintInputOutputInfo(runtime)

    if args.save_outputs:
        SaveOutputsOneByOne(args.save_data_dir, runtime)

    logging.info("Run ok")
