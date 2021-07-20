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

import argparse
import onnx
from onnx import version_converter

def _replace_all_initializer_with_constant(graph):
    """Recursively replace all initializers of a graph with constant op.
    """
    for initializer in graph.initializer:
        constant_node_name = 'initializer_temp_constant_{}'.format(initializer.name)
        constant_node = onnx.helper.make_node(
            'Constant', 
            inputs = [], 
            outputs = [initializer.name], 
            name = constant_node_name, 
            value = initializer)
        graph.node.append(constant_node)
    
    graph.ClearField('initializer')

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField('g'):
                _replace_all_initializer_with_constant(attr.g)


def _find_and_remove_value_info_by_name(graph, name):
    """Find and remove value info in a graph by name.
    """
    for value_info in graph.value_info:
        if value_info.name == name:
            graph.value_info.remove(value_info)
            break


def _change_constant_back_to_initializer(graph):
    """Recursively change constant op back to initializer in a graph.
    """
    to_remove_node = []
    for node in graph.node:
        if node.op_type == 'Constant' and 'initializer_temp_constant_' in node.name:
            tensor = node.attribute[0].t
            graph.initializer.append(tensor)
            to_remove_node.append(node)
    
    for node in to_remove_node:
        _find_and_remove_value_info_by_name(graph, node.output[0])  # onnx version_converter may produce value_info, should delete it
        graph.node.remove(node)

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField('g'):
                _change_constant_back_to_initializer(attr.g)


def convert_onnx_opset_version(input_model_path, output_model_path, output_opset):
    """Convert onnx model opset version.

    A wrapper of onnx.version_converter. Avoid onnx.version_converter's intiailizer bug
    by move intializer to constant op temporarily, then run onnx.version_converter.convert_version()
    function, finally move constant op back to original intializer.

    onnx.version_converter's doc: https://github.com/onnx/onnx/blob/master/docs/VersionConverter.md

    Args:
        input_model_path: path to input onnx model.
        output_model_path: path to save output onnx model.
        output_opset: output onnx model opset version number.
    
    Returns:
        None
    
    Raises:
        onnx.load(), onnx.version_converter.convert_version() onnx.save() & other onnx method 
        may raise errors.
    """
    input_model = onnx.load(input_model_path)
    input_opset = input_model.opset_import[0].version
    if input_opset == output_opset:
        onnx.save(input_model, output_model_path)
        return

    _replace_all_initializer_with_constant(input_model.graph)
    output_model = version_converter.convert_version(input_model, output_opset)
    _change_constant_back_to_initializer(output_model.graph)

    onnx.save(output_model, output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert onnx model version.')
    parser.add_argument('--input_model', type=str, help='input onnx model path.')
    parser.add_argument('--output_model', type=str, help='output onnx model path.')
    parser.add_argument('--output_opset', type=int, help='output onnx model opset.')
    args = parser.parse_args()

    convert_onnx_opset_version(args.input_model, args.output_model, args.output_opset)
