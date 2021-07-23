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

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>

#include <opencv2/opencv.hpp>

#include "ppl/nn/models/onnx/onnx_runtime_builder_factory.h"
#include "ppl/nn/engines/x86/engine_factory.h"
#include "ppl/nn/engines/x86/x86_options.h"
#include "ppl/kernel/x86/common/threading_tools.h"

#include "imagenet_labels.h"

using namespace std;
using namespace cv;
using namespace ppl::nn;
using namespace ppl::common;

// change OpenCV Mat to NCHW format fp32 data
int32_t ImagePreprocess(const Mat& src_img, float* in_data) {
    const int32_t height = src_img.rows;
    const int32_t width = src_img.cols;
    const int32_t channels = src_img.channels();

    // convert data from bgr/gray to rgb
    Mat rgb_img;
    if (channels == 3) {
        cvtColor(src_img, rgb_img, COLOR_BGR2RGB);
    } else if (channels == 1) {
        cvtColor(src_img, rgb_img, COLOR_GRAY2RGB);
    } else {
        fprintf(stderr, "unsupported channel num: %d\n", channels);
        return -1;
    }

    // split 3 channel to change HWC to CHW
    vector<Mat> rgb_channels(3);
    split(rgb_img, rgb_channels);

    // by this constructor, when cv::Mat r_channel_fp32 changed, in_data will also change
    Mat r_channel_fp32(height, width, CV_32FC1, in_data + 0 * height * width); 
    Mat g_channel_fp32(height, width, CV_32FC1, in_data + 1 * height * width);
    Mat b_channel_fp32(height, width, CV_32FC1, in_data + 2 * height * width);
    vector<Mat> rgb_channels_fp32{r_channel_fp32, g_channel_fp32, b_channel_fp32};

    // convert uint8 to fp32, y = (x - mean) / std
    const float mean[3] = {0, 0, 0}; // change mean & std according to your dataset & training param
    const float std[3] = {255.0f, 255.0f, 255.0f};
    for (uint32_t i = 0; i < rgb_channels.size(); ++i) {
        rgb_channels[i].convertTo(rgb_channels_fp32[i], CV_32FC1, 1.0f / std[i], -mean[i] / std[i]);
    }

    return 0;
}

// get classification result from network output
int32_t GetClassificationResult(const float* scores, const int32_t size) {
    vector<pair<float, int>> pairs(size);
    for (int32_t i = 0; i < size; i++) {
        pairs[i] = make_pair(scores[i], i);
    }

    auto cmp_func = [](const pair<float, int>& p0, const pair<float, int>& p1) -> bool {
        return p0.first > p1.first;
    };

    const int32_t top_k = 5;
    nth_element(pairs.begin(), pairs.begin() + top_k, pairs.end(), cmp_func); // get top K results & sort
    sort(pairs.begin(), pairs.begin() + top_k, cmp_func);

    printf("top %d results:\n", top_k);
    for (int32_t i = 0; i < top_k; ++i) {
        printf("%dth: %-10f %-10d %s\n", i + 1, pairs[i].first, pairs[i].second, imagenet_labels_tab[pairs[i].second]);
    }

    return 0;
}

// run classification model
int RunClassificationModel(const Mat& src_img, const char* onnx_model_path) {
    /************************ 1. preprocess image *************************/
    const int32_t height = src_img.rows;
    const int32_t width = src_img.cols;
    const int32_t channels = src_img.channels();

    vector<float> in_data_(height * width * channels); // network need NCHW(RGB order) fp32 data
    float* in_data = in_data_.data();

    int32_t ret = ImagePreprocess(src_img, in_data); // convert NHWC to NCHW & substract mean div std
    if (ret != 0) {
        fprintf(stderr, "image preprocess failed!\n");
        return ret;
    }

    printf("image preprocess succeed!\n");

    /************************ 2. create runtime builder from onnx model *************************/
    auto x86_engine = X86EngineFactory::Create(); // create x86 engine

    // register all engines you want to use
    vector<unique_ptr<Engine>> engines;
    vector<Engine*> engine_ptrs;
    engines.emplace_back(unique_ptr<Engine>(x86_engine));
    engine_ptrs.emplace_back(engines[0].get());

    // create onnx runtime builder according to onnx model & engines registered before
    auto builder = unique_ptr<OnnxRuntimeBuilder>(
        OnnxRuntimeBuilderFactory::Create(onnx_model_path, engine_ptrs.data(), engine_ptrs.size()));
    if (!builder) {
        fprintf(stderr, "create OnnxRuntimeBuilder from onnx model %s failed!\n", onnx_model_path);
        return -1;
    }

    printf("successfully create runtime builder!\n");

    /************************ 3. build runtime *************************/
    // configure runtime options
    RuntimeOptions runtime_options;
    runtime_options.mm_policy = MM_LESS_MEMORY; // configure to less memory usage

    // use runtime builder to build runtime, one builder can be used to build multiple runtimes sharing constant data & topo
    // here we only build one runtime for easy to understand
    unique_ptr<Runtime> runtime;
    runtime.reset(builder->CreateRuntime(runtime_options));
    if (!runtime) {
        fprintf(stderr, "build runtime failed!\n");
        return -1;
    }

    printf("successfully build runtime!\n");

    /************************ 4. set input data to runtime *************************/
    // reshape network's input tensor
    auto input_tensor = runtime->GetInputTensor(0);

    const std::vector<int64_t> input_shape{1, channels, height, width};
    input_tensor->GetShape().Reshape(input_shape); // pplnn can reshape input dynamically even if onnx model has static input shape
    auto status = input_tensor->ReallocBuffer();   // must do this after tensor's shape has changed
    if (status != RC_SUCCESS) {
        fprintf(stderr, "ReallocBuffer for tensor [%s] failed: %s\n", input_tensor->GetName(), GetRetCodeStr(status));
        return -1;
    }

    // set input data descriptor
    TensorShape src_desc = input_tensor->GetShape(); // description of your prepared data, not input tensor's description
    src_desc.SetDataType(DATATYPE_FLOAT32);
    src_desc.SetDataFormat(DATAFORMAT_NDARRAY); // for 4-D Tensor, NDARRAY == NCHW

    status = input_tensor->ConvertFromHost(in_data, src_desc); // convert data type & format from src_desc to input_tensor & fill data
    if (status != RC_SUCCESS) {
        fprintf(stderr, "set input data to tensor [%s] failed: %s\n", input_tensor->GetName(), GetRetCodeStr(status));
        return -1;
    }

    printf("successfully set input data to tensor [%s]!\n", input_tensor->GetName());

    /************************ 5. forward *************************/
    status = runtime->Run(); // forward
    if (status != RC_SUCCESS) {
        fprintf(stderr, "run network failed: %s\n", GetRetCodeStr(status));
        return -1;
    }

    status = runtime->Sync(); // wait for all ops run finished, not implemented yet.
    if (status != RC_SUCCESS) { // now sync is done by runtime->Run() function.
        fprintf(stderr, "runtime sync failed: %s\n", GetRetCodeStr(status));
        return -1;
    }

    printf("successfully run network!\n");

    /************************ 6. get output data *************************/
    // prepare output data's buffer
    auto output_tensor = runtime->GetOutputTensor(0);

    uint64_t output_size = output_tensor->GetShape().GetElementsExcludingPadding();
    std::vector<float> output_data_(output_size);
    float* output_data = output_data_.data();

    // set output data descriptor
    TensorShape dst_desc = output_tensor->GetShape(); // description of your output data buffer, not output_tensor's description
    dst_desc.SetDataType(DATATYPE_FLOAT32);
    dst_desc.SetDataFormat(DATAFORMAT_NDARRAY); // output is 1-D Tensor, NDARRAY == vector

    status = output_tensor->ConvertToHost(output_data, dst_desc); // convert data type & format from output_tensor to dst_desc
    if (status != RC_SUCCESS) {
        fprintf(stderr, "get output data from tensor [%s] failed: %s\n", output_tensor->GetName(),
                GetRetCodeStr(status));
        return -1;
    }

    printf("successfully get outputs!\n");

    return GetClassificationResult(output_data, output_size);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <image_file> <onnx_model_file>\n", argv[0]);
        return -1;
    }

    const char* img_path = argv[1];
    Mat src_img = imread(img_path);
    if (src_img.empty()) {
        fprintf(stderr, "read image file %s failed!\n", img_path);
        return -1;
    }

    const bool resize_input = false; // pplnn can adapt dynamic input size even if onnx model has static input size
    if (resize_input) {
        resize(src_img, src_img, Size(224, 224));
    }

    const char* onnx_model_path = argv[2];
    int32_t ret = RunClassificationModel(src_img, onnx_model_path);
    if (ret != 0) {
        fprintf(stderr, "run classification model failed!\n");
        return -1;
    }

    return 0;
}
