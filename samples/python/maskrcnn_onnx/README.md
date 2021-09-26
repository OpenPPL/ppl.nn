# MaskRcnn ONNX Model Inference With The PPL Backend In Python

**Table Of Contents**

- [MaskRcnn ONNX Model Inference With The PPL Backend In Python](#maskrcnn-onnx-model-inference-with-the-ppl-backend-in-python)
  - [Description](#description)
  - [How does this sample work?](#how-does-this-sample-work)
  - [Prerequisites](#prerequisites)
  - [Running the sample](#running-the-sample)
  - [Additional resourcesV](#additional-resourcesv)

## Description

This sample, run_maskrcnn_onnx, implements a ONNX-based pipeline for performing inference with the maskrcnn network, with an input size of 800 x 1200 pixels, including pre and post-processing. 

## How does this sample work?

- First, convert maskrcnn from the [mmdetection](https://github.com/open-mmlab/mmdetection) to the Open Neural Network Exchange (ONNX) format.

- Second, use this ONNX Model of MaskRcnn to build a PPL engine, followed by inference on a sample image. 

- After inference, draw predicted bounding boxes and masks on the original input image and saved to disk.

## Prerequisites

1. Install the dependencies for Python.

    ```sh
    python3 -m pip install numpy opencv-python argparse
    ```

2. Download sample data.

## Running the sample

1. Create an ONNX model of MaskRcnn use [mmdetection/tools/deployment/pytorch2onnx.py](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/pytorch2onnx.md),you can also follow this [tutorial](https://github.com/openppl-public/ppl.nn/blob/master/docs/en/model-convert-guide.md)

2. Build a PPL engine from the generated ONNX file and run inference on a sample image

    ```sh
    PYTHONPATH=./pplnn-build/install python3.8 ./samples/python/maskrcnn_onnx/run_maskrcnn_onnx.py -i tests/testdata/cat0.png -o cat0.jpg -m mask_rcnn.onnx
    ```

## Additional resourcesV

The following resources provide a deeper understanding about the model used in this sample, as well as the dataset it was trained on:

**Model**
- [MaskRcnn: A conceptually simple, flexible, and general framework for object instance segmentation.](https://arxiv.org/abs/1703.06870)

**Dataset**
- [COCO dataset](http://cocodataset.org/#home)

