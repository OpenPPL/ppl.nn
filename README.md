# PPLNN

[![website](docs/images/Website-OpenPPL-brightgreen.svg)](https://openppl.ai/)
[![License](docs/images/License-Apache-2.0-green.svg)](LICENSE)
[![qq](docs/images/Chat-on-QQ-red.svg)](https://qm.qq.com/cgi-bin/qm/qr?k=X7JWUqOdBih71dUU9AZF2gD3PKjxaxB-)
[![zhihu](docs/images/Discuss-on-Zhihu.svg)](https://www.zhihu.com/people/openppl)

### Overview

`PPLNN`, which is short for "**P**PLNN is a **P**rimitive **L**ibrary for **N**eural **N**etwork", is a high-performance deep-learning inference engine for efficient AI inferencing. It can run various ONNX models and has better support for [OpenMMLab](https://github.com/open-mmlab).

![alt arch](docs/images/arch.png)

### About LLM_CUDA

## Features

- Flash Attention
- Split-k Attention(Similar with Flash Decoding)
- Group-query Attention
- Dynamic Batching(Also called Continous Batching or In-flight Batching)
- Tensor Parallelism
- Graph Optimization
- INT8 groupwise KV Cache(Numerical accuracy is very close to FP16🚀)
- INT8 per token per channel Quantization(W8A8)

## Model Zoo

- [LLaMA 1/2/3](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/llama)
- [ChatGLM 2/3](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/chatglm2)
- [Baichuan 1/2 7B](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/baichuan)
- [InternLM 1](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/internlm)
- [InternLM 2](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/internlm2)
- [Mixtral](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/mixtral)
- [Qwen 1/1.5](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/qwen)
- [Falcon](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/falcon)
- [Bigcode](https://github.com/openppl-public/ppl.pmx/tree/master/model_zoo/bigcode)

## Known Issues

 - NCCL issue on some Device: Currently reported that L40S and H800 may encounter illegal memory access on NCCL AllReduce. We suggest trying to turn NCCL protocol `Simple` off by setting environment `NCCL_PROTO=^Simple` to fix this issue.

## Detials

[Here](docs/en/llm-cuda-overview.md)

### Hello, world!

* Installing prerequisites:

    - On Debian or Ubuntu:

    ```bash
    apt-get install build-essential cmake git python3 python3-dev
    ```

    - On RedHat or CentOS:

    ```bash
    yum install gcc gcc-c++ cmake3 make git python3 python3-devel
    ```

* Cloning source code:

```bash
git clone https://github.com/openppl-public/ppl.nn.git
```

* Building from source:

```bash
cd ppl.nn
./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_ENABLE_PYTHON_API=ON
```

* Running python demo:

```bash
PYTHONPATH=./pplnn-build/install/lib python3 ./tools/pplnn.py --use-x86 --onnx-model tests/testdata/conv.onnx
```

Refer to [Documents](#documents) for more details.

### Documents

* [Building from Source](docs/en/building-from-source.md)
* [How to Integrate](docs/en/how-to-integrate.md)
* APIs
  - C++
    - [Getting Started](docs/en/cpp-getting-started.md)
    - [API Reference](docs/en/cpp-api-reference.md)
  - Python
    - [Getting Started](docs/en/python-getting-started.md)
    - [API Reference](docs/en/python-api-reference.md)
* Develop Guide
  - [Adding New Engines and Ops](docs/en/add-new-engines-and-ops.md)
  - X86
    - [Supported Ops and Platforms](docs/en/x86-doc/supported-ops-and-platforms.md)
    - [Adding Ops](docs/en/x86-doc/add_op.md)（[中文版](docs/cn/x86-doc/add_op.md)）
    - [Benchmark](docs/en/x86-doc/benchmark_tool.md)（[中文版](docs/cn/x86-doc/benchmark_tool.md)）
  - CUDA
    - [Supported Ops and Platforms](docs/en/cuda-doc/supported-ops-and-platforms.md)
    - [Adding Ops](docs/en/cuda-doc/add_op.md)（[中文版](docs/cn/cuda-doc/add_op.md)）
    - [Benchmark](docs/en/cuda-doc/benchmark_tool.md)（[中文版](docs/cn/cuda-doc/benchmark_tool.md)）
  - RISCV
    - [Supported Ops and Platforms](docs/en/riscv-doc/supported-ops-and-platforms.md)
    - [Adding Ops](docs/en/riscv-doc/add_op.md)（[中文版](docs/cn/riscv-doc/add_op.md)）
    - [Benchmark](docs/en/riscv-doc/benchmark_tool.md)（[中文版](docs/cn/riscv-doc/benchmark_tool.md)）
  - ARM
    - [Adding Ops](docs/en/arm-doc/add_op.md)（[中文版](docs/cn/arm-doc/add_op.md)）
    - [Benchmark](docs/en/arm-doc/benchmark_tool.md)（[中文版](docs/cn/arm-doc/benchmark_tool.md)）
  - LLM-CUDA
    - [Overview](docs/en/llm-cuda-overview.md)
* Models
  - [Converting ONNX Opset](docs/en/onnx-model-opset-convert-guide.md)
  - [Generating ONNX models from OpenMMLab](docs/en/model-convert-guide.md)
* [实现细节](docs/cn/details.md)

### Contact Us

Questions, reports, and suggestions are welcome through GitHub Issues!

| WeChat Official Account | QQ Group |
| :----:| :----: |
| OpenPPL | 627853444 |
| ![OpenPPL](docs/images/qrcode_for_gh_303b3780c847_258.jpg)| ![QQGroup](docs/images/qqgroup_s.jpg) |

### Contributions

This project uses [Contributor Covenant](https://www.contributor-covenant.org/) as code of conduct. Any contributions would be highly appreciated.

### Acknowledgements

* [onnxruntime](https://github.com/microsoft/onnxruntime)
* [onnx](https://github.com/onnx/onnx)
* [openvino](https://github.com/openvinotoolkit/openvino)
* [oneDNN](https://github.com/oneapi-src/oneDNN)
* [TensorRT](https://github.com/NVIDIA/TensorRT)
* [OpenMMLab](https://github.com/open-mmlab)

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
