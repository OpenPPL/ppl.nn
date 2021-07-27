## PPLNN

### Overview

`PPLNN`, which is short for "**P**PLNN is a **P**rimitive **L**ibrary for **N**eural **N**etwork", is a high-performance deep-learning inference engine for efficient AI inferencing. It can run various ONNX models and has better support for [OpenMMLab](https://github.com/open-mmlab).

![alt arch](docs/images/arch.png)

### Documents

* [Supported Ops and Platforms](docs/en/supported-ops-and-platforms.md)
* [Building from Source](docs/en/building-from-source.md)
* [Generating ONNX models from OpenMMLab](docs/en/model-convert-guide.md)
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
    - [Adding Ops](docs/en/x86-doc/add_op.md)（[中文版](docs/cn/x86-doc/add_op.md)）
    - [Benchmark](docs/en/x86-doc/benchmark_tool.md)（[中文版](docs/cn/x86-doc/benchmark_tool.md)）
  - CUDA
    - [Adding Ops](docs/en/cuda-doc/add_op.md)（[中文版](docs/cn/cuda-doc/add_op.md)）
    - [Benchmark](docs/en/cuda-doc/benchmark_tool.md)（[中文版](docs/cn/cuda-doc/benchmark_tool.md)）

### Contact Us

* [OpenPPL Homepage](https://openppl.ai/)
* [OpenPPL on Github](https://github.com/openppl-public/ppl.nn)

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
