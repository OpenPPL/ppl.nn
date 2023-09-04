# PPLNN for LLM

`ppl.nn.llm` is a collection of Large Language Models(LLM) inferencing engines based on [ppl.nn](https://github.com/openppl-public/ppl.nn).

## Prerequisites

* Linux running on x86_64 or arm64 CPUs
* GCC >= 9.4.0
* [CMake](https://cmake.org/download/) >= 3.18
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.4. 11.6 recommended. (for CUDA)

## Quick Start

* Installing prerequisites(on Debian 12 or Ubuntu 20.04 for example):

```bash
apt-get install build-essential cmake git
```

* Cloning source code:

```bash
git clone https://github.com/openppl-public/ppl.nn.llm.git
```

* Building from source:

```bash
cd ppl.nn.llm
./build.sh -DPPLNN_USE_LLM_CUDA=ON -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'"
```

## Documents

Refer to [ppl.pmx](https://github.com/openppl-public/ppl.pmx) for how to export onnx models. Refer to APIs in [documents](https://github.com/openppl-public/ppl.nn#documents) of ppl.nn.

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
