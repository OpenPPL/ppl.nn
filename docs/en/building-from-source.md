### Prerequisites

* Linux running on x86 compatible CPUs
* GCC >= 4.9 or LLVM/Clang >= 6.0
* [CMake](https://cmake.org/download/) >= 3.13
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 10.2 (for CUDA version)
* [Python](https://www.python.org/downloads/) >= 3 (for CUDA version)

### Cloning Source Code

```bash
git clone https://github.com/openppl-public/ppl.nn.git
```

### Building X86-64 Engine

```bash
./build.sh
```

Headers and libraries are installed in `pplnn-build/install`.

If you want to enable openmp, please specify `HPCC_USE_OPENMP` as following:

```bash
./build.sh -DHPCC_USE_OPENMP=ON
```

If you are building on MacOS (Darwin), install `libomp` by [homebrew](https://brew.sh/) first:
```bash
brew install libomp
```

### Building CUDA Engine

X86-64 engine is enabled by default when building CUDA engine.

```bash
./build.sh -DHPCC_USE_CUDA=ON
```

Headers and libraries are installed in `pplnn-build/install`.

If you want to use specified CUDA toolkit version, please specify `CUDA_TOOLKIT_ROOT_DIR` as following:

```bash
./build.sh -DHPCC_USE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda-toolkit-root-dir
```

### Buliding Python API support

`PPLNN` builds python API by default. If you want to use a specified version of python, you can pass `PYTHON3_INCLUDE_DIRS` to `build.sh`:

```bash
./build.sh -DPYTHON3_INCLUDE_DIRS=/path/to/your/python/include/dir
```

### Testing

There is a test tool named `pplnn` in `tools/pplnn.cc`. You can run `pplnn` using the following command:

```bash
./pplnn-build/tools/pplnn --onnx-model tests/testdata/conv.onnx
```

NOTE: if CUDA engine is enabled, `pplnn` uses CUDA only.

You can run the python demo with:

```bash
PYTHONPATH=./pplnn-build/install python3 ./tools/pplnn.py --use-x86 --onnx-model tests/testdata/conv.onnx
```

or

```bash
PYTHONPATH=./pplnn-build/install python3 ./tools/pplnn.py --use-cuda --onnx-model tests/testdata/conv.onnx
```

or use both engines:

```bash
PYTHONPATH=./pplnn-build/install python3 ./tools/pplnn.py --use-x86 --use-cuda --onnx-model tests/testdata/conv.onnx
```

### Installing Pyppl Modules Using `Pip`

There is a python packaging configuration in [python/package](../../python/package). You can install pyppl using `pip`:

```bash
cd ppl.nn
rm -rf /tmp/pyppl-package # remove old packages
cp -r python/package /tmp/pyppl-package
cp -r pplnn-build/install/pyppl/* /tmp/pyppl-package/pyppl # Copy .so files. See WARNING below.
cd /tmp/pyppl-package
pip3 install .
```

**WARNING**: `Pip3` will delete all of the installed pyppl .so files before installation. Make sure that you have put all the .so files needed in `package/pyppl` before executing `pip3 install .`.

After installation, you can use `from pyppl import nn` directly without setting the `PYTHONPATH` env.
