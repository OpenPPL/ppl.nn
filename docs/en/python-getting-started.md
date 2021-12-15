This section shows how to use python APIs. Refer to [python API reference](python-api-reference.md) for more details and [pplnn.py](../../tools/pplnn.py) for usage examples.

For brevity, all code snippets assume that the following two lines are present:

```python
from pyppl import nn as pplnn
from pyppl import common as pplcommon
```

### Creating Engines

In `PPLNN`, an `Engine` is a collection of op implementations running on specified devices such as CPU or NVIDIA GPU. For example, we can use the built-in `X86EngineFactory`:

```python
x86_options = pplnn.X86EngineOptions()
x86_engine = pplnn.X86EngineFactory.Create(x86_options)
```

to create an engine running on x86-compatible CPUs, or use

```python
cuda_options = pplnn.CudaEngineOptions()
cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
```

to create an engine running on NVIDIA GPUs.

### Creating a RuntimeBuilder

Use

```python
onnx_model_file = "/path/to/onnx_model_file"
engines = [pplnn.Engine(x86_engine)] # or engines = [pplnn.Engine(cuda_engine)]
runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(onnx_model_file, engines)
```

to create a `RuntimeBuilder`, which is used for creating `Runtime` instances. Note that `x86_engine` and `cuda_engine` need to be converted to `Engine` explicitly.

`PPLNN` also supports multiple engines running in the same model. For example:

```python
engines = [pplnn.Engine(x86_engine), pplnn.Engine(cuda_engine)]
runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(onnx_model_file, engines)
```

`PPLNN` will partition the model into several parts and assign different ops to these engines according to configurations.

### Creating a Runtime Instance

```python
runtime = runtime_builder.CreateRuntime()
```

### Filling Inputs

We can get graph inputs using the following functions of `Runtime`:

```python
input_count = runtime.GetInputCount()
tensor = runtime.GetInputTensor(idx)
```

and fill input data(using randomg data in this snippet):

```python
for i in range(runtime.GetInputCount()):
    tensor = runtime.GetInputTensor(i)
    dims = GenerateRandomDims(tensor.GetShape())

    in_data = np.random.uniform(-1.0, 1.0, dims)
    status = tensor.ConvertFromHost(in_data)
    if status != pplcommon.RC_SUCCESS:
        logging.error("copy data to tensor[" + tensor.GetName() + "] failed: " +
                      pplcommon.GetRetCodeStr(status))
        sys.exit(-1)
```

### Evaluating the Model

```python
ret_code = runtime.Run()
```

### Getting Results

```python
for i in range(runtime.GetOutputCount()):
    tensor = runtime.GetOutputTensor(i)
    shape = tensor.GetShape()
    tensor_data = tensor.ConvertToHost()
    out_data = np.array(tensor_data, copy=False)
```
