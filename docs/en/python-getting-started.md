This section shows how to use python APIs. Refer to [python API reference](python-api-reference.md) for more details and [pplnn.py](../../tools/pplnn.py) for usage examples.

For brevity, all code snippets assume that the following two lines are present:

```python
from pyppl import nn as pplnn
from pyppl import common as pplcommon
```

### Creating Engines

In `PPLNN`, an `Engine` is a collection of op implementations running on specified devices such as CPU or NVIDIA GPU. For example, we can use the built-in `x86.EngineFactory`:

```python
x86_options = pplnn.x86.EngineOptions()
x86_engine = pplnn.x86.EngineFactory.Create(x86_options)
```

to create an engine running on x86-compatible CPUs, or use

```python
cuda_options = pplnn.cuda.EngineOptions()
cuda_engine = pplnn.cuda.EngineFactory.Create(cuda_options)
```

to create an engine running on NVIDIA GPUs.

### Creating an OnnxRuntimeBuilder

Use

```python
runtime_builder = pplnn.onnx.RuntimeBuilderFactory.Create()
```

to create a `onnx.RuntimeBuilder`, which is used for creating `Runtime` instances.

### Creating a Runtime Instance

```python
onnx_model_file = "/path/to/onnx_model_file"
status = runtime_builder.LoadModelFromFile(onnx_model_file)
```

loads an ONNX model from the specified file.

```python
resources = RuntimeBuilderResources()
resources.engines = [x86_engine] # or = [cuda_engine]
runtime_builder.SetResources(resources)
```

`PPLNN` also supports multiple engines running in the same model. For example:

```python
resources.engines = [x86_engine, cuda_engine]
status = runtime_builder.SetResources(resources)
```

The model will be partitioned into several parts and assign different ops to these engines automatically.

```python
status = runtime_builder.Preprocess()
```

does some preparations before creating `Runtime` instances.

```python
runtime = runtime_builder.CreateRuntime()
```

creates a `Runtime` instances.

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
