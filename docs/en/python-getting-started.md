This section shows how to use python APIs. Refer to [python API reference](python-api-reference.md) for more details and [pplnn.py](../../tools/pplnn.py) for usage examples.

For brevity, all code snippets assume that the following two lines are present:

```python
import pypplnn as pplnn
import pypplcommon as pplcommon
```

### Creating Engines

In `PPLNN`, an `Engine` is a collection of op implementations running on specified devices such as CPU or NVIDIA GPU. For example, we can use the built-in `X86EngineFactory`:

```python
x86_engine = pplnn.X86EngineFactory.Create()
```

to create an engine running on x86-compatible CPUs, or use

```python
cuda_options = CudaEngineOptions()
cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
```

to create an engine running on NVIDIA GPUs.

### Creating a RuntimeBuilder

Use

```python
onnx_model_file = "/path/to/onnx_model_file"
engines = [pplnn.Engine(x86_engine)] # or engines = [pplnn.Engine(cuda_engine)]
pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(onnx_model_file, engines)
```

to create a `RuntimeBuilder`, which is used for creating `Runtime` instances. Note that `x86_engine` and `cuda_engine` need to be converted to `Engine` explicitly.

`PPLNN` also supports multiple engines running in the same model. For example:

```python
engines = [pplnn.Engine(x86_engine), pplnn.Engine(cuda_engine)]
pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(onnx_model_file, engines)
```

`PPLNN` will partition the model into several parts and assign different ops to these engines according to configurations.

### Creating a Runtime Instance

```python
runtime = runtime_builder.CreateRuntime()
```

### Filling Inputs

We can get graph inputs using the following functions of `Runtime`:

```python
input_count = runtime.GetInputCound()
tensor = runtime.GetInputTensor(idx)
```

and fill input data(using randomg data in this snippet):

```python
for i in range(runtime.GetInputCount()):
    tensor = runtime.GetInputTensor(i)
    dims = GenerateRandomDims(tensor.GetShape())

    in_data = np.random.uniform(-1.0, 1.0, dims)
    status = tensor.CopyFromHost(in_data)
    if status != pplcommon.RC_SUCCESS:
        logging.error("copy data to tensor[" + tensor.GetName() + "] failed: " +
                      pplcommon.GetRetCodeStr(status))
        sys.exit(-1)
```

### Evaluating the Model

```python
ret_code = runtime.Run()
```

and waits for all operations to finish(some engine may run asynchronously):

```python
ret_code = runtime.Sync()
```

### Getting Results

```python
for i in range(runtime.GetOutputCount()):
    tensor = runtime.GetOutputTensor(i)
    shape = tensor.GetShape()
    ndarray = tensor.CopyToHost()
    out_data = np.array(ndarray, copy=False)
```
