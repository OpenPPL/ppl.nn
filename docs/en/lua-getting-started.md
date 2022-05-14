This section shows how to use Lua APIs. Refer to [Lua API reference](lua-api-reference.md) for more details and [pplnn.lua](../../tools/pplnn.lua) for usage examples.

For brevity, all code snippets assume that the following two lines are present:

```lua
local pplnn = require("luappl.nn")
local pplcommon = require("luappl.common")
```

### Creating Engines

In `PPLNN`, an `Engine` is a collection of op implementations running on specified devices such as CPU or NVIDIA GPU. For example, we can use the built-in `x86.EngineFactory`:

```lua
x86_options = pplnn.x86.EngineOptions()
x86_engine = pplnn.x86.EngineFactory:Create(x86_options)
```

to create an engine running on x86-compatible CPUs, or use

```lua
cuda_options = pplnn.cuda.EngineOptions()
cuda_engine = pplnn.cuda.EngineFactory:Create(cuda_options)
```

to create an engine running on NVIDIA GPUs.

### Creating a RuntimeBuilder

Use

```lua
runtime_builder = pplnn.onnx.RuntimeBuilderFactory:Create()
```

to create a `onnx.RuntimeBuilder`, which is used for creating `Runtime` instances.

### Creating a Runtime Instance

```lua
onnx_model_file = "/path/to/onnx_model_file"
status = runtime_builder:LoadModelFromFile(onnx_model_file)
```

loads an ONNX model from the specified file.

```lua
resources = pplnn.onnx.RuntimeBuilderResources()
resources.engines = [x86_engine] # or = [cuda_engine]
status = runtime_builder:SetResources(resources)
```

`PPLNN` also supports multiple engines running in the same model. For example:

```lua
resources.engines = [x86_engine, cuda_engine]
status = runtime_builder:SetResources(resources)
```

`PPLNN` will partition the model into several parts and assign different ops to these engines automatically.

```lua
runtime_builder:Preprocess()
```

does some preparations before creating `Runtime` instances.

```lua
runtime = runtime_builder:CreateRuntime()
```

creates a `Runtime` instance.

### Filling Inputs

We can get graph inputs using the following functions of `Runtime`:

```lua
input_count = runtime:GetInputCound()
tensor = runtime:GetInputTensor(idx)
```

and fill input data(using randomg data in this snippet):

```lua
for i =1, runtime.GetInputCount() do
    local tensor = runtime:GetInputTensor(i - 1)
    local shape = tensor:GetShape()
    local data_type = shape:GetDataType()

    local dims = shape:GetDims()
    local in_data = GenerateRandomData(dims)
    local status = tensor:ConvertFromHost(in_data, dims, data_type)
    if status ~= pplcommon.RC_SUCCESS then
        logging.error("copy data to tensor[" .. tensor:GetName() .. "] failed: " ..
                      pplcommon.GetRetCodeStr(status))
        os.exit(-1)
    end
end
```

### Evaluating the Model

```lua
ret_code = runtime:Run()
```

### Getting Results

```lua
for i = 1, runtime.GetOutputCount() do
    local tensor = runtime:GetOutputTensor(i - 1)
    local shape = tensor:GetShape()
    local dims = shape:GetDims()
    local tensor_data = tensor:ConvertToHost()
end
```
