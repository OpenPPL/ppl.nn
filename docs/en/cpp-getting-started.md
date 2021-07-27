This section shows how to use `PPLNN` step by step with an example [api_intro.cc](../../samples/cpp/api/api_intro.cc). Refer to [API Reference](cpp-api-reference.md) for more details.

### Creating engines

In `PPLNN`, an `Engine` is a collection of op implementations running on specified devices such as CPU or NVIDIA GPU. For example, we can use the built-in `X86EngineFactory`:

```c++
Engine* X86EngineFactory::Create();
```

to create an engine which runs on x86-compatible CPUs:

```c++
Engine* x86_engine = X86EngineFactory::Create();
```

Or use

```c++
CudaEngineOptions options;
// ... set options
Engine* CudaEngineFactory::Create(options);
```

to create an engine running on NVIDIA GPUs.

### Creating a RuntimeBuilder

We create a `RuntimeBuilder` with the following function:

```c++
OnnxRuntimeBuilder* OnnxRuntimeBuilderFactory::Create(
    const char* model_file, std::vector<std::unique_ptr<Engine>>&& engines);
```

where the second parameter `engines` is the `x86_engine` we created:

```c++
vector<unique_ptr<Engine>> engines;
engines.emplace_back(unique_ptr<Engine>(x86_engine));

const char* model_file = "tests/testdata/conv.onnx";

RuntimeBuilder* builder = OnnxRuntimeBuilderFactory::Create(model_file, std::move(engines));
```

`PPLNN` supports multiple engines running in the same model. For example:

```c++
Engine* x86_engine = X86EngineFactory::Create();
Engine* cuda_engine = CudaEngineFactory::Create(CudaEngineOptions());

vector<unique_ptr<Engine>> engines;
engines.emplace_back(unique_ptr<Engine>(x86_engine));
engines.emplace_back(unique_ptr<Engine>(cuda_engine));
// add other engines

const char* model_file = "/path/to/onnx/model";
// use x86 and cuda engines to run this model
RuntimeBuilder* builder = OnnxRuntimeBuilderFactory::Create(model_file, std::move(engines));
```

`PPLNN` will partition the model and assign different ops to these engines according to configurations.

### Creating a Runtime

We can use

```c++
Runtime* OnnxRuntimeBuilder::CreateRuntime(const RuntimeOptions&);
```

to create a `Runtime`:

```c++
RuntimeOptions runtime_options;
Runtime* runtime = builder->CreateRuntime(runtime_options);
```

### Filling Inputs

We can get graph inputs using the following functions of `Runtime`:

```c++
uint32_t Runtime::GetInputCount() const;
Tensor* Runtime::GetInputTensor(uint32_t idx) const;
```

and fill input data(using random data in this example):

```c++
for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
    auto t = runtime->GetInputTensor(c);
    auto& shape = t->GetShape();

    auto nr_element = shape.GetBytesIncludingPadding() / sizeof(float);
    unique_ptr<float[]> buffer(new float[nr_element]);

    // fill random input data
    std::default_random_engine eng;
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (uint32_t i = 0; i < nr_element; ++i) {
        buffer.get()[i] = dis(eng);
    }

    auto status = t->ReallocBuffer();
    if (status != RC_SUCCESS) {
        // ......
    }

    // our random data is treated as NDARRAY
    TensorShape src_desc = t->GetShape();
    src_desc.SetDataFormat(DATAFORMAT_NDARRAY);

    // input tensors may require different data format
    status = t->ConvertFromHost(buffer.get(), src_desc);
    if (status != RC_SUCCESS) {
        // ......
    }
}
```

### Evaluating the Compute Graph

use the `Runtime::Run()`:

```c++
RetCode status = runtime->Run();
```

### Getting Results

Before getting results we must wait for all operations to finish(some engine may run asynchronously):

```c++
RetCode status = runtime->Sync();
```

Then iterate each output:

```c++
for (uint32_t c = 0; c < runtime->GetOutputCount(); ++c) {
    auto t = runtime->GetOutputTensor(c);

    TensorShape dst_desc = t->GetShape();
    dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);
    auto bytes = dst_desc.GetBytesIncludingPadding();
    unique_ptr<char[]> buffer(new char[bytes]);

    auto status = t->ConvertToHost(buffer.get(), dst_desc);
    // ......
}
```
