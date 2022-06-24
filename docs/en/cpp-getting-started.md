This section shows how to use `PPLNN` step by step with an example [api_intro.cc](../../samples/cpp/api/api_intro.cc). Refer to [API Reference](cpp-api-reference.md) for more details.

For brevity, we assume that `using namespace ppl::nn;` is always used in the following code snippets.

### Creating engines

In `PPLNN`, an `Engine` is a collection of op implementations running on specified devices such as CPU or NVIDIA GPU. For example, we can use the built-in `x86::EngineFactory`:

```c++
Engine* x86::EngineFactory::Create(const x86::EngineOptions&);
```

to create an engine which runs on x86-compatible CPUs:

```c++
x86::EngineOptions x86_options;
Engine* x86_engine = x86::EngineFactory::Create(x86_options);
```

Or use

```c++
cuda::EngineOptions cuda_options;
// ... set options
Engine* cuda::EngineFactory::Create(cuda_options);
```

to create an engine running on NVIDIA GPUs.

### Registering Built-in Op Implementations(optional)

For example, use `x86::RegisterBuiltinOpImpls()` to load built-in op implementations. You may also need to call `cuda::RegisterBuiltinOpImpls()` for cuda engine, etc.

### Creating an ONNX RuntimeBuilder

We can create an `onnx::RuntimeBuilder` with the following function:

```c++
onnx::RuntimeBuilder* onnx::RuntimeBuilderFactory::Create();
```

and load model from a file or a buffer:

```c++
ppl::common::RetCode LoadModel(const char* model_file);
ppl::common::RetCode LoadModel(const char* model_buf, uint64_t buf_len);
```

Then we set the resources used for processing:

```c++
struct Resources final {
    Engine** engines;
    uint32_t engine_num;
};

ppl::common::RetCode SetResources(const Resources&)
```

where the field `engines` of `Resources` is the `x86_engine` we created:

```c++
vector<Engine*> engine_ptrs;
engines.push_back(x86_engine);
resources.engines = engine_ptrs.data();
```

Note that the caller **MUST** guarantee that elements of `engines` are valid during the life cycle of the `Runtime` object.

`PPLNN` also supports multiple engines running in the same model. For example:

```c++
Engine* x86_engine = x86::EngineFactory::Create(x86::EngineOptions());
Engine* cuda_engine = cuda::EngineFactory::Create(cuda::EngineOptions());

vector<unique_ptr<Engine>> engines;
engines.emplace_back(unique_ptr<Engine>(x86_engine));
engines.emplace_back(unique_ptr<Engine>(cuda_engine));
// TODO add other engines

// use x86 and cuda engines to run this model
vector<Engine*> engine_ptrs = {x86_engine, cuda_engine};

Resources resources;
resources.engines = engine_ptrs.data();
resources.engine_num = engine_ptrs.size();
...
```

`PPLNN` will partition the model and assign different ops to these engines automatically.

### Creating a Runtime

Before creating `Runtime` instances we need to do some preparations:

```c++
builder->Preprocess();
```

then use

```c++
Runtime* RuntimeBuilder::CreateRuntime();
```

to create a `Runtime`.

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
    auto shape = t->GetShape();

    auto nr_element = shape->CalcBytesIncludingPadding() / sizeof(float);
    vector<float> buffer(nr_element);

    // fill random input data
    std::default_random_engine eng;
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (uint32_t i = 0; i < nr_element; ++i) {
        buffer[i] = dis(eng);
    }

    auto status = t->ReallocBuffer();
    if (status != RC_SUCCESS) {
        // ......
    }

    // our random data is treated as NDARRAY
    TensorShape src_desc = *t->GetShape();
    src_desc.SetDataFormat(DATAFORMAT_NDARRAY);

    // input tensors may require different data format
    status = t->ConvertFromHost((const void*)buffer.data(), src_desc);
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

Iterates each output:

```c++
for (uint32_t c = 0; c < runtime->GetOutputCount(); ++c) {
    auto t = runtime->GetOutputTensor(c);

    const TensorShape* dst_desc = t->GetShape();
    dst_desc->SetDataFormat(DATAFORMAT_NDARRAY);
    auto bytes = dst_desc->CalcBytesIncludingPadding();
    vector<char> buffer(bytes);

    auto status = t->ConvertToHost((void*)buffer.data(), *dst_desc);
    // ......
}
```
