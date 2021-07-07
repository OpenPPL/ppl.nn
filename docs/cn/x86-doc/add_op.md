## 自定义算子添加

添加自定义算子需要先添加算子定义及其在不同架构上的实现。本章仅介绍在x86架构上添加算子的实现细节。

### 0. 概述

PPLNN在x86架构上添加自定义算子的步骤如下：

1) 添加算子参数定义与解析(如果算子无需参数，或参数定义与解析已添加，则跳过此步骤)
2) 添加算子定义，包括数据类型推断、维度计算、数据排布选择等
3) 添加算子调用接口
4) 添加kernel函数

几点约定：

* 本章中自定义算子的名称将以\<opname\>来代替。
* 本章将以框架中已实现的LeakyReLU为例，来帮助理解添加自定义算子的流程。**在介绍时会根据需要省略一部分代码**，完整代码请直接参考源码。
* LeakyReLU算子的定义可参见onnx文档：[https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu)

### 1. 添加算子参数定义与解析

若算子参数定义与解析已添加，则跳过此步骤。

#### 1.1. 添加参数定义

若算子无需参数，则跳过此步骤。

在ppl.nn/src/ppl/nn/params目录下创建\<domain_name\>/\<opname\>_param.h，用于定义参数结构体。
定义参数结构体时需要重载==运算符来支持框架上的图优化操作。

以LeakyReLU为例，其参数定义在ppl.nn/src/ppl/nn/params/onnx/leaky_relu_param.h：

```c++
struct LeakyReLUParam {
    float alpha;  // LeakyReLU仅需要一个参数alpha
    bool operator==(const LeakyReLUParam& p) const { return this->alpha == p.alpha; }   // 对==运算符进行重载
};
```

`LeakyReLUParam`定义了LeakyReLU算子所需的参数。参考onnx的LeakyReLU定义可知，该算子仅需一个float型的参数alpha。另外`LeakyReLUParam`重载了运算符==，用于判断两个参数对象是否相等。

#### 1.2. 添加参数解析函数

若算子无需参数，则跳过此步骤。

在ppl.nn/src/ppl/nn/models/onnx/parsers目录下创建parse_\<opname\>\_param.h和parse_\<opname\>\_param.cc，将参数解析函数放在其中。

以LeakyReLU为例，其参数解析函数实现在ppl.nn/src/ppl/nn/models/onnx/parsers/parse_leaky_relu_param.cc内：

```c++
ppl::common::RetCode ParseLeakyReLUParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::common::LeakyReLUParam*>(arg);
    param->alpha = utils::GetNodeAttrByKey<float>(pb_node, "alpha", 0.01);  // 从onnx中解析alpha字段，如果不存在则使用默认值0.01
    return ppl::common::RC_SUCCESS;
}
```

函数`ParseLeakyReLUParam`读入onnx node信息`pb_node`，通过`GetNodeAttrByKey`解析其中的alpha字段，放入上文定义的参数结构体中，完成参数解析。

#### 1.3. 注册参数和参数解析函数

添加好参数结构体和参数解析函数后，需要将其注册在ppl.nn/src/ppl/nn/models/onnx/param_parser_manager.cc的`ParamParserManager()`函数中。

注册时，有两个宏可以使用：

* `PPL_REGISTER_OP_WITHOUT_PARAM`：用于注册无需参数的算子
* `PPL_REGISTER_OP_WITH_PARAM`：用于注册需要参数的算子

以LeakyReLU为例：

```c++
PPL_REGISTER_OP_WITH_PARAM("", "LeakyRelu", ppl::nn::common::LeakyReLUParam, ParseLeakyReLUParam);
```

其中第一个参数为domain_name，必须跟onnx模型中的domain保持一致；第二个参数"LeakyRelu"为op_type，必须跟onnx模型中的op_type保持一致；第三、四个参数分别为上文定义的参数结构体和解析函数。

### 2. 添加算子定义

在ppl.nn/src/ppl/nn/engines/x86/optimizer/\<domain_name\>目录下添加\<opname\>\_op.h和\<opname\>\_op.cc，用于定义和实现算子定义。

例如LeakyReLU，其算子定义类放在ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.h内：

```c++
class LeakyReluOp final : public X86OptKernel {
public:
    LeakyReluOp(const ir::Node* node)
        : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;    // 初始化(必需)
    KernelImpl* CreateKernelImpl() const override;                          // 创建调用接口(必需)
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,          // 排布选择(可选)
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;

private:
    std::shared_ptr<ppl::common::LeakyReLUParam> param_;    // 参数结构体指针(可选，若算子无需参数则无需此变量)
};
```

* `Init`(必需): 进行一些初始化操作，如加载参数、注册数据类型推断函数、注册维度计算函数等
* `CreateKernelImpl`(必需): 创建算子调用接口对象
* `SelectFormat`(可选): 进行数据排布选择
* `param_`(可选): 上文定义的参数结构体指针，若算子无需参数则无需此变量

可根据实际需要，添加或删除部分成员函数和成员变量。

#### 2.1. 注册数据类型推断函数

数据类型推断函数用于根据输入的数据类型，推断出输出的数据类型。

需要在`Init`函数中将数据类型推断函数注册到`infer_type_func_`。
`infer_type_func_`是一个std::function对象，输入InputOutputInfo*，返回void。
可以用函数、lambda表达式来定义数据类型推断函数，再将其赋值给`infer_type_func_`即可完成注册。

例如LeakyReLU，其在ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc的Init函数中注册了数据类型推断函数：

```c++
infer_type_func_ = GenericInferType;
```

`GenericInferType`是一个框架预定义的数据类型推断函数，其代码在ppl.nn/src/ppl/nn/engines/x86/optimizer/opt_kernel.h：

```c++
static void GenericInferType(InputOutputInfo* info) {   // 所有输出的数据类型和第一个输入的数据类型保持一致
    auto& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
        auto out_shape = &info->GetOutput<TensorImpl>(i)->GetShape();
        out_shape->SetDataType(in_shape0.GetDataType());
    }
}
```

可根据自定义算子的需要，注册预定义的数据类型推断函数或自定义的函数。

#### 2.2. 注册维度计算函数

维度计算函数用于根据输入的数据维度，推断出输出的数据维度。

需要在`Init`函数中将维度计算函数注册到`infer_dims_func_`。
`infer_dims_func_`是一个std::function对象，输入InputOutputInfo*，返回ppl::common::RetCode。
可以用函数、lambda表达式来定义维度计算函数，再将其赋值给`infer_dims_func_`即可完成注册。

例如LeakyReLU，其在ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc的Init函数中注册了维度计算函数：

```c++
infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
    return oputils::ReshapeLeakyReLU(info, nullptr);
};
```

`ReshapeLeakyReLU`代码在ppl.nn/src/ppl/nn/oputils/onnx/reshape_leaky_relu.cc中：

```c++
RetCode ReshapeLeakyReLU(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }
    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    auto out_shape0 = &info->GetOutput<TensorImpl>(0)->GetShape();
    if (in_shape0.IsScalar()) {
        out_shape0->ReshapeAsScalar();
    } else {
        out_shape0->Reshape(in_shape0.GetDims(), in_shape0.GetDimCount());
    }
    return RC_SUCCESS;
}
```

在注册自定义算子维度计算函数时，既可以像LeakyReLU一样在ppl.nn/src/ppl/nn/oputils下写单独的Reshape函数，也可以直接在Init函数中写完所有逻辑。

#### 2.3. 编写数据排布选择函数

数据排布选择函数`SelectFormat`根据参数、输入的数据类型、排布、维度、算子底层支持的排布等信息，选择该算子**需要**的输入数据排布，以及输出的数据排布。

算子的输入排布和算子***需要***的输入排布可以不同。算子的输入排布是指在本算子进行排布选择之前，输入数据的真实排布(通常是由上一个算子的输出或网络的输入决定)；而算子***需要***的输入排布是指算子根据参数、输入的数据类型、排布、维度、算子底层支持的排布等信息，选择出该算子所需要的输入排布。
当两者不同时，框架会自动插入一个排布转换算子，用于转换不同的数据排布。

目前x86架构支持的数据排布有：
* NDARRAY (对于4维来说，即为NCHW)
* N16CX (对于4维来说，即为N16CHW，或称为N16CHWc16、NCHWc16)

以LeakyReLU为例，其排布选择函数在ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc内：

```c++
RetCode LeakyReluOp::SelectFormat(const InputOutputInfo& info,
                                  vector<dataformat_t>* selected_input_formats, // 需要的输入数据排布，默认全为NDARRAY
                                  vector<dataformat_t>* selected_output_formats // 输出数据排布，默认全为NDARRAY
    ) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) { // 算子输入的数据排布
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}
```

selected_input_formats是算子需要的输入数据排布，selected_output_formats是算子输出的数据排布，默认值全为NDARRAY。
由于LeakyReLU实现了NDARRAY和N16CX两个排布的版本，因此当算子的输入排布为N16CX时，算子需要的输入排布和输出排布选择N16CX；当输入为NDARRAY格式时，需要的输入排布和输出排布选择默认值NDARRAY，函数不做任何动作。

在添加自定义算子时，如果算子仅实现了NDARRAY版本(输入输出仅支持NDARRAY排布)，`SelectFormat`函数可以不写，这时会使用基类的默认排布选择函数，需要的输入和输出排布会选择NDARRAY。

#### 2.4. 添加CreateKernelImpl

`CreateKernelImpl`函数用于创建算子调用接口，根据算子是否需要参数，可使用两种函数：

* `CreateKernelImplWithoutParam`：用于无需参数的算子
* `CreateKernelImplWithParam`：用于需要参数的算子，需要传入参数结构体的指针

LeakyReLU使用的是带参数的版本，其实现在ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc：

```c++
KernelImpl* LeakyReluOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LeakyReluKernel>(param_.get());    // 需要传入参数结构体的指针
}
```

#### 2.5. 注册算子定义

完成算子定义后，需要使用宏`REGISTER_OPT_KERNEL_CREATOR`将其注册在ppl.nn/src/ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.cc的`OptKernelCreatorManager()`函数中。

以LeakyReLU为例：

```c++
REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", LeakyReluOp);
```

第一个参数为domain；第二个参数为op_type；第三个参数为上文定义的算子定义类的名称。

### 3. 添加算子调用接口

在ppl.nn/src/ppl/nn/engines/x86/kernels/\<domain_name\>目录下添加\<opname\>\_kernel.h和\<opname\>\_kernel.cc，用于定义和实现算子调用接口。

例如LeakyReLU，其算子调用接口定义在ppl.nn/src/ppl/nn/engines/x86/kernels/onnx/leaky_relu_kernel.h：

```c++
class LeakyReluKernel : public X86Kernel {
public:
    LeakyReluKernel(const ir::Node* node) : X86Kernel(node) {}
    void SetParam(const ppl::nn::common::LeakyReLUParam* p) { param_ = p; } // 若算子无参数则无需此函数
private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;    // 算子调用函数
private:
    const ppl::nn::common::LeakyReLUParam* param_ = nullptr;    // 若算子无参数则无需此函数
};
```

若算子无参数的话，`SetParam`和`param_`无需添加，只需构造函数和`DoExecute`函数即可。

#### 3.1. 添加DoExecute

`DoExecute`函数读入算子输入，调用kernel函数，得到算子输出。

以LeakyReLU为例，其`DoExecute`函数在ppl.nn/src/ppl/nn/engines/x86/kernels/onnx/leaky_relu_kernel.cc：

```c++
ppl::common::RetCode LeakyReluKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);  // 输入tensor
    auto y = ctx->GetOutput<TensorImpl>(0); // 输出tensor

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());   // 输出调试信息，仅在Debug模式下生效
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {   // 数据类型判断
        if (MayUseISA(ppl::common::ISA_X86_AVX)) {  // 判断是否支持avx指令集
            return kernel::x86::leaky_relu_fp32_avx(&y->GetShape(), x->GetBufferPtr<float>(),   // avx kernel函数
                                                    param_->alpha, y->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {   // 判断是否支持sse指令集
            return kernel::x86::leaky_relu_fp32_sse(&y->GetShape(), x->GetBufferPtr<float>(),   // sse kernel函数
                                                    param_->alpha, y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}
```

宏`PPLNN_X86_DEBUG_TRACE`和`PPL_X86_TENSOR_PRINT_DEBUG_MSG`用于输出调试信息，仅在编译Debug模式下生效。

`MayUseISA`函数用于判断当前环境是否可以执行指定的ISA，从而调用不同的kernel函数。
常用的ISA有：AVX512、FMA、AVX、SSE。当使用标量代码时，可不做此判断。

#### 3.2. 添加CanDoExecute

`CanDoExecute`函数执行在`DoExecute`之前，用于判断是否可以执行`DoExecute`。

大部分算子使用基类的实现(ppl.nn/src/ppl/nn/engines/x86/kernel.cc)：

```c++
bool X86Kernel::CanDoExecute(const KernelExecContext& ctx) const {  // 如果输入中存在空tensor，则返回false，否则返回true
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape().GetBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}
```

绝大多数情况不需要重载此函数。如果需要使用跟基类不同的行为，则需要重载此函数。

### 4. 添加kernel函数

x86的kernel函数是最底层的计算函数，放在ppl.nn/src/ppl/nn/engines/x86/impls目录下。

由于kernel函数跟上层框架间的耦合度较低，因此可根据自定义算子的特点，自由的安排代码结构。这里仅给出通用的编写kernel函数的规范参考，可不必严格按照本章的方式编写。

#### 4.1. kernel函数声明

kernel函数的接口声明统一放在ppl.nn/src/ppl/nn/engines/x86/impls/include/ppl/kernel/x86目录下，按照数据类型放在不同的子目录下。建议的的文件路径为ppl.nn/src/ppl/nn/engines/x86/impls/include/ppl/kernel/x86/\<data_type\>/\<opname\>.h

函数输入参数可根据需要自行定义，返回一个ppl::common::RetCode用于指示函数是否执行成功。

函数命名建议：\<opname\>\_\<data_format\>\_\<特化描述\>\_\<data_type\>\_\<isa_type\>

例如LeakyReLU的fp32 kernel函数接口声明在ppl.nn/src/ppl/nn/engines/x86/impls/include/ppl/kernel/x86/fp32/leaky_relu.h下，其函数命名为`leaky_relu_fp32_avx`和`leaky_relu_fp32_sse`：

```c++
ppl::common::RetCode leaky_relu_fp32_avx(   // avx kernel函数声明
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);

ppl::common::RetCode leaky_relu_fp32_sse(   // sse kernel函数声明
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);
```

这里不使用\<data_format\>字段是因为LeakyReLU的实现支持任意数据格式。

#### 4.2. Kernel函数实现

kernel函数的实现放在ppl.nn/src/ppl/nn/engines/x86/impls/src/ppl/kernel/x86目录下，按照数据类型放在不同的子目录下。由于kernel函数的实现可能需要多个文件，因此建议每个算子单独建立一个目录。
不同ISA架构的代码应该放在不同的.cpp文件内，并在文件命名时以ISA架构名来区分。

建议的文件路径为ppl.nn/src/ppl/nn/engines/x86/impls/src/ppl/kernel/x86/\<data_type\>/\<opname\>/\<opname\>\_\<data_type\>\_<isa_type>.cpp。这种文件命名方式会被ppl.nn/src/ppl/nn/engines/x86/impls/CMakeLists.txt所识别并自动加上对应ISA的编译指令。

以LeakyReLU为例，该算子实现了fp32的avx和sse版本，因此在ppl.nn/src/ppl/nn/engines/x86/impls/src/ppl/kernel/x86/fp32/leaky_relu目录下有leaky_relu_fp32_avx.cpp和leaky_relu_fp32_sse.cpp两个文件，分别实现了`leaky_relu_fp32_avx`和`leaky_relu_fp32_sse`函数。

### 5. 几点说明

#### 5.1. 编译

添加自定义算子后，由于有新增的.cpp文件，需要将CMakeCache.txt删除后重新运行cmake，否则会提示找不到符号的问题。

若kernel的文件命名格式不能被ppl.nn/src/ppl/nn/engines/x86/impls/CMakeLists.txt识别，则会报ISA指令不支持的错误。

#### 5.2. 其他的参考例子

添加的自定义算子有可能会跟本章中的例子LeakyReLU有所不同，因此这里给出更多的参考算子，这些算子的代码结构和本章的LeakyReLU基本一致，可根据自定义算子的情况来选择合适的参考：

| 参考算子          | 有参数  | 支持NDARRAY | 支持N16CX | 支持多ISA | 在输入有空tensor时依然应当运行 | 需要离线处理weight |
|-------------------|---------|-------------|-----------|-----------|--------------------------------|--------------------|
| Exp/Tanh          |         | &check;     | &check;   | &check;   |                                |                    |
| LeakyReLU/Softmax | &check; | &check;     | &check;   | &check;   |                                |                    |
| Tile/TopK         | &check; | &check;     |           |           |                                |                    |
| Clip/Resize       | &check; | &check;     | &check;   | &check;   | &check;                        |                    |
| Conv/FC           | &check; | &check;     | &check;   | &check;   |                                | &check;            |
