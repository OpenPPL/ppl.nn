## 自定义算子添加

添加自定义算子需要先添加算子定义及其在不同架构上的实现。本章仅介绍在ARM架构上添加算子的实现细节。

### 0. 概述

PPLNN在ARM架构上添加自定义算子的步骤如下：

1) 添加算子参数定义与解析(如果算子无需参数，或参数定义与解析已添加，则跳过此步骤)
2) 添加算子定义，包括数据类型推断、维度计算、数据类型选择、数据排布选择等
3) 添加算子调用接口
4) 添加kernel函数

几点约定：

* 本章中自定义算子的名称将以\<opname\>来代替。
* 本章将以框架中已实现的LeakyReLU为例，来帮助理解添加自定义算子的流程。**在介绍时会根据需要省略一部分代码**，完整代码请直接参考源码。
* LeakyReLU算子的定义可参见onnx文档：[https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyReLU](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyReLU)

### 1. 添加算子参数定义与解析

算子参数的添加和解析可以参考[x86自定义算子添加](../x86-doc/add_op.md)的对应章节。

若算子参数定义与解析已添加，则跳过此步骤。

### 2. 添加算子定义

在ppl.nn/src/ppl/nn/engines/arm/optimizer/\<domain_name\>目录下添加\<opname\>\_op.h和\<opname\>\_op.cc，用于定义和实现算子定义。

例如LeakyReLU，其算子定义类放在ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.h内：

```c++
class LeakyReLUOp final : public ArmOptKernel {
public:
    LeakyReLUOp(const ir::Node* node)
        : ArmOptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;    // 初始化(必需)
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,        // 数据类型选择（可选）
                                        std::vector<ppl::common::datatype_t>* selected_input_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_types,
                                        const ppl::common::datatype_t preferred_fp_datatype) override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,          // 数据排布选择(可选)
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    KernelImpl* CreateKernelImpl() const override;                          // 创建调用接口(必需)

private:
    std::shared_ptr<ppl::common::LeakyReLUParam> param_;    // 参数结构体指针(可选，若算子无需参数则无需此变量)
};
```

* `Init`(必需): 进行一些初始化操作，如加载参数、注册数据类型推断函数、注册维度计算函数等
* `CreateKernelImpl`(必需): 创建算子调用接口对象
* `SelectFormat`(可选): 进行数据排布选择
* `SelectDataType`(可选): 进行数据类型选择
* `param_`(可选): 上文定义的参数结构体指针，若算子无需参数则无需此变量

可根据实际需要，添加或删除部分成员函数和成员变量。

#### 2.1. 注册数据类型推断函数

数据类型推断函数用于根据输入的数据类型，推断出输出的数据类型。

需要在`Init`函数中将数据类型推断函数注册到`infer_type_func_`。
`infer_type_func_`是一个std::function对象，输入InputOutputInfo*，返回void。
可以用函数、lambda表达式来定义数据类型推断函数，再将其赋值给`infer_type_func_`即可完成注册。

例如LeakyReLU，其在ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc的Init函数中注册了数据类型推断函数：

```c++
infer_type_func_ = GenericInferType;
```

`GenericInferType`是一个框架预定义的数据类型推断函数，其代码在ppl.nn/src/ppl/nn/engines/arm/optimizer/opt_kernel.h：

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

例如LeakyReLU，其在ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc的Init函数中注册了维度计算函数：

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

算子的输入排布和算子**需要**的输入排布可以不同。算子的输入排布是指在本算子进行排布选择之前，输入数据的真实排布(通常是由上一个算子的输出或网络的输入决定)；而算子**需要**的输入排布是指算子根据参数、输入的数据类型、排布、维度、算子底层支持的排布等信息，选择出该算子所需要的输入排布。
当两者不同时，框架会自动插入一个排布转换算子，用于转换不同的数据排布。

目前ARM架构支持的数据排布有：
* NDARRAY (对于4维来说，即为NCHW)
* N4CX (对于4维来说，即为N4CHW，或称为N4CHWc4、NCHWc4): 仅在 fp32 精度下使用
* N8CX (对于4维来说，即为N8CHW，或称为N8CHWc8、NCHWc8): 仅在 fp16 精度下使用

以LeakyReLU为例，其排布选择函数在ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc内：

```c++
RetCode LeakyReLUOp::SelectFormat(const InputOutputInfo& info,
                                  std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                  std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = selected_output_formats->at(0) = // 算子所选择的数据排布跟输入算子的数据排布一致
        info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    return RC_SUCCESS;
}
```

其中，`selected_input_formats`是算子需要的输入数据排布，`selected_output_formats`是算子输出的数据排布。

在添加自定义算子时，如果算子仅实现了NDARRAY版本(输入输出仅支持NDARRAY排布)，`SelectFormat`函数可以不写，这时会使用基类的默认排布选择函数，需要的输入和输出排布会选择NDARRAY。

### 2.4. 编写数据类型选择函数

数据类型选择函数`SelectDataType`跟上述排布选择类似，根据参数、输入的数据类型、排布、维度、算子底层支持的数据类型等信息，选择该算子**需要**的输入数据类型，以及输出的数据类型。
当输入算子的数据类型与算子**需要**的数据类型不同时，框架会自动插入一个数据类型转换算子，用于转换不同的数据数据类型。

以LeakyReLU为例，其数据类型选择函数在ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc内：

```c++
RetCode LeakyReLUOp::SelectDataType(const InputOutputInfo& info,
                                    std::vector<ppl::common::datatype_t>* selected_input_types,
                                    std::vector<ppl::common::datatype_t>* selected_output_types,
                                    const ppl::common::datatype_t preferred_fp_datatype) { // 偏好的浮点精度类型，目前仅支持 fp16 和 fp32
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype); // 通用的数据类型选择函数，定义在 ppl.nn/src/ppl/nn/engines/arm/optimizer/opt_kernel.h
    return RC_SUCCESS;
}
```

#### 2.5. 添加CreateKernelImpl

`CreateKernelImpl`函数用于创建算子调用接口，根据算子是否需要参数，可使用两种函数：

* `CreateKernelImplWithoutParam`：用于无需参数的算子
* `CreateKernelImplWithParam`：用于需要参数的算子，需要传入参数结构体的指针

LeakyReLU使用的是带参数的版本，其实现在ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc：

```c++
KernelImpl* LeakyReLUOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LeakyReLUKernel>(param_.get());    // 需要传入参数结构体的指针
}
```

#### 2.6. 注册算子定义

完成算子定义后，需要使用宏`REGISTER_OPT_KERNEL_CREATOR`将其注册在ppl.nn/src/ppl/nn/engines/arm/optimizer/opt_kernel_creator_manager.cc的`OptKernelCreatorManager()`函数中。

以LeakyReLU为例：

```c++
REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", 6, 16, LeakyReLUOp);
```

第一个参数为domain；第二个参数为op_type；第3、4个参数分别为算子opset的最小、最大支持版本；最后一个参数为上文定义的算子定义类的名称。

### 3. 添加算子调用接口

在ppl.nn/src/ppl/nn/engines/arm/kernels/\<domain_name\>目录下添加\<opname\>\_kernel.h和\<opname\>\_kernel.cc，用于定义和实现算子调用接口。

例如LeakyReLU，其算子调用接口定义在ppl.nn/src/ppl/nn/engines/arm/kernels/onnx/leaky_relu_kernel.h：

```c++
class LeakyReLUKernel : public ArmKernel {
public:
    LeakyReLUKernel(const ir::Node* node) : ArmKernel(node) {}
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

以LeakyReLU为例，其`DoExecute`函数在ppl.nn/src/ppl/nn/engines/arm/kernels/onnx/leaky_relu_kernel.cc：

```c++
ppl::common::RetCode LeakyReLUKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);  // 输入tensor
    auto y = ctx->GetOutput<TensorImpl>(0); // 输出tensor

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());   // 输出调试信息，仅在Debug模式下生效
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Output [y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_ARM_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {   // fp32 实现
        if (MayUseISA(ppl::common::ISA_ARMV8)) {
            return ppl::kernel::arm_server::neon::leaky_relu_fp32(x->GetShape(), x->GetBufferPtr<float>(),
                                                                  param_->alpha, y->GetBufferPtr<float>());
        }
    } 
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (data_type == ppl::common::DATATYPE_FLOAT16) {  // fp16 实现，需要 armv8.2-a 指令集支持
        if (MayUseISA(ppl::common::ISA_ARMV8_2)) {
            return ppl::kernel::arm_server::neon::leaky_relu_fp16(x->GetShape(), x->GetBufferPtr<__fp16>(),
                                                                  param_->alpha, y->GetBufferPtr<__fp16>());
        }
    } 
#endif
    else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}
```

宏`PPLNN_ARM_DEBUG_TRACE`和`PPL_ARM_TENSOR_PRINT_DEBUG_MSG`用于输出调试信息，仅在编译Debug模式下生效。

`MayUseISA`函数用于判断当前环境是否可以执行指定的ISA。
目前PPL.NN ARM支持的ISA有：ARMV8(默认支持)、ARMV8_2(需要编译器支持armv8.2-a指令集)。

#### 3.2. 添加CanDoExecute

`CanDoExecute`函数执行在`DoExecute`之前，用于判断是否可以执行`DoExecute`。

大部分算子使用基类的实现(ppl.nn/src/ppl/nn/engines/arm/kernel.cc)：

```c++
bool ArmKernel::CanDoExecute(const KernelExecContext& ctx) const {  // 如果输入中存在空tensor，则返回false，否则返回true
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

ARM的kernel函数是最底层的计算函数，放在ppl.nn/src/ppl/nn/engines/arm/impls目录下。

由于kernel函数跟上层框架间的耦合度较低，因此可根据自定义算子的特点，自由的安排代码结构。这里仅给出通用的编写kernel函数的规范参考，可不必严格按照本章的方式编写。

#### 4.1. kernel函数声明

kernel函数的接口声明统一放在ppl.nn/src/ppl/nn/engines/arm/impls/include/ppl/kernel/arm_server目录下。建议的文件路径为ppl.nn/src/ppl/nn/engines/arm/impls/include/ppl/kernel/arm_server/\<opname\>/\<isa\>/\<opname\>.h

函数输入参数可根据需要自行定义，返回一个`ppl::common::RetCode`用于指示函数是否执行成功。

函数命名建议：\<opname\>\_\<data_format\>\_\<特化描述\>\_\<data_type\>

例如LeakyReLU的kernel函数接口声明在ppl.nn/src/ppl/nn/engines/arm/impls/include/ppl/kernel/arm_server/leaky_relu/neon/leaky_relu.h下，其函数命名为`leaky_relu_fp32`和`leaky_relu_fp16`：

```c++
ppl::common::RetCode leaky_relu_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);

#ifdef PPLNN_USE_ARMV8_2_FP16  // 需要编译器支持 armv8.2-a 指令集
ppl::common::RetCode leaky_relu_fp16(
    const ppl::nn::TensorShape *src_shape,
    const __fp16 *src,
    const float alpha,
    __fp16 *dst);
#endif
```

这里不使用\<data_format\>字段是因为LeakyReLU的实现支持任意数据格式。

#### 4.2. Kernel函数实现

kernel函数的实现放在ppl.nn/src/ppl/nn/engines/arm/impls/src/ppl/kernel/arm_server目录下，按照数据类型放在不同的子目录下。由于kernel函数的实现可能需要多个文件，因此建议每个算子单独建立一个目录。

建议的文件路径为ppl.nn/src/ppl/nn/engines/arm/impls/src/ppl/kernel/arm_server/\<opname\>/\<isa\>/\<opname\>.cpp。

以LeakyReLU为例，该算子实现了neon版本的代码，因此在ppl.nn/src/ppl/nn/engines/arm/impls/src/ppl/kernel/arm_server/leaky_relu/neon目录下有leaky_relu.cpp文件，实现了`leaky_relu_fp32`和`leaky_relu_fp16`函数。

### 5. 几点说明

#### 5.1. 编译

添加自定义算子后，由于有新增的.cpp文件，需要将CMakeCache.txt删除后重新运行cmake，否则会提示找不到符号的问题。

#### 5.2. FP16 编译宏

所有要用到 armv8.2-a 指令的代码（最典型的就是 FP16 的指令），都需要放在宏`PPLNN_USE_ARMV8_2_FP16`中，避免编译器不支持 armv8.2-a 时无法通过编译。

#### 5.3. 其他的参考例子

添加的自定义算子有可能会跟本章中的例子LeakyReLU有所不同，因此这里给出更多的参考算子，这些算子的代码结构和本章的LeakyReLU基本一致，可根据自定义算子的情况来选择合适的参考：

| 参考算子          | 有参数  | 支持NDARRAY | 支持N4CX/N8CX | 在输入有空tensor时依然应当运行 | 需要离线处理weight |
|-------------------|---------|-------------|--------------|--------------------------------|--------------------|
| Exp/Sigmoid       |         | &check;     | &check;      |                                |                    |
| LeakyReLU/Softmax | &check; | &check;     | &check;      |                                |                    |
| Tile/TopK         | &check; | &check;     |              |                                |                    |
| Clip/Resize       | &check; | &check;     | &check;      | &check;                        |                    |
| Conv/FC           | &check; | &check;     | &check;      |                                | &check;            |
