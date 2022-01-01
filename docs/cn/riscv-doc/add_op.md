## 自定义算子添加
添加自定义算子需要先添加算子定义及其在不同架构上的实现。本章仅介绍在riscv架构上添加算子的实现细节。

### 0. 概述
PPLNN在riscv架构上添加自定义算子的步骤如下：
1. 添加算子参数定义与解析
1. 添加算子定义，包括数据类型推断，维度计算、数据排布等
2. 添加算子调用接口
3. 添加kernel函数

### 1. 添加算子参数定义与解析
算子参数的添加和解析可以参考[x86自定义算子添加](../x86-doc/add_op.md)的对应章节。

### 2. 添加算子定义

在ppl.nn/src/ppl/nn/engines/riscv/optimizer/ops/onnx目录下添加<opname\>_op.h和<opname\>_op.cc，用于定义和实现算子。

以Clip为例，其算子定义类在ppl.nn/src/ppl/nn/engines/riscv/optimizer/ops/onnx/clip_op.h：
``` c++
class ClipOp final : public RiscvOptKernel {
public:
    ClipOp(const ir::Node* node) : RiscvOptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                        std::vector<ppl::common::datatype_t>* selected_input_data_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_data_types) override;
};
```

#### 2.1 注册维度计算函数

维度计算函数用于根据输入的数据维度，推断出输出的数据维度。

需要在`Init`函数中将维度计算函数注册到`infer_dims_func_`。 `infer_dims_func_`是一个std::function对象，输入InputOutputInfo*，返回ppl::common::RetCode。 可以用函数、lambda表达式来定义维度计算函数，再将其赋值给`infer_dims_func_`即可完成注册。

Clip使用了一个默认的维度计算函数`GenericInferDims`，该函数的具体实现可以在src/ppl/nn/engines/riscv/optimizer/opt_kernel.h中找到。对于不同的算子需求，可以在lambda表达式中自定义计算逻辑进行注册。

#### 2.2 编写数据类型选择函数

数据类型选择函数`SelectFormat`根据参数、输入的数据类型、排布、维度、算子底层支持的类型等信息，选择该算子需要的输入数据类型，以及输出的数据类型。

算子的输入类型和算子需要的输入类型可以不同。算子的输入类型是指在本算子进行类型选择之前，输入数据的真实类型(通常是由上一个算子的输出或网络的输入决定)；而算子需要的输入类型是指算子根据参数、输入的数据类型、排布、维度、算子底层支持的类型等信息，选择出该算子所需要的输入类型。 当两者不同时，框架会自动插入一个类型转换算子，用于转换不同的数据类型。

以Clip为例，具体的使用方法如下：
``` c++
RetCode ClipOp::SelectDataType(const InputOutputInfo& info,
                               std::vector<datatype_t>* selected_input_data_types,
                               std::vector<datatype_t>* selected_output_data_types) {

    if (DATATYPE_FLOAT16 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT16;
    } else if (DATATYPE_FLOAT32 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT32;
    }
    return RC_SUCCESS;
}
```

selected_input_data_types的传入值包含了算子的输入数据类型，selected_output_data_types的传入值默认都是FP32。在`SelectDataType`函数中把算子需要的输入数据类型和算子需要的输出数据类型分别写入selected_input_data_types和selected_output_data_types中。

由于支持混合精度的需要，在riscv架构中没有使用数据类型推断函数（与维度计算函数类似，在`Init`函数中定义并赋值给`infer_type_func_`），但仍需在`Init`函数中将默认的数据类型推断函数`GenericInferType`赋值给`infer_type_func_`。

#### 2.3 编写数据排布选择函数

数据排布选择函数SelectFormat根据参数、输入的数据类型、排布、维度、算子底层支持的排布等信息，选择该算子需要的输入数据排布，以及输出的数据排布。

与算子的数据类型类似，算子的输入排布和算子需要的输入排布可以不同。算子的输入排布是指在本算子进行排布选择之前，输入数据的真实排布(通常是由上一个算子的输出或网络的输入决定)；而算子需要的输入排布是指算子根据参数、输入的数据类型、排布、维度、算子底层支持的排布等信息，选择出该算子所需要的输入排布。 当两者不同时，框架会自动插入一个排布转换算子，用于转换不同的数据排布。

目前riscv架构支持的数据排布有：NDARRAY, N4CX, N8CX。

以Clip为例，具体的使用方法如下：
``` c++
RetCode ClipOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (DATAFORMAT_N8CX == selected_input_formats->at(0)) {
        selected_output_formats->at(0) = DATAFORMAT_N8CX;
    } else if(DATAFORMAT_N4CX == selected_input_formats->at(0)) {
        selected_output_formats->at(0) = DATAFORMAT_N4CX;
    }

    return RC_SUCCESS;
}
```
selected_input_formats的传入值包含了算子的输入数据排布，selected_output_formats的传入值默认都是NDARRAY。在`SelectFormat`函数中把算子需要的输入数据排布和算子需要的输出数据排布分别写入selected_input_formats和selected_output_formats中。

#### 2.4 添加CreaeteKernelImpl

`CreateKernelImpl`函数用于创建算子调用接口，根据算子是否需要参数，可使用两种函数：
`CreateKernelImplWithoutParam`：用于无需参数的算子
`CreateKernelImplWithParam`：用于需要参数的算子，需要传入参数结构体的指针。

Clip是无需参数的算子，实现如下：
``` c++
KernelImpl* TestOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<TestKernel>();
}
```

#### 2.5 注册算子定义

完成算子定义后，需要使用宏`REGISTER_OPT_KERNEL_CREATOR`将其注册在ppl.nn/src/ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.cc的`OptKernelCreatorManager()`函数中。

Clip的算子注册如下：
``` c++
REGISTER_OPT_KERNEL_CREATOR("", "Test", 7, 11, TestOp);
```

第一个参数为domain； 第二个参数为op_type； 第三和第四个参数表示该op支持的opset范围，比如在这个例子中TestOp支持opset(7)~opset(11)；第四个参数为上文定义的算子定义类的名称。

### 3. 添加算子接口调用

在ppl.nn/src/ppl/nn/engines/riscv/kernels/onnx目录下添加<opname\>_kernel.h和<opname\>_kernel.cc，用于定义和实现算子调用接口。

Clip的算子调用接口定义如下：
``` c++
class ClipKernel : public RiscvKernel {
public:
    ClipKernel(const ir::Node* node) : RiscvKernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;
};
```

若算子无参数的话，`SetParam`和`param_`无需添加，只需构造函数和`DoExecute`函数即可。`DoExecute`函数可以从参数中拿到算子的输入输出信息，用于调用自定义的kernel函数，将结果写入输出tensor。

`CanDoExecute`函数执行在`DoExecute`之前，用于判断是否可以执行`DoExecute`。绝大多数情况不需要重载此函数。如果需要使用跟基类不同的行为，则需要重载此函数。

### 4. 添加kernel函数

kernel函数是实现计算的函数，放在ppl.nn/src/ppl/nn/engines/riscv/impls目录下。

由于kernel函数跟上层框架间的耦合度较低，因此可根据自定义算子的特点，自由地安排代码结构。这里仅给出通用的编写kernel函数的规范参考，可不必严格按照本章的方式编写。

#### 4.1 kernel函数声明

kernel函数的接口声明统一放在ppl.nn/src/ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv目录下，按照数据类型放在不同的子目录下。建议的路径为ppl.nn/src/ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv/<data_type\>/<opname\>.h。

#### 4.2 kernel函数实现

kernel函数的实现放在ppl.nn/src/ppl/nn/engines/riscv/impls/src/ppl/kernel/riscv目录下，按照数据类型放在不同的子目录下。由于kernel函数的实现可能需要多个文件，因此建议每个算子单独建立一个目录。建议的文件路径为ppl.nn/src/ppl/nn/engines/riscv/impls/src/ppl/kernel/riscv/<data_type>/<opname\>/<opname\>_<data_type\>.cpp。