Most `PPLNN` supported ops are based on onnx opset 11. If you are using onnx model with different opset version, you need to convert your onnx model opset version to 11.

## Use ONNX Official Opset Convert Tool

`ONNX` officially provided an opset convert tool `version_converter`. Its tutorials is at: [Version Conversion](https://github.com/onnx/tutorials/blob/master/tutorials/VersionConversion.md). Please update to onnx v1.11(or above) and try `version_converter`:

```Python
import onnx
from onnx import version_converter

model = onnx.load("<your_path_to_onnx_model>")
converted_model = version_converter.convert_version(onnx_model, 11)
onnx.save(converted_model, "<your_save_path>")
```
