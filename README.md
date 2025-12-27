# BiRefNetSharp

BiRefNet 的 .NET SDK 实现，基于 ONNX Runtime 的图像分割推理库。

[English Documentation](README_en.md)

## 快速开始

### 1. 下载模型

访问 [ModelScope - BiRefNet-ONNX](https://modelscope.cn/models/onnx-community/BiRefNet-ONNX) 下载 `model_fp16.onnx`，放置在 `models/onnx/` 目录下。

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--onnx_path` | ../models/onnx/model_fp16.onnx | ONNX 模型文件路径 |
| `--input_dir` | ./ | 输入图片目录 |
| `--output_dir` | ../output | 输出目录 |
| `--device` | cpu | 运行设备（可安装 GPU 对应的包，如 Microsoft.ML.OnnxRuntime.Gpu） |
| `--keep_tree` | false | 是否保留输入目录的子文件夹结构 |
| `--threshold` | 0.1 | Mask 阈值，范围 (0, 1]，用于区分前景和背景 |

### 作为类库使用

```csharp
using Dogvane.BiRefNet;

// 创建推理实例
using var engine = new BiRefNetInference("path/to/model.onnx", "cpu");

// 对单张图片进行推理
var (mask, width, height) = engine.Infer("path/to/image.jpg");

// mask 为 uint8 数组，范围为 0-255
// width 和 height 为原始图片的尺寸
```

### Example 应用程序

程序会在输出目录生成两类文件：

1. **掩码图**：灰度 PNG 图片，文件名与原图相同（白色=前景，黑色=背景）
2. **抠图结果**：在文件名后添加 `_masked` 后缀，背景替换为白色，前景保持原色

```
input/
└── photo.jpg

output/
├── photo.png          # 灰度掩码图
└── photo_masked.png   # 抠图结果（白色背景）
```

## 依赖项

- **.NET 6.0+**
- **Microsoft.ML.OnnxRuntime** - ONNX 模型推理
- **SixLabors.ImageSharp** - 图像处理
- **CommandLineParser** - 命令行参数解析

## 许可证

本项目采用 MIT 许可证。

## 相关链接

- [BiRefNet 论文](https://arxiv.org/abs/2401.03407)
- [BiRefNet GitHub 仓库](https://github.com/ZhengPeng7/BiRefNet)
- [ModelScope - BiRefNet-ONNX](https://modelscope.cn/models/onnx-community/BiRefNet-ONNX)

## 作者

Dogvane

## 致谢

感谢 BiRefNet 原作者提供的优秀模型。
