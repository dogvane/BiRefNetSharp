# BiRefNetSharp

.NET SDK implementation of BiRefNet, an image segmentation inference library based on ONNX Runtime.

## Quick Start

### 1. Download Model

Visit [ModelScope - BiRefNet-ONNX](https://modelscope.cn/models/onnx-community/BiRefNet-ONNX) to download `model_fp16.onnx` and place it in the `models/onnx/` directory.

**Parameter Description:**

| Parameter | Default | Description |
|------|--------|------|
| `--onnx_path` | ../models/onnx/model_fp16.onnx | ONNX model file path |
| `--input_dir` | ./ | Input images directory |
| `--output_dir` | ../output | Output directory |
| `--device` | cpu | Runtime device (you can install GPU packages such as Microsoft.ML.OnnxRuntime.Gpu) |
| `--keep_tree` | false | Whether to preserve subfolder structure from input directory |
| `--threshold` | 0.1 | Mask threshold in range (0, 1] for foreground/background separation |

### As a Library

```csharp
using Dogvane.BiRefNet;

// Create inference instance
using var engine = new BiRefNetInference("path/to/model.onnx", "cpu");

// Run inference on a single image
var (mask, width, height) = engine.Infer("path/to/image.jpg");

// mask is a uint8 array with range 0-255
// width and height are original image dimensions
```

### Example Application

The program generates two types of files in the output directory:

1. **Mask Image**: Grayscale PNG image with same filename as original (white=foreground, black=background)
2. **Masked Result**: Adds `_masked` suffix to filename, background replaced with white, foreground kept original

```
input/
└── photo.jpg

output/
├── photo.png          # Grayscale mask
└── photo_masked.png   # Masked result (white background)
```

## Dependencies

- **.NET 6.0+**
- **Microsoft.ML.OnnxRuntime** - ONNX model inference
- **SixLabors.ImageSharp** - Image processing
- **CommandLineParser** - Command line argument parsing

## License

This project is licensed under the MIT License.

## Related Links

- [BiRefNet Paper](https://arxiv.org/abs/2401.03407)
- [BiRefNet GitHub Repository](https://github.com/ZhengPeng7/BiRefNet)
- [ModelScope - BiRefNet-ONNX](https://modelscope.cn/models/onnx-community/BiRefNet-ONNX)

## Author

Dogvane

## Acknowledgments

Thanks to the original BiRefNet authors for providing an excellent model.
