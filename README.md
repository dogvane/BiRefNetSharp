# BiRefNetSharp

BiRefNet's ONNX inference implementation under .NET

## Overview

BiRefNetSharp is a .NET library that provides ONNX Runtime inference for BiRefNet (Bilateral Reference Network), a state-of-the-art deep learning model for image segmentation and background removal.

## Features

- 🚀 High-performance ONNX Runtime inference
- 🖼️ Image preprocessing with ImageNet normalization
- 🎨 Multiple output options: masks, transparent backgrounds
- 🔄 Synchronous and asynchronous APIs
- 🎯 Easy-to-use helper methods
- 📦 Cross-platform support (.NET 8.0)

## Requirements

- .NET 8.0 or higher
- BiRefNet ONNX model file

## Installation

### From Source

```bash
git clone https://github.com/dogvane/BiRefNetSharp.git
cd BiRefNetSharp
dotnet build
```

### NuGet Package (Coming Soon)

```bash
dotnet add package BiRefNetSharp
```

## Quick Start

### Basic Usage

```csharp
using BiRefNetSharp;

// Load the model
using var model = new BiRefNetModel("path/to/model.onnx");

// Remove background from an image
BiRefNetHelper.RemoveBackground(
    model: model,
    imagePath: "input.jpg",
    outputPath: "output.png"
);
```

### Advanced Usage

```csharp
using BiRefNetSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// Load the model
using var model = new BiRefNetModel("model.onnx");

// Load input image
using var inputImage = Image.Load<Rgb24>("input.jpg");

// Get segmentation mask
using var mask = model.Predict(inputImage);

// Apply custom post-processing
using var smoothedMask = ImagePostprocessor.SmoothMask(mask, sigma: 2.0f);
using var binaryMask = ImagePostprocessor.ApplyThreshold(smoothedMask, threshold: 128);

// Apply mask to original image
using var result = BiRefNetModel.ApplyMask(inputImage, smoothedMask);

// Save result
result.SaveAsPng("output.png");
```

### Async Operations

```csharp
using BiRefNetSharp;

// Load the model
using var model = new BiRefNetModel("model.onnx");

// Process image asynchronously
await BiRefNetHelper.RemoveBackgroundAsync(
    model: model,
    imagePath: "input.jpg",
    outputPath: "output.png",
    cancellationToken: cancellationToken
);
```

## Command Line Sample

The repository includes a command-line sample application:

```bash
cd samples/BiRefNetSharp.Sample
dotnet run -- <model_path> <image_path> [output_path]
```

Example:

```bash
dotnet run -- model.onnx input.jpg output.png
```

## API Documentation

### BiRefNetModel

Main class for running BiRefNet inference.

#### Constructor

```csharp
public BiRefNetModel(string modelPath, SessionOptions? options = null)
```

#### Methods

- `Image<L8> Predict(Image<Rgb24> inputImage)` - Run inference and get segmentation mask
- `Task<Image<L8>> PredictAsync(Image<Rgb24> inputImage, CancellationToken cancellationToken = default)` - Async version
- `static Image<Rgba32> ApplyMask(Image<Rgb24> inputImage, Image<L8> mask)` - Apply mask to image

### BiRefNetHelper

Utility methods for common operations.

#### Methods

- `RemoveBackground(...)` - Remove background from image (synchronous)
- `RemoveBackgroundAsync(...)` - Remove background from image (asynchronous)
- `GetMask(...)` - Extract segmentation mask only

### ImagePreprocessor

Image preprocessing utilities.

#### Methods

- `DenseTensor<float> PreprocessImage(Image<Rgb24> image, int targetSize = 1024)` - Preprocess image for model input
- `DenseTensor<float> PreprocessImage(string imagePath, int targetSize = 1024)` - Preprocess from file path

### ImagePostprocessor

Image postprocessing utilities.

#### Methods

- `Image<L8> PostprocessOutput(float[] output, int originalWidth, int originalHeight)` - Convert model output to mask
- `Image<L8> ApplyThreshold(Image<L8> mask, byte threshold = 128)` - Apply binary threshold
- `Image<L8> SmoothMask(Image<L8> mask, float sigma = 2.0f)` - Smooth mask with Gaussian blur

## Model Information

BiRefNetSharp expects an ONNX model exported from BiRefNet. The model should:

- Accept input tensor of shape `[1, 3, H, W]` (NCHW format)
- Use ImageNet normalization (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
- Output tensor of shape `[1, H, W]` or `[H*W]` containing segmentation logits

Default input size is 1024x1024, but this can be configured.

## Project Structure

```
BiRefNetSharp/
├── src/
│   └── BiRefNetSharp/           # Main library
│       ├── BiRefNetModel.cs     # Core model class
│       ├── BiRefNetHelper.cs    # Helper utilities
│       ├── ImagePreprocessor.cs # Preprocessing
│       └── ImagePostprocessor.cs # Postprocessing
├── samples/
│   └── BiRefNetSharp.Sample/    # Command-line sample
└── BiRefNetSharp.sln            # Solution file
```

## Dependencies

- [Microsoft.ML.OnnxRuntime](https://github.com/microsoft/onnxruntime) - ONNX Runtime for .NET
- [SixLabors.ImageSharp](https://github.com/SixLabors/ImageSharp) - Cross-platform image processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BiRefNet model by [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- ONNX Runtime by Microsoft
- ImageSharp by Six Labors

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/dogvane/BiRefNetSharp).

