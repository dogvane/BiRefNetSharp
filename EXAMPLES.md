# BiRefNetSharp Examples

This document provides detailed examples for using BiRefNetSharp.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Advanced Usage](#advanced-usage)
3. [Async Operations](#async-operations)
4. [Custom Processing](#custom-processing)
5. [Batch Processing](#batch-processing)
6. [Performance Tips](#performance-tips)

## Basic Usage

### Remove Background from Single Image

The simplest way to remove background from an image:

```csharp
using BiRefNetSharp;

// Load the model once
using var model = new BiRefNetModel("model.onnx");

// Remove background
BiRefNetHelper.RemoveBackground(
    model: model,
    imagePath: "input.jpg",
    outputPath: "output.png"
);
```

### Extract Segmentation Mask Only

If you only need the mask without applying it:

```csharp
using BiRefNetSharp;

using var model = new BiRefNetModel("model.onnx");

BiRefNetHelper.GetMask(
    model: model,
    imagePath: "input.jpg",
    outputPath: "mask.png"
);
```

## Advanced Usage

### Manual Control Over Each Step

```csharp
using BiRefNetSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// Load model
using var model = new BiRefNetModel("model.onnx");

// Load image
using var inputImage = Image.Load<Rgb24>("input.jpg");

// Run inference
using var mask = model.Predict(inputImage);

// Optional: Apply post-processing
using var smoothedMask = ImagePostprocessor.SmoothMask(mask, sigma: 2.5f);

// Optional: Create binary mask
using var binaryMask = ImagePostprocessor.ApplyThreshold(smoothedMask, threshold: 128);

// Apply mask to original image
using var result = BiRefNetModel.ApplyMask(inputImage, smoothedMask);

// Save result
result.SaveAsPng("output.png");

// Or save the mask separately
smoothedMask.SaveAsPng("mask.png");
```

### Custom ONNX Session Options

```csharp
using BiRefNetSharp;
using Microsoft.ML.OnnxRuntime;

// Configure ONNX Runtime options
var options = new SessionOptions
{
    // Use GPU if available
    // ExecutionMode = ExecutionMode.ORT_PARALLEL,
    // GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
};

using var model = new BiRefNetModel("model.onnx", options);

// Use the model as normal
```

### Custom Input Size

```csharp
using BiRefNetSharp;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// Preprocess with custom size
using var image = Image.Load<Rgb24>("input.jpg");
var tensor = ImagePreprocessor.PreprocessImage(image, targetSize: 512);

// Note: You'll need to run inference manually with this approach
```

## Async Operations

### Async Background Removal

For responsive applications, use async methods:

```csharp
using BiRefNetSharp;

using var model = new BiRefNetModel("model.onnx");

await BiRefNetHelper.RemoveBackgroundAsync(
    model: model,
    imagePath: "input.jpg",
    outputPath: "output.png",
    cancellationToken: cancellationToken
);
```

### Async with Progress Reporting

```csharp
using BiRefNetSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using var model = new BiRefNetModel("model.onnx");

Console.WriteLine("Loading image...");
using var image = await Image.LoadAsync<Rgb24>("input.jpg");

Console.WriteLine("Running inference...");
using var mask = await model.PredictAsync(image, cancellationToken);

Console.WriteLine("Applying mask...");
using var result = BiRefNetModel.ApplyMask(image, mask);

Console.WriteLine("Saving result...");
await result.SaveAsPngAsync("output.png");

Console.WriteLine("Done!");
```

## Custom Processing

### Create Binary Mask with Threshold

```csharp
using BiRefNetSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using var model = new BiRefNetModel("model.onnx");
using var image = Image.Load<Rgb24>("input.jpg");

// Get mask
using var mask = model.Predict(image);

// Apply different thresholds
using var lowThreshold = ImagePostprocessor.ApplyThreshold(mask, 100);
using var mediumThreshold = ImagePostprocessor.ApplyThreshold(mask, 128);
using var highThreshold = ImagePostprocessor.ApplyThreshold(mask, 180);

// Save different versions
lowThreshold.SaveAsPng("mask_low.png");
mediumThreshold.SaveAsPng("mask_medium.png");
highThreshold.SaveAsPng("mask_high.png");
```

### Apply Different Smoothing Levels

```csharp
using BiRefNetSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using var model = new BiRefNetModel("model.onnx");
using var image = Image.Load<Rgb24>("input.jpg");
using var mask = model.Predict(image);

// Try different smoothing levels
using var noSmoothing = mask.Clone();
using var lightSmoothing = ImagePostprocessor.SmoothMask(mask, sigma: 1.0f);
using var mediumSmoothing = ImagePostprocessor.SmoothMask(mask, sigma: 2.0f);
using var heavySmoothing = ImagePostprocessor.SmoothMask(mask, sigma: 4.0f);

// Apply to image
using var result1 = BiRefNetModel.ApplyMask(image, noSmoothing);
using var result2 = BiRefNetModel.ApplyMask(image, lightSmoothing);
using var result3 = BiRefNetModel.ApplyMask(image, mediumSmoothing);
using var result4 = BiRefNetModel.ApplyMask(image, heavySmoothing);

result1.SaveAsPng("no_smooth.png");
result2.SaveAsPng("light_smooth.png");
result3.SaveAsPng("medium_smooth.png");
result4.SaveAsPng("heavy_smooth.png");
```

## Batch Processing

### Process Multiple Images

```csharp
using BiRefNetSharp;

// Load model once for all images
using var model = new BiRefNetModel("model.onnx");

var imageFiles = Directory.GetFiles("input", "*.jpg");

foreach (var imagePath in imageFiles)
{
    var fileName = Path.GetFileNameWithoutExtension(imagePath);
    var outputPath = Path.Combine("output", $"{fileName}_nobg.png");

    Console.WriteLine($"Processing {fileName}...");

    BiRefNetHelper.RemoveBackground(
        model: model,
        imagePath: imagePath,
        outputPath: outputPath
    );
}

Console.WriteLine($"Processed {imageFiles.Length} images");
```

### Parallel Batch Processing

```csharp
using BiRefNetSharp;
using System.Collections.Concurrent;

var imageFiles = Directory.GetFiles("input", "*.jpg");
var completedCount = 0;

// Process in parallel
Parallel.ForEach(imageFiles, new ParallelOptions { MaxDegreeOfParallelism = 4 }, imagePath =>
{
    // Each thread gets its own model instance
    using var model = new BiRefNetModel("model.onnx");

    var fileName = Path.GetFileNameWithoutExtension(imagePath);
    var outputPath = Path.Combine("output", $"{fileName}_nobg.png");

    BiRefNetHelper.RemoveBackground(
        model: model,
        imagePath: imagePath,
        outputPath: outputPath
    );

    var count = Interlocked.Increment(ref completedCount);
    Console.WriteLine($"Completed {count}/{imageFiles.Length}: {fileName}");
});

Console.WriteLine("All images processed!");
```

### Async Batch Processing with Semaphore

```csharp
using BiRefNetSharp;

var imageFiles = Directory.GetFiles("input", "*.jpg");
var maxConcurrency = 4;
var semaphore = new SemaphoreSlim(maxConcurrency);

var tasks = imageFiles.Select(async imagePath =>
{
    await semaphore.WaitAsync();
    try
    {
        using var model = new BiRefNetModel("model.onnx");

        var fileName = Path.GetFileNameWithoutExtension(imagePath);
        var outputPath = Path.Combine("output", $"{fileName}_nobg.png");

        await BiRefNetHelper.RemoveBackgroundAsync(
            model: model,
            imagePath: imagePath,
            outputPath: outputPath
        );

        Console.WriteLine($"Completed: {fileName}");
    }
    finally
    {
        semaphore.Release();
    }
});

await Task.WhenAll(tasks);
Console.WriteLine("All images processed!");
```

## Performance Tips

### 1. Reuse Model Instance

```csharp
// ❌ Bad: Creating new model for each image
foreach (var image in images)
{
    using var model = new BiRefNetModel("model.onnx"); // Slow!
    // ...
}

// ✅ Good: Reuse model instance
using var model = new BiRefNetModel("model.onnx");
foreach (var image in images)
{
    // Use same model instance
}
```

### 2. Dispose Images Properly

```csharp
// Use 'using' statements to ensure proper disposal
using var image = Image.Load<Rgb24>("input.jpg");
using var mask = model.Predict(image);
using var result = BiRefNetModel.ApplyMask(image, mask);
result.SaveAsPng("output.png");
// All resources automatically disposed here
```

### 3. Process in Memory Streams

```csharp
using BiRefNetSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using var model = new BiRefNetModel("model.onnx");

// Load from stream
using var inputStream = File.OpenRead("input.jpg");
using var image = Image.Load<Rgb24>(inputStream);

// Process
using var mask = model.Predict(image);
using var result = BiRefNetModel.ApplyMask(image, mask);

// Save to stream
using var outputStream = File.Create("output.png");
await result.SaveAsPngAsync(outputStream);
```

### 4. Optimize for GPU (if available)

```csharp
using BiRefNetSharp;
using Microsoft.ML.OnnxRuntime;

var options = new SessionOptions();
// Uncomment if you have CUDA support
// options.AppendExecutionProvider_CUDA(0);

using var model = new BiRefNetModel("model.onnx", options);
```

## Error Handling

### Robust Error Handling

```csharp
using BiRefNetSharp;

try
{
    using var model = new BiRefNetModel("model.onnx");

    BiRefNetHelper.RemoveBackground(
        model: model,
        imagePath: "input.jpg",
        outputPath: "output.png"
    );

    Console.WriteLine("Success!");
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"File not found: {ex.Message}");
}
catch (ArgumentException ex)
{
    Console.WriteLine($"Invalid argument: {ex.Message}");
}
catch (Exception ex)
{
    Console.WriteLine($"Unexpected error: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
}
```
