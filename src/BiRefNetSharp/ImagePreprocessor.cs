using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BiRefNetSharp;

/// <summary>
/// Provides image preprocessing functionality for BiRefNet model input.
/// </summary>
public static class ImagePreprocessor
{
    /// <summary>
    /// Default input size for the BiRefNet model.
    /// </summary>
    public const int DefaultInputSize = 1024;

    /// <summary>
    /// Mean values for image normalization (ImageNet statistics).
    /// </summary>
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };

    /// <summary>
    /// Standard deviation values for image normalization (ImageNet statistics).
    /// </summary>
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    /// <summary>
    /// Preprocesses an image for BiRefNet model input.
    /// </summary>
    /// <param name="image">Input image to preprocess.</param>
    /// <param name="targetSize">Target size for the model input (default: 1024x1024).</param>
    /// <returns>Preprocessed tensor ready for model inference.</returns>
    public static DenseTensor<float> PreprocessImage(Image<Rgb24> image, int targetSize = DefaultInputSize)
    {
        if (image == null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        if (targetSize <= 0)
        {
            throw new ArgumentException("Target size must be positive.", nameof(targetSize));
        }

        // Clone and resize the image
        var resizedImage = image.Clone(ctx => ctx.Resize(targetSize, targetSize));

        // Create tensor with shape [1, 3, height, width] (NCHW format)
        var tensor = new DenseTensor<float>(new[] { 1, 3, targetSize, targetSize });

        // Process each pixel and normalize
        resizedImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < targetSize; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < targetSize; x++)
                {
                    var pixel = pixelRow[x];

                    // Normalize to [0, 1] and apply ImageNet normalization
                    tensor[0, 0, y, x] = (pixel.R / 255f - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = (pixel.G / 255f - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = (pixel.B / 255f - Mean[2]) / Std[2];
                }
            }
        });

        resizedImage.Dispose();

        return tensor;
    }

    /// <summary>
    /// Preprocesses an image from a file path.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="targetSize">Target size for the model input (default: 1024x1024).</param>
    /// <returns>Preprocessed tensor ready for model inference.</returns>
    public static DenseTensor<float> PreprocessImage(string imagePath, int targetSize = DefaultInputSize)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path cannot be null or empty.", nameof(imagePath));
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image file not found: {imagePath}", imagePath);
        }

        using var image = Image.Load<Rgb24>(imagePath);
        return PreprocessImage(image, targetSize);
    }
}
