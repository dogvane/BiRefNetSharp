using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BiRefNetSharp;

/// <summary>
/// Provides image postprocessing functionality for BiRefNet model output.
/// </summary>
public static class ImagePostprocessor
{
    /// <summary>
    /// Postprocesses the model output to create a segmentation mask.
    /// Assumes model output is square (HxH). For non-square outputs, use the overload that accepts explicit dimensions.
    /// </summary>
    /// <param name="output">Raw model output as a float array.</param>
    /// <param name="originalWidth">Original image width to resize the mask to.</param>
    /// <param name="originalHeight">Original image height to resize the mask to.</param>
    /// <returns>Segmentation mask as a grayscale image.</returns>
    /// <exception cref="ArgumentException">Thrown when output array is not square or dimensions are invalid.</exception>
    public static Image<L8> PostprocessOutput(float[] output, int originalWidth, int originalHeight)
    {
        if (output == null || output.Length == 0)
        {
            throw new ArgumentException("Output array cannot be null or empty.", nameof(output));
        }

        if (originalWidth <= 0 || originalHeight <= 0)
        {
            throw new ArgumentException("Image dimensions must be positive.");
        }

        // Calculate the dimensions of the output (assume square)
        int outputSize = (int)Math.Sqrt(output.Length);
        
        // Validate that the output is actually square
        if (outputSize * outputSize != output.Length)
        {
            throw new ArgumentException(
                $"Output array length ({output.Length}) is not a perfect square. " +
                "Use the overload that accepts explicit width and height for non-square outputs.",
                nameof(output));
        }

        return PostprocessOutput(output, outputSize, outputSize, originalWidth, originalHeight);
    }

    /// <summary>
    /// Postprocesses the model output to create a segmentation mask with explicit output dimensions.
    /// </summary>
    /// <param name="output">Raw model output as a float array.</param>
    /// <param name="outputWidth">Width of the model output tensor.</param>
    /// <param name="outputHeight">Height of the model output tensor.</param>
    /// <param name="targetWidth">Target width to resize the mask to.</param>
    /// <param name="targetHeight">Target height to resize the mask to.</param>
    /// <returns>Segmentation mask as a grayscale image.</returns>
    public static Image<L8> PostprocessOutput(
        float[] output, 
        int outputWidth, 
        int outputHeight, 
        int targetWidth, 
        int targetHeight)
    {
        if (output == null || output.Length == 0)
        {
            throw new ArgumentException("Output array cannot be null or empty.", nameof(output));
        }

        if (outputWidth <= 0 || outputHeight <= 0)
        {
            throw new ArgumentException("Output dimensions must be positive.");
        }

        if (targetWidth <= 0 || targetHeight <= 0)
        {
            throw new ArgumentException("Target dimensions must be positive.");
        }

        if (output.Length != outputWidth * outputHeight)
        {
            throw new ArgumentException(
                $"Output array length ({output.Length}) does not match expected dimensions " +
                $"({outputWidth}x{outputHeight}={outputWidth * outputHeight}).");
        }

        // Create a temporary mask image at model output size
        var tempMask = new Image<L8>(outputWidth, outputHeight);

        // Apply sigmoid activation and convert to byte values
        tempMask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < outputHeight; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < outputWidth; x++)
                {
                    int index = y * outputWidth + x;
                    float value = output[index];

                    // Apply sigmoid activation
                    float sigmoid = 1.0f / (1.0f + MathF.Exp(-value));

                    // Convert to byte [0, 255]
                    byte pixelValue = (byte)(sigmoid * 255f);

                    pixelRow[x] = new L8(pixelValue);
                }
            }
        });

        // Resize to target dimensions if needed
        if (outputWidth != targetWidth || outputHeight != targetHeight)
        {
            tempMask.Mutate(ctx => ctx.Resize(targetWidth, targetHeight));
        }

        return tempMask;
    }

    /// <summary>
    /// Applies a threshold to the mask to create a binary segmentation.
    /// </summary>
    /// <param name="mask">Input mask image.</param>
    /// <param name="threshold">Threshold value (0-255). Pixels below this value become black, above become white.</param>
    /// <returns>Thresholded binary mask.</returns>
    public static Image<L8> ApplyThreshold(Image<L8> mask, byte threshold = 128)
    {
        if (mask == null)
        {
            throw new ArgumentNullException(nameof(mask));
        }

        var result = mask.Clone();

        result.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < result.Height; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < result.Width; x++)
                {
                    var pixel = pixelRow[x];
                    pixelRow[x] = new L8(pixel.PackedValue >= threshold ? (byte)255 : (byte)0);
                }
            }
        });

        return result;
    }

    /// <summary>
    /// Applies smoothing to the mask to reduce noise.
    /// </summary>
    /// <param name="mask">Input mask image.</param>
    /// <param name="sigma">Gaussian blur sigma value.</param>
    /// <returns>Smoothed mask.</returns>
    public static Image<L8> SmoothMask(Image<L8> mask, float sigma = 2.0f)
    {
        if (mask == null)
        {
            throw new ArgumentNullException(nameof(mask));
        }

        var result = mask.Clone();
        result.Mutate(ctx => ctx.GaussianBlur(sigma));
        return result;
    }
}
