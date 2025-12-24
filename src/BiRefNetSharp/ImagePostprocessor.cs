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
    /// </summary>
    /// <param name="output">Raw model output as a float array.</param>
    /// <param name="originalWidth">Original image width.</param>
    /// <param name="originalHeight">Original image height.</param>
    /// <returns>Segmentation mask as a grayscale image.</returns>
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

        // Calculate the dimensions of the output
        int outputSize = (int)Math.Sqrt(output.Length);

        // Create a temporary mask image at model output size
        var tempMask = new Image<L8>(outputSize, outputSize);

        // Apply sigmoid activation and convert to byte values
        tempMask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < outputSize; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < outputSize; x++)
                {
                    int index = y * outputSize + x;
                    float value = output[index];

                    // Apply sigmoid activation
                    float sigmoid = 1.0f / (1.0f + MathF.Exp(-value));

                    // Convert to byte [0, 255]
                    byte pixelValue = (byte)(sigmoid * 255f);

                    pixelRow[x] = new L8(pixelValue);
                }
            }
        });

        // Resize to original dimensions if needed
        if (outputSize != originalWidth || outputSize != originalHeight)
        {
            tempMask.Mutate(ctx => ctx.Resize(originalWidth, originalHeight));
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
