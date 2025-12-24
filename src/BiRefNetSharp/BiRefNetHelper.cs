using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace BiRefNetSharp;

/// <summary>
/// Provides utility methods for BiRefNet operations.
/// </summary>
public static class BiRefNetHelper
{
    /// <summary>
    /// Removes the background from an image using BiRefNet model.
    /// </summary>
    /// <param name="model">BiRefNet model instance.</param>
    /// <param name="imagePath">Path to the input image.</param>
    /// <param name="outputPath">Path to save the output image with transparent background.</param>
    /// <param name="smoothMask">Whether to apply smoothing to the mask.</param>
    /// <param name="maskSigma">Sigma value for Gaussian blur if smoothing is enabled.</param>
    public static void RemoveBackground(
        BiRefNetModel model,
        string imagePath,
        string outputPath,
        bool smoothMask = true,
        float maskSigma = 2.0f)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path cannot be null or empty.", nameof(imagePath));
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image file not found: {imagePath}", imagePath);
        }

        // Load the input image
        using var inputImage = Image.Load<Rgb24>(imagePath);

        // Run inference
        using var mask = model.Predict(inputImage);

        // Apply smoothing if requested
        using var finalMask = smoothMask ? ImagePostprocessor.SmoothMask(mask, maskSigma) : mask.Clone();

        // Apply mask to get transparent background
        using var result = BiRefNetModel.ApplyMask(inputImage, finalMask);

        // Save the result
        result.SaveAsPng(outputPath);
    }

    /// <summary>
    /// Removes the background from an image using BiRefNet model asynchronously.
    /// </summary>
    /// <param name="model">BiRefNet model instance.</param>
    /// <param name="imagePath">Path to the input image.</param>
    /// <param name="outputPath">Path to save the output image with transparent background.</param>
    /// <param name="smoothMask">Whether to apply smoothing to the mask.</param>
    /// <param name="maskSigma">Sigma value for Gaussian blur if smoothing is enabled.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public static async Task RemoveBackgroundAsync(
        BiRefNetModel model,
        string imagePath,
        string outputPath,
        bool smoothMask = true,
        float maskSigma = 2.0f,
        CancellationToken cancellationToken = default)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path cannot be null or empty.", nameof(imagePath));
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image file not found: {imagePath}", imagePath);
        }

        // Load the input image
        using var inputImage = await Image.LoadAsync<Rgb24>(imagePath, cancellationToken);

        // Run inference
        using var mask = await model.PredictAsync(inputImage, cancellationToken);

        // Apply smoothing if requested
        using var finalMask = smoothMask ? ImagePostprocessor.SmoothMask(mask, maskSigma) : mask.Clone();

        // Apply mask to get transparent background
        using var result = BiRefNetModel.ApplyMask(inputImage, finalMask);

        // Save the result
        await result.SaveAsPngAsync(outputPath, cancellationToken);
    }

    /// <summary>
    /// Gets only the segmentation mask from an image.
    /// </summary>
    /// <param name="model">BiRefNet model instance.</param>
    /// <param name="imagePath">Path to the input image.</param>
    /// <param name="outputPath">Path to save the mask image.</param>
    /// <param name="smoothMask">Whether to apply smoothing to the mask.</param>
    /// <param name="maskSigma">Sigma value for Gaussian blur if smoothing is enabled.</param>
    public static void GetMask(
        BiRefNetModel model,
        string imagePath,
        string outputPath,
        bool smoothMask = true,
        float maskSigma = 2.0f)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path cannot be null or empty.", nameof(imagePath));
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image file not found: {imagePath}", imagePath);
        }

        // Load the input image
        using var inputImage = Image.Load<Rgb24>(imagePath);

        // Run inference
        using var mask = model.Predict(inputImage);

        // Apply smoothing if requested
        using var finalMask = smoothMask ? ImagePostprocessor.SmoothMask(mask, maskSigma) : mask.Clone();

        // Save the mask
        finalMask.SaveAsPng(outputPath);
    }
}
