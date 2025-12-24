using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace BiRefNetSharp;

/// <summary>
/// BiRefNet model for image segmentation and background removal using ONNX Runtime.
/// </summary>
public class BiRefNetModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly SessionOptions _sessionOptions;
    private bool _disposed;

    /// <summary>
    /// Gets the input name for the ONNX model.
    /// </summary>
    public string InputName { get; }

    /// <summary>
    /// Gets the output name for the ONNX model.
    /// </summary>
    public string OutputName { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="BiRefNetModel"/> class.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional session options for ONNX Runtime.</param>
    public BiRefNetModel(string modelPath, SessionOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);
        }

        _sessionOptions = options ?? new SessionOptions();
        _session = new InferenceSession(modelPath, _sessionOptions);

        // Get input and output names from the model
        InputName = _session.InputMetadata.Keys.First();
        OutputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Runs inference on the input image and returns the segmentation mask.
    /// </summary>
    /// <param name="inputImage">Input image to process.</param>
    /// <returns>Segmentation mask as a grayscale image.</returns>
    public Image<L8> Predict(Image<Rgb24> inputImage)
    {
        if (inputImage == null)
        {
            throw new ArgumentNullException(nameof(inputImage));
        }

        // Preprocess the image
        var inputTensor = ImagePreprocessor.PreprocessImage(inputImage);

        // Create input for ONNX Runtime
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName, inputTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // Postprocess the output
        var mask = ImagePostprocessor.PostprocessOutput(output, inputImage.Width, inputImage.Height);

        return mask;
    }

    /// <summary>
    /// Runs inference on the input image and returns the segmentation mask asynchronously.
    /// </summary>
    /// <param name="inputImage">Input image to process.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Segmentation mask as a grayscale image.</returns>
    public Task<Image<L8>> PredictAsync(Image<Rgb24> inputImage, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Predict(inputImage), cancellationToken);
    }

    /// <summary>
    /// Applies the segmentation mask to the input image to remove the background.
    /// </summary>
    /// <param name="inputImage">Original input image.</param>
    /// <param name="mask">Segmentation mask.</param>
    /// <returns>Image with transparent background.</returns>
    public static Image<Rgba32> ApplyMask(Image<Rgb24> inputImage, Image<L8> mask)
    {
        if (inputImage == null)
        {
            throw new ArgumentNullException(nameof(inputImage));
        }

        if (mask == null)
        {
            throw new ArgumentNullException(nameof(mask));
        }

        if (inputImage.Width != mask.Width || inputImage.Height != mask.Height)
        {
            throw new ArgumentException("Input image and mask must have the same dimensions.");
        }

        var result = new Image<Rgba32>(inputImage.Width, inputImage.Height);

        inputImage.ProcessPixelRows(mask, result, (inputAccessor, maskAccessor, resultAccessor) =>
        {
            for (int y = 0; y < inputImage.Height; y++)
            {
                var inputRow = inputAccessor.GetRowSpan(y);
                var maskRow = maskAccessor.GetRowSpan(y);
                var resultRow = resultAccessor.GetRowSpan(y);

                for (int x = 0; x < inputImage.Width; x++)
                {
                    var pixel = inputRow[x];
                    var alpha = maskRow[x].PackedValue;

                    resultRow[x] = new Rgba32(pixel.R, pixel.G, pixel.B, alpha);
                }
            }
        });

        return result;
    }

    /// <summary>
    /// Releases all resources used by the <see cref="BiRefNetModel"/>.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases the unmanaged resources used by the <see cref="BiRefNetModel"/> and optionally releases the managed resources.
    /// </summary>
    /// <param name="disposing">true to release both managed and unmanaged resources; false to release only unmanaged resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _session?.Dispose();
                _sessionOptions?.Dispose();
            }

            _disposed = true;
        }
    }
}
