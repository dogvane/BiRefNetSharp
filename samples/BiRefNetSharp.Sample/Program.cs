using BiRefNetSharp;

Console.WriteLine("BiRefNet ONNX Inference Sample");
Console.WriteLine("==============================\n");

// Check command line arguments
if (args.Length < 2)
{
    Console.WriteLine("Usage: BiRefNetSharp.Sample <model_path> <image_path> [output_path]");
    Console.WriteLine("\nExample:");
    Console.WriteLine("  BiRefNetSharp.Sample model.onnx input.jpg output.png");
    Console.WriteLine("\nArguments:");
    Console.WriteLine("  model_path  - Path to the BiRefNet ONNX model file");
    Console.WriteLine("  image_path  - Path to the input image");
    Console.WriteLine("  output_path - (Optional) Path to save the output image with transparent background");
    Console.WriteLine("                If not specified, saves as 'output.png' in current directory");
    return 1;
}

string modelPath = args[0];
string imagePath = args[1];
string outputPath = args.Length > 2 ? args[2] : "output.png";

try
{
    // Validate input files
    if (!File.Exists(modelPath))
    {
        Console.WriteLine($"Error: Model file not found: {modelPath}");
        return 1;
    }

    if (!File.Exists(imagePath))
    {
        Console.WriteLine($"Error: Image file not found: {imagePath}");
        return 1;
    }

    Console.WriteLine($"Model: {modelPath}");
    Console.WriteLine($"Input Image: {imagePath}");
    Console.WriteLine($"Output Image: {outputPath}\n");

    // Load the model
    Console.WriteLine("Loading BiRefNet model...");
    using var model = new BiRefNetModel(modelPath);
    Console.WriteLine("Model loaded successfully!\n");

    // Process the image
    Console.WriteLine("Processing image...");
    var startTime = DateTime.Now;

    BiRefNetHelper.RemoveBackground(
        model: model,
        imagePath: imagePath,
        outputPath: outputPath,
        smoothMask: true,
        maskSigma: 2.0f
    );

    var elapsed = DateTime.Now - startTime;
    Console.WriteLine($"Processing completed in {elapsed.TotalSeconds:F2} seconds\n");

    Console.WriteLine($"Output saved to: {outputPath}");
    Console.WriteLine("\nSuccess!");

    return 0;
}
catch (Exception ex)
{
    Console.WriteLine($"\nError: {ex.Message}");
    Console.WriteLine($"\nStack trace:\n{ex.StackTrace}");
    return 1;
}
