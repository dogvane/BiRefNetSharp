using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CommandLine;
using Dogvane.BiRefNet;

namespace Dogvane.BiRefNet.Example;

internal class Options
{
    [Option("onnx_path", Required = false, HelpText = "ONNX model path", Default = "../models/onnx/model_fp16.onnx")]
    public string OnnxPath { get; set; } = "../models/onnx/model_fp16.onnx";

    [Option("input_dir", Required = false, HelpText = "Input images folder", Default = "./")]
    public string InputDir { get; set; } = "./";

    [Option("output_dir", Required = false, HelpText = "Output folder", Default = "../output")]
    public string OutputDir { get; set; } = "../output";

    [Option("device", Default = "cpu", HelpText = "cpu/cuda")]
    public string Device { get; set; } = "cpu";

    [Option("keep_tree", Default = false, HelpText = "Keep input subfolder structure under output_dir")]
    public bool KeepTree { get; set; }

    [Option("threshold", Default = 0.1, HelpText = "Mask threshold in range (0, 1], lower values = more aggressive foreground extraction")]
    public double Threshold { get; set; } = 0.1;
}

internal static class Program
{
    private static int Main(string[] args)
    {
        return Parser.Default.ParseArguments<Options>(args)
            .MapResult(Run, errs => 1);
    }

    private static int Run(Options opts)
    {
        // 显示欢迎信息和配置
        Console.WriteLine("========================================");
        Console.WriteLine("  BiRefNetSharp - Image Segmentation");
        Console.WriteLine("========================================");
        Console.WriteLine();
        Console.WriteLine("[config] Model path: " + Path.GetFullPath(opts.OnnxPath));
        Console.WriteLine("[config] Input directory: " + Path.GetFullPath(opts.InputDir));
        Console.WriteLine("[config] Output directory: " + Path.GetFullPath(opts.OutputDir));
        Console.WriteLine("[config] Device: " + opts.Device);
        Console.WriteLine("[config] Keep tree structure: " + (opts.KeepTree ? "yes" : "no"));
        Console.WriteLine("[config] Mask threshold: " + opts.Threshold.ToString("F2") + " (0-1]");
        Console.WriteLine();

        if (!File.Exists(opts.OnnxPath))
        {
            Console.WriteLine($"[error] ONNX model not found: {opts.OnnxPath}");
            Console.WriteLine();
            Console.WriteLine("Please download the model from:");
            Console.WriteLine("https://modelscope.cn/models/onnx-community/BiRefNet-ONNX");
            Console.WriteLine();
            Console.WriteLine("And place it at: models/onnx/model_fp16.onnx");
            return 1;
        }

        if (!Directory.Exists(opts.InputDir))
        {
            Console.WriteLine($"[error] input_dir not found: {opts.InputDir}");
            return 1;
        }

        Directory.CreateDirectory(opts.OutputDir);

            using var runner = new BiRefNetInference(opts.OnnxPath, opts.Device);

        var files = Utils.CollectFiles(opts.InputDir).ToList();
        if (files.Count == 0)
        {
            Console.WriteLine("[warn] no images found.");
            return 0;
        }

        Console.WriteLine($"[info] providers: {string.Join(", ", runner.GetProviders())}");
        Console.WriteLine($"[info] total images: {files.Count}");

        var sw = Stopwatch.StartNew();
        foreach (var file in files)
        {
            try
            {
                var (mask, w, h) = runner.Infer(file);
                var outPath = Utils.BuildOutputPath(file, opts.InputDir, opts.OutputDir, opts.KeepTree);
                Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);

                Utils.SaveGrayscalePng(mask, w, h, outPath);

                var maskedPath = Path.Combine(Path.GetDirectoryName(outPath)!, Path.GetFileNameWithoutExtension(outPath) + "_masked.png");
                try
                {
                    Utils.SaveMaskedPng(file, mask, w, h, maskedPath, opts.Threshold);
                }
                catch (Exception ex2)
                {
                    Console.WriteLine($"[warn] failed to save masked image for {file}: {ex2.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[skip] {file}: {ex.Message}");
            }
        }

        sw.Stop();
        Console.WriteLine($"[ok] done in {sw.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"[out] {Path.GetFullPath(opts.OutputDir)}");
        return 0;
    }
}
