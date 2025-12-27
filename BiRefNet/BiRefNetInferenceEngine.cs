using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Dogvane.BiRefNet;

/// <summary>
/// Wraps an ONNX Runtime session to run BiRefNet inference for a collection of images.
/// </summary>
public sealed class BiRefNetInference : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _modelInputLayerName;
    private readonly int _modelInputH;
    private readonly int _modelInputW;

    /// <summary>
    /// 创建并初始化 ONNX Runtime 会话。
    /// </summary>
    /// <param name="onnxPath">ONNX 模型文件路径。</param>
    /// <param name="device">运行设备，仅支持 "cpu"。</param>
    public BiRefNetInference(string onnxPath, string device)
    {
        _session = BuildSession(onnxPath, device);
        _modelInputLayerName = _session.InputMetadata.Keys.First();

        // 从模型元数据读取期望的输入尺寸（通常为 N,C,H,W）并设置内部 inputH/inputW
        try
        {
            var meta = _session.InputMetadata[_modelInputLayerName];
            var dims = meta.Dimensions.ToArray();
            if (dims.Length >= 4 && dims[2] > 0 && dims[3] > 0)
            {
                _modelInputH = dims[2];
                _modelInputW = dims[3];
                Console.WriteLine($"[info] model input size detected: {_modelInputW}x{_modelInputH} (W x H)");
            }
            else
            {
                _modelInputH = 512;
                _modelInputW = 512;
            }
        }
        catch
        {
            _modelInputH = 512;
            _modelInputW = 512;
        }
    }

    /// <summary>
    /// 释放 ONNX 会话占用的资源。
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
    }

    /// <summary>
    /// 对单张图像执行推理，并返回已经缩放到原始分辨率的单通道 uint8 掩码数据（不做保存）。
    /// </summary>
    /// <param name="imagePath">输入图片路径。</param>
    /// <returns>返回元组 (maskBytes, width, height)，maskBytes 为行优先的 uint8 灰度数据。</returns>
    public (byte[] mask, int width, int height) Infer(string imagePath)
    {
        var tensor = Utils.Preprocess(imagePath, _modelInputH, _modelInputW, out var origW, out var origH);

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_modelInputLayerName, tensor) });
        var last = results.Last();
        var output = last.AsTensor<float>();

        var (prob, outW, outH) = Utils.ExtractFirstChannel(output);
        Utils.ApplySigmoid(prob);
        Utils.Clamp01(prob);

        // 使用双线性浮点插值将概率图从模型输出尺寸缩放到原始分辨率
        var resizedProb = Utils.Resize(prob, outW, outH, origW, origH);

        // 转为 uint8 并返回
        var mask = new byte[origW * origH];
        for (var i = 0; i < mask.Length; i++)
        {
            var v = resizedProb[i] * 255.0f + 0.5f;
            if (v < 0f) v = 0f; if (v > 255f) v = 255f;
            mask[i] = (byte)v;
        }

        return (mask, origW, origH);
    }

    // 返回运行时 provider 列表（占位实现，必要时可改为查询真实 provider）
    public IReadOnlyList<string> GetProviders() => new[] { "CPU" };

    /// <summary>
    /// 构建仅限 CPU 的 ONNX Runtime 会话。
    /// </summary>
    private static InferenceSession BuildSession(string onnxPath, string device)
    {
        if (!device.Equals("cpu", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine("[warn] Only CPU execution is supported in this build; forcing CPU providers.");
        }

        Console.SetError(TextWriter.Null);
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        };

        return new InferenceSession(onnxPath, options);
    }
}
