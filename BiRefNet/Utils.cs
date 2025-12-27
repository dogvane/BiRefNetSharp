using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Dogvane.BiRefNet;

/// <summary>
/// BiRefNet ONNX 辅助工具集合，包含图像预处理、输出处理、文件收集等通用函数。
/// </summary>
public static class Utils
{
    public static readonly HashSet<string> ValidExts = new(new[] { 
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp" }, StringComparer.OrdinalIgnoreCase);

    public static readonly float[] ImagenetMean = { 0.485f, 0.456f, 0.406f };
    public static readonly float[] ImagenetStd = { 0.229f, 0.224f, 0.225f };

    /// <summary>
    /// 递归收集指定目录下的图片文件，按允许的扩展名过滤并排序返回。
    /// </summary>
    /// <param name="inputDir">输入目录路径。</param>
    /// <returns>按字典序排序的图片文件路径列表。</returns>
    public static IEnumerable<string> CollectFiles(string inputDir)
    {
        return Directory.EnumerateFiles(inputDir, "*.*", SearchOption.AllDirectories)
            .Where(f => ValidExts.Contains(Path.GetExtension(f)))
            .OrderBy(f => f);
    }

    /// <summary>
    /// 预处理图片：加载、缩放到目标尺寸、BGR->RGB（ImageSharp 已为 RGB）、按 ImageNet 均值/方差归一化，
    /// 并打包为 ONNX 可接受的 CHW 格式张量（1,3,H,W）。
    /// </summary>
    /// <param name="path">输入图片路径。</param>
    /// <param name="targetH">目标高度。</param>
    /// <param name="targetW">目标宽度。</param>
    /// <param name="origW">输出：原始图片宽度。</param>
    /// <param name="origH">输出：原始图片高度。</param>
    /// <returns>返回类型为 DenseTensor<float> 的 CHW 张量。</returns>
    public static DenseTensor<float> Preprocess(string path, int targetH, int targetW, out int origW, out int origH)
    {
        using var image = Image.Load<Rgb24>(path);
        origW = image.Width;
        origH = image.Height;

        image.Mutate(x => x.Resize(new ResizeOptions { Size = new SixLabors.ImageSharp.Size(targetW, targetH), Mode = ResizeMode.Stretch, Sampler = KnownResamplers.Bicubic }));

        var data = new float[1 * 3 * targetH * targetW];
        var stride = targetH * targetW;

        for (var y = 0; y < targetH; y++)
        {
            for (var x = 0; x < targetW; x++)
            {
                var p = image[x, y];
                var r = p.R / 255f;
                var g = p.G / 255f;
                var b = p.B / 255f;

                data[y * targetW + x] = (r - ImagenetMean[0]) / ImagenetStd[0];
                data[stride + y * targetW + x] = (g - ImagenetMean[1]) / ImagenetStd[1];
                data[2 * stride + y * targetW + x] = (b - ImagenetMean[2]) / ImagenetStd[2];
            }
        }

        return new DenseTensor<float>(data, new[] { 1, 3, targetH, targetW });
    }

    /// <summary>
    /// 对浮点数组执行双线性插值缩放，适用于概率图的高质量缩放。
    /// </summary>
    /// <param name="src">源数组（按行优先 HxW 存储）。</param>
    /// <param name="srcW">源宽度。</param>
    /// <param name="srcH">源高度。</param>
    /// <param name="dstW">目标宽度。</param>
    /// <param name="dstH">目标高度。</param>
    /// <returns>返回目标尺寸的浮点数组（行优先）。</returns>
    public static float[] Resize(float[] src, int srcW, int srcH, int dstW, int dstH)
    {
        if (srcW == dstW && srcH == dstH) return src;
        var dst = new float[dstW * dstH];
        var xRatio = (float)srcW / dstW;
        var yRatio = (float)srcH / dstH;

        for (var j = 0; j < dstH; j++)
        {
            var sy = j * yRatio;
            var y0 = Math.Clamp((int)MathF.Floor(sy), 0, srcH - 1);
            var y1 = Math.Clamp(y0 + 1, 0, srcH - 1);
            var wy = sy - y0;

            for (var i = 0; i < dstW; i++)
            {
                var sx = i * xRatio;
                var x0 = Math.Clamp((int)MathF.Floor(sx), 0, srcW - 1);
                var x1 = Math.Clamp(x0 + 1, 0, srcW - 1);
                var wx = sx - x0;

                var v00 = src[y0 * srcW + x0];
                var v01 = src[y0 * srcW + x1];
                var v10 = src[y1 * srcW + x0];
                var v11 = src[y1 * srcW + x1];

                var v0 = v00 * (1 - wx) + v01 * wx;
                var v1 = v10 * (1 - wx) + v11 * wx;
                dst[j * dstW + i] = v0 * (1 - wy) + v1 * wy;
            }
        }

        return dst;
    }

    /// <summary>
    /// 从模型输出张量中提取第一通道的浮点数组，同时返回宽和高。
    /// 支持张量形状 (N,C,H,W)、(1,H,W) 或 (H,W) 等常见形式。
    /// </summary>
    /// <param name="tensor">ONNX 输出张量。</param>
    /// <returns>元组 (data, width, height)。</returns>
    public static (float[] data, int w, int h) ExtractFirstChannel(Tensor<float> tensor)
    {
        var dims = tensor.Dimensions;
        if (dims.Length == 4)
        {
            var h = dims[2];
            var w = dims[3];
            var src = tensor.ToArray();
            var dst = new float[h * w];
            Array.Copy(src, 0, dst, 0, dst.Length);
            return (dst, w, h);
        }

        if (dims.Length == 3)
        {
            var h = dims[1];
            var w = dims[2];
            var src = tensor.ToArray();
            return (src, w, h);
        }

        if (dims.Length == 2)
        {
            var h = dims[0];
            var w = dims[1];
            var src = tensor.ToArray();
            return (src, w, h);
        }

        throw new InvalidOperationException($"Unexpected output shape: [{string.Join(",", dims.ToArray())}]");
    }

    /// <summary>
    /// 对数组就地应用 Sigmoid 函数。
    /// </summary>
    public static void ApplySigmoid(float[] data)
    {
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = 1f / (1f + MathF.Exp(-data[i]));
        }
    }

    /// <summary>
    /// 将数组元素就地截断到 [0,1] 区间。
    /// </summary>
    public static void Clamp01(float[] data)
    {
        for (var i = 0; i < data.Length; i++)
        {
            data[i] = Math.Clamp(data[i], 0f, 1f);
        }
    }

    public static string BuildOutputPath(string file, string inputDir, string outputDir, bool keepTree)
    {
        if (keepTree)
        {
            var rel = Path.GetRelativePath(inputDir, file);
            var withoutExt = Path.Combine(Path.GetDirectoryName(rel) ?? string.Empty, Path.GetFileNameWithoutExtension(rel));
            return Path.Combine(outputDir, withoutExt + ".png");
        }

        var name = Path.GetFileNameWithoutExtension(file);
        return Path.Combine(outputDir, name + ".png");
    }

    /// <summary>
    /// 将 mask 保存为灰度 PNG 图片。
    /// </summary>
    /// <param name="mask">掩码数据（uint8，范围 0-255）。</param>
    /// <param name="w">图片宽度。</param>
    /// <param name="h">图片高度。</param>
    /// <param name="outPath">输出文件路径。</param>
    public static void SaveGrayscalePng(byte[] mask, int w, int h, string outPath)
    {
        var pixels = new Rgb24[mask.Length];
        for (var i = 0; i < mask.Length; i++)
        {
            var v = mask[i];
            pixels[i] = new Rgb24(v, v, v);
        }

        using var img = Image.LoadPixelData(pixels, w, h);
        using var fs = File.Create(outPath);
        img.Save(fs, new PngEncoder());
    }

    /// <summary>
    /// 将 mask 应用到原图并保存，背景设置为白色。
    /// </summary>
    /// <param name="srcPath">原图路径。</param>
    /// <param name="mask">掩码数据（uint8，范围 0-255）。</param>
    /// <param name="w">图片宽度。</param>
    /// <param name="h">图片高度。</param>
    /// <param name="outPath">输出文件路径。</param>
    /// <param name="threshold">阈值，范围 (0, 1]，用于区分前景和背景。</param>
    public static void SaveMaskedPng(string srcPath, byte[] mask, int w, int h, string outPath, double threshold)
    {
        using var orig = Image.Load<Rgb24>(srcPath);
        if (orig.Width != w || orig.Height != h)
        {
            orig.Mutate(x => x.Resize(new ResizeOptions { Size = new Size(w, h), Mode = ResizeMode.Stretch, Sampler = KnownResamplers.Bicubic }));
        }

        // 将 mask 应用到原图：mask 值低的像素设置为白色背景
        // threshold 是 [0, 1] 范围的值，需要转换为 byte 阈值（乘以 255）
        // mask 值 < byteThreshold = 背景（设为白色）
        // mask 值 >= byteThreshold = 前景（保持原色）
        var byteThreshold = (byte)Math.Clamp((int)(threshold * 255), 1, 255);
        
        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var idx = y * w + x;
                var maskValue = mask[idx];
                
                if (maskValue < byteThreshold)
                {
                    // 背景：设置为白色
                    orig[x, y] = new Rgb24(255, 255, 255);
                }
                else
                {
                    // 前景：保持原色
                    orig[x, y] = orig[x, y];
                }
            }
        }

        using var fs = File.Create(outPath);
        orig.Save(fs, new PngEncoder());
    }
}
