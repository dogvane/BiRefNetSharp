# Changelog

All notable changes to BiRefNetSharp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-24

### Added
- Initial release of BiRefNetSharp
- Core BiRefNetModel class for ONNX inference
- ImagePreprocessor with ImageNet normalization support
- ImagePostprocessor with mask generation and refinement
- BiRefNetHelper utilities for common operations
- Support for both synchronous and asynchronous operations
- Sample console application demonstrating usage
- Comprehensive documentation and examples
- Support for custom ONNX session options
- Background removal functionality
- Mask extraction and manipulation
- Threshold and smoothing operations
- XML documentation for IntelliSense support

### Dependencies
- Microsoft.ML.OnnxRuntime 1.19.2
- SixLabors.ImageSharp 3.1.12
- .NET 8.0

[1.0.0]: https://github.com/dogvane/BiRefNetSharp/releases/tag/v1.0.0
