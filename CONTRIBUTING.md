# Contributing to BiRefNetSharp

Thank you for your interest in contributing to BiRefNetSharp! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, .NET version, etc.)
- Any relevant code samples or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Why it would be useful
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Ensure your code follows the project's coding standards
5. Add or update tests as needed
6. Update documentation if necessary
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

### Prerequisites

- .NET 8.0 SDK or later
- Git
- Your favorite code editor (Visual Studio, VS Code, Rider, etc.)

### Building the Project

```bash
# Clone the repository
git clone https://github.com/dogvane/BiRefNetSharp.git
cd BiRefNetSharp

# Restore dependencies
dotnet restore

# Build the solution
dotnet build

# Run the sample (requires ONNX model)
cd samples/BiRefNetSharp.Sample
dotnet run -- <model_path> <image_path>
```

## Coding Standards

### General Guidelines

- Follow standard C# coding conventions
- Use meaningful variable and method names
- Add XML documentation comments for public APIs
- Keep methods focused and concise
- Handle errors appropriately
- Dispose of resources properly

### Code Style

This project uses an `.editorconfig` file to maintain consistent code style. Key points:

- Use 4 spaces for indentation
- Place opening braces on new lines
- Use `var` when the type is obvious
- Prefer explicit accessibility modifiers
- Use `using` statements for IDisposable objects

### Documentation

- Add XML documentation comments for all public types and members
- Include `<summary>`, `<param>`, and `<returns>` tags as appropriate
- Provide code examples in documentation when helpful
- Update README.md if you add new features
- Add entries to EXAMPLES.md for new usage patterns

### Testing

Currently, this project doesn't have automated tests, but we welcome contributions in this area!

If you add tests:
- Write clear, focused test cases
- Test both success and failure scenarios
- Use descriptive test names
- Mock external dependencies where appropriate

## License

By contributing to BiRefNetSharp, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Reach out to the maintainers

Thank you for contributing to BiRefNetSharp! 🎉
