# Contributing to Local RAG

We love your input! We want to make contributing to Local RAG as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. Set up your development environment:
   ```bash
   # Using conda (recommended)
   conda env create -f env.yml
   conda activate enterprise-rag

   # Or using pip
   pip install -r requirements.txt
   ```
3. Make your changes
4. Ensure your code follows our style guidelines
5. Test your changes
6. Issue that pull request!

## Environment Setup

The project uses Python and requires specific dependencies for GPU-accelerated document processing and embedding generation. We provide two options for setting up your environment:

1. Using Conda (Recommended):
   - Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - Create the environment using: `conda env create -f env.yml`
   - Activate the environment: `conda activate enterprise-rag`

2. Using pip:
   - Create a virtual environment: `python -m venv venv`
   - Activate it:
     - Windows: `venv\Scripts\activate`
     - Unix/MacOS: `source venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`

## Code Style Guidelines

We follow standard Python coding conventions:

1. Use [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
2. Use meaningful variable and function names
3. Write docstrings for functions and classes
4. Keep functions focused and modular
5. Comment complex logic
6. Use type hints where possible

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the requirements.txt or env.yml if you add or remove dependencies
3. The PR will be merged once you have the sign-off of at least one maintainer

## Reporting Bugs

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](../../issues/new); it's that easy!

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We love feature requests! When submitting a feature request, please:

1. Check if the feature has already been requested
2. Provide a clear description of the feature and its benefits
3. If possible, outline how the feature might be implemented

## Project Structure

```
enterprise-rag/
├── src/
│   ├── chat_interface.py      # Chat interface implementation
│   ├── document_processor_gpu.py  # GPU-accelerated document processing
│   ├── embedding_models.py    # Embedding model implementations
│   ├── language_models.py     # Language model implementations
│   ├── rag_config.py         # Configuration settings
│   └── rag_pipeline.py       # Main RAG pipeline implementation
├── data/                     # Data directory
├── env.yml                   # Conda environment specification
└── requirements.txt          # Python package requirements
```

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

Don't hesitate to ask questions by opening an issue or reaching out to the maintainers.

Thank you for contributing to Local RAG!
