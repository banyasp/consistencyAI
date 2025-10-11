# Installation Guide

Complete installation instructions for ConsistencyAI.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB+ recommended for large experiments)
- Internet connection (for downloading models and querying APIs)

## Quick Installation

### Install from Source

```bash
# Clone the repository
git clone https://github.com/banyasp/consistencyAI.git
cd consistencyAI

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Verify installation
python verify_installation.py
```

## Configuration

### Step 1: Get API Keys

You need at least one API key:

**OpenRouter (Recommended):**
- Sign up at [openrouter.ai](https://openrouter.ai)
- Get your API key from the dashboard
- OpenRouter provides access to 100+ models from different providers

**Optional Direct Provider Keys:**
- OpenAI: [platform.openai.com](https://platform.openai.com)
- Google Gemini: [ai.google.dev](https://ai.google.dev)

### Step 2: Set API Keys

Choose one method:

**A. Environment Variables (Recommended for Production)**

```bash
# Linux/Mac
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_API_KEY="your-openai-key"      # Optional
export GOOGLE_API_KEY="your-google-key"      # Optional

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="your-openrouter-key"
$env:OPENAI_API_KEY="your-openai-key"        # Optional
$env:GOOGLE_API_KEY="your-google-key"        # Optional

# Add to shell profile for persistence
echo 'export OPENROUTER_API_KEY="your-key"' >> ~/.bashrc  # or ~/.zshrc
```

**B. Python Configuration (Quick Testing)**

```python
from duplicity import config

config.set_openrouter_key("your-openrouter-key")
config.set_openai_key("your-openai-key")      # Optional
config.set_google_key("your-google-key")      # Optional
```

**C. Configuration File (Not Recommended for Security)**

Edit `duplicity/config.py`:
```python
OPENROUTER_API_KEY = "your-openrouter-key"
```

## Verify Installation

```bash
python verify_installation.py
```

Expected output:
```
============================================================
ConsistencyAI Installation Verification
============================================================

Testing imports...
  ✓ Main package imported
  ✓ llm_tool imported
  ✓ personas imported
  ...

SUCCESS: All tests passed!
```

## First Run

Try running the main.ipynb notebook or create your own script to test the installation.

Expected runtime: 5-10 minutes for a basic experiment

## Troubleshooting

### ImportError: No module named 'duplicity'

**Solution:**
```bash
# Make sure you're in the package directory
cd /path/to/consistencyAI
pip install -e .
```

### ValueError: OPENROUTER_API_KEY is not set

**Solution:**
```bash
export OPENROUTER_API_KEY="your-key-here"
```

Or in Python:
```python
from duplicity import config
config.set_openrouter_key("your-key")
```

### ModuleNotFoundError: No module named 'torch'

Some dependencies take time to install (especially PyTorch).

**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### MemoryError during similarity computation

Large experiments may require more RAM.

**Solution:**
- Reduce the number of personas
- Process fewer topics at once
- Use smaller batch sizes

```python
# Reduce memory usage
personas = get_and_clean_personas(length=10)  # Instead of 100
results = query_llm_fast(..., initial_batch_size=10)  # Instead of 50
```

### SSL Certificate Error

**Solution:**
```bash
pip install --upgrade certifi
```

### Rate Limit Errors

If you hit API rate limits:

**Solution:**
- Reduce concurrency: `initial_concurrency=5` instead of 20
- Reduce batch size: `initial_batch_size=10` instead of 50
- Enable adaptive mode (default) to automatically adjust

### SentenceTransformers Model Download

First run will download the SentenceBERT model (~80MB).

**Solution:**
- Wait for the download to complete
- Ensure internet connection
- Or pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

## Platform-Specific Notes

### macOS

If you have issues with SSL or certificates:
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

### Windows

If you have issues with long paths:
```bash
# Enable long path support (requires admin)
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

### Linux

Make sure you have required system libraries:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
```

## Development Installation

For contributing to the project:

```bash
# Clone the repository
git clone https://github.com/banyasp/consistencyAI.git
cd consistencyAI

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if configured)
pre-commit install
```

## Docker Installation (Optional)

Coming soon: Docker container for easy deployment.

## Cloud Deployment

For running on cloud platforms:

**Google Colab:**
```python
!git clone https://github.com/banyasp/consistencyAI.git
%cd consistencyAI
!pip install -e .
import os
os.environ['OPENROUTER_API_KEY'] = 'your-key'
```

**AWS/Azure/GCP:**
- Use secrets management for API keys
- Deploy as a containerized application
- Scale with managed services

## Uninstallation

```bash
pip uninstall duplicity
```

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review the [QUICKSTART.md](QUICKSTART.md)
3. Read the [API.md](API.md) documentation
4. Search existing [GitHub issues](https://github.com/banyasp/consistencyAI/issues)
5. Open a new issue with:
   - Python version (`python --version`)
   - Operating system
   - Error message and full stack trace
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](QUICKSTART.md) for a quick tutorial
2. Read [API.md](API.md) for complete API documentation
3. Build your own experiments!

Happy benchmarking!

