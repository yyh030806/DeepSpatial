# Installation

This page provides a complete installation guide for DeepSpatial, including environment setup, package installation, verification, and troubleshooting.

## System Requirements

- Python: 3.9 to 3.11 recommended
- OS: Linux, macOS, or Windows (WSL recommended for GPU workflows on Windows)
- GPU (optional but recommended): NVIDIA GPU with CUDA support for faster training

## 1. Create a Clean Python Environment

Using Conda:

```bash
conda create -n deepspatial python=3.10 -y
conda activate deepspatial
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
```

## 2. Install DeepSpatial

### Option A: Install from PyPI

```bash
pip install deepspatial
```

### Option B: Install from Source (Recommended for Development)

```bash
git clone https://github.com/yyh030806/DeepSpatial.git
cd DeepSpatial
pip install -e .
```

## 3. PyTorch and GPU Support

DeepSpatial depends on PyTorch. If you need CUDA acceleration, install the matching PyTorch build for your CUDA version.

Example (CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only environments:

```bash
pip install torch torchvision torchaudio
```

## 4. Verify Installation

Run the following check:

```bash
python -c "import deepspatial, torch; print('deepspatial', deepspatial.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

Expected outcome:

- `deepspatial` version is printed
- PyTorch version is shown
- `cuda True` if GPU is available and configured correctly, otherwise `cuda False`