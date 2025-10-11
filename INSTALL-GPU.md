# Installing GPU-appropriate PyTorch and the project's dependencies

This document explains how to install PyTorch (torch) and the project's Python dependencies so the project runs on NVIDIA (CUDA) and AMD (ROCm) GPUs, and across Linux, macOS, and Windows.

Principles
- The `requirements.txt` file contains the stable, pinned requirements for the project excluding `torch`. This maximizes cross-platform pip installability for the rest of the stack.
- Install `torch` separately using the vendor-appropriate wheels/command (PyTorch provides platform-specific wheels for CUDA and ROCm). Installing `torch` incorrectly is the most common source of cross-platform failures.
- Prefer conda/mamba for heavy binary packages (`pyarrow`, `torch`, etc.) on macOS and Windows; pip is fine for many Linux setups but may require build tools.

1) Prepare environment (example, Linux/macOS zsh)

```bash
# create a venv (or use conda if preferred)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install `torch` (pick the correct option)

- NVIDIA (CUDA) — Linux or Windows with supported CUDA driver
  - Go to https://pytorch.org/get-started/locally/ and select your OS, package (pip/conda), language (Python), and CUDA version. Follow the exact command shown.
  - Example (pip, CPU-only):
    ```bash
    pip install --index-url https://download.pytorch.org/whl/cpu torch
    ```
  - Example (pip, CUDA 12.2) — replace cuda version with the one you need (example only):
    ```bash
    pip install --index-url https://download.pytorch.org/whl/cu122 torch
    ```

- AMD (ROCm) — Linux with ROCm support
  - PyTorch provides ROCm builds for supported ROCm versions. Use the official ROCm install command from https://pytorch.org. Example pattern (adjust ROCm version):
    ```bash
    pip install --index-url https://download.pytorch.org/whl/rocm5.6 torch
    ```
  - Note: ROCm is only supported on certain AMD GPUs and Linux distributions.

- macOS
  - For Intel macOS or Apple silicon, official PyTorch builds are available; for Apple Silicon prefer conda-forge or the official pip wheel if available. If you need Apple GPU acceleration (MPS), recent PyTorch builds include `mps` support. Example conda approach:
    ```bash
    conda create -n thai-sentiment python=3.12
    conda activate thai-sentiment
    conda install pytorch -c pytorch
    ```

3) Install the rest of dependencies

After `torch` is installed, install the rest using the requirements file:

```bash
pip install -r requirements.txt
```

4) If any package fails to build
- Install system build tools (Linux): `build-essential`, `python3-dev`, `cmake`.
- macOS: install Xcode CLT: `xcode-select --install`.
- Windows: install Visual Studio Build Tools (C++) and use conda where possible.
- If `tokenizers` fails to build, install Rust via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.

5) Verify installation

Run this quick check:

```bash
python - <<'PY'
import sys
print('Python', sys.version)
for pkg in ('torch','transformers','datasets','pyarrow','tokenizers'):
    try:
        m = __import__(pkg)
        print(pkg, 'version', getattr(m, '__version__', 'unknown'))
    except Exception as e:
        print(pkg, 'import failed:', e)
PY
```

Notes and caveats
- Exact `torch` wheel URLs and options change frequently. Always prefer the official PyTorch selector (https://pytorch.org/get-started/locally/) to get the right command for your OS, CUDA/ROCm version, and package manager.
- If you need a single-file reproducible environment for CI or other users, consider providing a `environment.yml` (conda) with `torch` channel entries for each platform variant, or separate `environment-linux-cuda.yml` / `environment-linux-rocm.yml` files.
