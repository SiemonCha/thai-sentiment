# Thai Sentiment Analysis with PhayaThaiBERT

This repository contains code to train and run a Thai sentiment classifier built on PhayaThaiBERT.

- Setup (Python environment and dependencies)
- Download dataset
- Train the model
- Evaluate on test data
- Run predictions (scripted and interactive demo)

Prerequisites

- Python 3.11+ (the project venv shows Python 3.12 but 3.11+ should work)
- pip and/or conda (conda recommended for heavy binary packages)
- Optional GPU drivers for NVIDIA (CUDA) or AMD (ROCm) if you want to train faster

Files overview (in execution order)

- `requirements.txt` — pip installable requirements (note: `torch` is installed separately)
- `INSTALL-GPU.md` — platform-specific instructions for installing `torch`/GPU support
- `data_download.py` — download and prepare the Wisesight dataset (saves to `./data/wisesight`)
- `train.py` — tokenizes, trains a sequence classification model and saves `./model`
- `evaluate.py` — evaluates the saved `./model` on the test split
- `predict.py` — batch/personal predictions using `./model` (example main included)
- `demo.py` — interactive CLI for quick manual testing

Quick start (CPU-only)

1. Create a virtual env and activate it (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install the pip requirements (note: `torch` is NOT installed here):

```bash
pip install -r requirements.txt
```

3. (Optional) Install `torch` now if you have a supported GPU or want CPU-only PyTorch. See `INSTALL-GPU.md` for platform-specific commands.

Download dataset

```bash
python data_download.py
```

This saves a cleaned dataset at `./data/wisesight` with `train` and `test` splits.

Train

Basic training (this script uses `clicknext/phayathaibert` as base and trains 3-class labels):

```bash
python train.py
```

Notes:

- Training detects hardware and will enable FP16 on CUDA-enabled NVIDIA GPUs or bfloat16 on Apple MPS when available.
- Training saves the final model and tokenizer to `./model`.

Evaluate

After training (or if you already have `./model`), run evaluation:

```bash
python evaluate.py
```

This will load `./data/wisesight/test` and `./model`, run predictions over the test set, and print classification metrics and the confusion matrix.

Predict (scripted)

Use `predict.py` to run example predictions or integrate the `predict_sentiment` function into other scripts. Example:

```bash
python predict.py
```

Predict (interactive)

```bash
python demo.py
```

This runs a simple command-line REPL where you can type Thai sentences and get sentiment + confidence.

Installing `torch` (GPU/CPU) — important

PyTorch (`torch`) is intentionally NOT pinned in `requirements.txt`. This avoids cross-platform wheel and GPU mismatches. Please install `torch` separately according to your OS/GPU using the instructions in `INSTALL-GPU.md`.
