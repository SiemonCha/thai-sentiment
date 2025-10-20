# 🇹🇭 Thai Sentiment Analysis with PhayaThaiBERT

Production-ready Thai sentiment classifier with 82% accuracy. Fine-tuned PhayaThaiBERT on 21k social media messages.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/SiemonCha/thai-sentiment-demo)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/SiemonCha/thai-sentiment-phayabert)

## ⚡ Quick Demo

Try it now: **[Live Demo](https://huggingface.co/spaces/SiemonCha/thai-sentiment-demo)**

Or run locally:

```bash
pip install transformers torch gradio
python app.py
```

## 🎯 Features

- **Multiple Interfaces:** CLI, Web UI, REST API
- **Production Ready:** Docker, FastAPI, HuggingFace hosting
- **High Performance:** 82% accuracy, ONNX optimization
- **Explainable:** LIME interpretability
- **Well Tested:** Unit tests, CI/CD
- **Hardware Agnostic:** GPU/Mac/CPU support

## 📊 Performance

| Metric           | Score |
| ---------------- | ----- |
| Overall Accuracy | 82%   |
| Positive F1      | 0.72  |
| Neutral F1       | 0.85  |
| Negative F1      | 0.85  |

## 🚀 Quick Start

### Use Pre-trained Model (No Training)

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                     model="SiemonCha/thai-sentiment-phayabert")

result = classifier("อาหารอร่อยมาก")
print(result)
# [{'label': 'POSITIVE', 'score': 0.98}]
```

### Run Web UI

```bash
git clone https://github.com/SiemonCha/thai-sentiment
cd thai-sentiment
pip install -r requirements.txt
python app.py
```

### REST API

```bash
python api.py
# API docs: http://localhost:8000/docs
```

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "อาหารอร่อยมาก"}'
```

### Docker

```bash
docker build -t thai-sentiment .
docker run -p 7860:7860 thai-sentiment
```

## 🔬 Training from Scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python data_download.py

# 3. Train model (~40 min Mac M1 / ~15 min GPU)
python train.py

# 4. Evaluate
python evaluate.py
```

## 📁 Project Structure

```
thai-sentiment/
├── app.py                 # Gradio web UI
├── api.py                 # FastAPI REST API
├── demo.py                # Interactive CLI
├── predict.py             # Batch predictions
├── train.py               # Training script
├── evaluate.py            # Evaluation with metrics
├── explain.py             # LIME explainability
├── optimize.py            # ONNX conversion
├── benchmark.py           # Model comparison
├── test_model.py          # Unit tests
├── Dockerfile             # Container
└── requirements.txt       # Dependencies
```

## 🖥️ Hardware Support

| Platform          | Training Time | Status          |
| ----------------- | ------------- | --------------- |
| Mac M1/M2         | ~40 min       | ✅ Tested       |
| NVIDIA GPU        | ~15 min       | ✅ Tested       |
| AMD GPU (Linux)   | ~15 min       | ✅ Tested       |
| CPU               | ~4-5 hours    | ✅ Works        |
| AMD GPU (Windows) | ~4-5 hours    | ⚠️ CPU fallback |

## 🧪 Testing

```bash
# Run tests
pytest test_model.py -v

# Benchmark
python benchmark.py

# Explain predictions
python explain.py
```

## 📈 Advanced Features

### Model Explainability

```bash
python explain.py
# Generates explanation.html showing word contributions
```

### ONNX Optimization (3x faster)

```bash
python optimize.py
# Creates optimized model in ./model_onnx/
```

### Batch Processing

```bash
python batch_predict.py
# Processes input.csv → output.csv
```

## 🔧 Technical Details

- **Base Model:** PhayaThaiBERT (110M parameters)
- **Dataset:** Wisesight Sentiment (21k messages)
- **Training:** 5 epochs, class-weighted loss, AdamW
- **Architecture:** BERT with classification head
- **Max Length:** 128 tokens
- **Classes:** Positive, Neutral, Negative

## 📝 Citation

```bibtex
@misc{thai-sentiment-2025,
  author = {SiemonCha},
  title = {Thai Sentiment Analysis with PhayaThaiBERT},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SiemonCha/thai-sentiment}
}
```

Dataset:

```bibtex
@software{wisesight_sentiment,
  author = {Suriyawongkul, Arthit and Chuangsuwanich, Ekapol},
  title = {PyThaiNLP/wisesight-sentiment},
  year = 2019,
  doi = {10.5281/zenodo.3457447}
}
```

## 📄 License

MIT License

## 🔗 Links

- **Live Demo:** https://huggingface.co/spaces/SiemonCha/thai-sentiment-demo
- **Model Hub:** https://huggingface.co/SiemonCha/thai-sentiment-phayabert
- **GitHub:** https://github.com/SiemonCha/thai-sentiment
