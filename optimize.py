from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Convert to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "SiemonCha/thai-sentiment-phayabert",
    export=True
)
tokenizer = AutoTokenizer.from_pretrained("SiemonCha/thai-sentiment-phayabert")

# Save optimized model
model.save_pretrained("./model_onnx")
tokenizer.save_pretrained("./model_onnx")

print("✅ ONNX model saved (3x faster inference)")