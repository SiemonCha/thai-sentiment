from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("--- Loading model from HuggingFace...")
model = AutoModelForSequenceClassification.from_pretrained("SiemonCha/thai-sentiment-phayabert")
tokenizer = AutoTokenizer.from_pretrained("SiemonCha/thai-sentiment-phayabert")
print(">>> Model loaded successfully!\n")

# Test prediction
text = "อาหารอร่อยมาก"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    prediction = torch.argmax(probs).item()

labels = {0: "positive", 1: "neutral", 2: "negative"}
print(f"Text: {text}")
print(f"Prediction: {labels[prediction]}")
print(f"Confidence: {probs[prediction]:.2%}")
print("\n>>> Model works from HuggingFace!")
