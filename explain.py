from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("SiemonCha/thai-sentiment-phayabert")
tokenizer = AutoTokenizer.from_pretrained("SiemonCha/thai-sentiment-phayabert")

LABELS = ['positive', 'neutral', 'negative']

def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# Create explainer
explainer = LimeTextExplainer(class_names=LABELS)

# Explain prediction
text = "อาหารอร่อยมาก บริการดีเยี่ยม"
exp = explainer.explain_instance(text, predict_proba, num_features=10, num_samples=500)

# Show explanation
print(f"\nPrediction: {LABELS[np.argmax(predict_proba([text]))]}")
print("\nTop words contributing to prediction:")
for word, weight in exp.as_list():
    print(f"  {word}: {weight:.3f}")

# Save visualization
exp.save_to_file('explanation.html')
print("\n✅ Saved explanation.html")