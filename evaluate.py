from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np

# Load test data
dataset = load_from_disk("./data/wisesight")
test_data = dataset['test']

# Load model
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
model.eval()

# Predict all test samples
print("Predicting on test set...")
all_preds = []
all_labels = []

for i, sample in enumerate(test_data):
    if i % 500 == 0:
        print(f"Progress: {i}/{len(test_data)}")
    
    inputs = tokenizer(sample['texts'], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    
    all_preds.append(pred)
    all_labels.append(sample['category'])

# Convert to numpy
preds = np.array(all_preds)
labels = np.array(all_labels)

# Detailed metrics
LABEL_NAMES = ['positive', 'neutral', 'negative']
print("\n>>>Test Set Performance:\n")
print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=3))

print("\n>>>Confusion Matrix:")
cm = confusion_matrix(labels, preds)
print("pos  neu  neg")
for i, name in enumerate(LABEL_NAMES):
    print(f"{name:8s}: {cm[i]}")

# Per-class accuracy
print("\n>>>Per-Class Accuracy:")
for i, name in enumerate(LABEL_NAMES):
    class_acc = cm[i][i] / cm[i].sum() * 100
    print(f"{name:8s}: {class_acc:.1f}%")

import matplotlib.pyplot as plt
import seaborn as sns

# After confusion matrix calculation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150)
print(">>> Saved confusion_matrix.png")

# Show worst predictions
errors = []
for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
    if pred != label:
        errors.append({
            'text': test_data[i]['texts'],
            'actual': LABEL_NAMES[label],
            'predicted': LABEL_NAMES[pred]
        })

print("\n>>> Sample Errors (first 5):")
for err in errors[:5]:
    print(f"Text: {err['text'][:60]}...")
    print(f"  Actual: {err['actual']}, Predicted: {err['predicted']}\n")