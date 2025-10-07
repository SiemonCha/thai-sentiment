from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load data
dataset = load_from_disk("./data/wisesight")
train_data = dataset['train']
test_data = dataset['test']

# Load PhayaThaiBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("clicknext/phayathaibert")

# Tokenize function
def tokenize(batch):
    return tokenizer(batch['texts'], padding='max_length', truncation=True, max_length=128)

# Tokenize datasets
train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

# Rename 'category' to 'labels' for Trainer
train_data = train_data.rename_column('category', 'labels')
test_data = test_data.rename_column('category', 'labels')

# Set format
train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load model (4 classes)
model = AutoModelForSequenceClassification.from_pretrained(
    "clicknext/phayathaibert",
    num_labels=4
)

# Metrics
def compute_metrics(pred):  
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Auto-detect hardware capabilities
def get_device_config():
    if torch.cuda.is_available():
        return {"fp16": True}  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return {"bf16": True}  # Mac M1/M2
    else:
        return {}  # CPU only
    
device_config = get_device_config()
print(f"Using device config: {device_config}")

# Training config
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    **device_config,  # Auto-apply fp16/bf16 based on hardware
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Train
print(">>> Starting training...")
trainer.train()

# Save
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("[yes] Model saved to ./model/")