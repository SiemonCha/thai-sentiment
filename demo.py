from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

LABELS = {0: "positive", 1: "neutral", 2: "negative"}
EMOJI = {0: "😊", 1: "😐", 2: "😞"}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        conf = probs[0][pred].item()
    return LABELS[pred], conf, EMOJI[pred]

print("Thai Sentiment Analysis - Interactive Demo")
print("=" * 50)
print("Type Thai text to analyze (or 'quit' to exit)\n")

while True:
    text = input("Thai text: ").strip()
    
    if text.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not text:
        continue
    
    sentiment, confidence, emoji = predict(text)
    print(f"   {emoji} {sentiment.upper()} ({confidence:.1%})\n")