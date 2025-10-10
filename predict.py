from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# Label mapping
LABELS = {0: "positive", 1: "neutral", 2: "negative"}

def predict_sentiment(text):
    """Predict sentiment of Thai text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()
    
    return {
        "text": text,
        "sentiment": LABELS[pred_label],
        "confidence": f"{confidence:.2%}",
        "all_scores": {LABELS[i]: f"{probs[0][i].item():.2%}" for i in range(4)}
    }

if __name__ == "__main__":
    # Test examples
    examples = [
        "อาหารอร่อยมาก บริการดีเยี่ยม",  # positive
        "ราคาแพงไป คุณภาพไม่คุ้ม",  # negative
        "ปกติครับ ไม่มีอะไรพิเศษ",  # neutral
        "ร้านนี้เปิดกี่โมงครับ?",  # question
    ]
    
    print("[predict] Thai Sentiment Analysis Demo\n")
    for text in examples:
        result = predict_sentiment(text)
        print(f"Text: {result['text']}")
        print(f"➜ Sentiment: {result['sentiment'].upper()} ({result['confidence']})")
        print(f"  Scores: {result['all_scores']}\n")