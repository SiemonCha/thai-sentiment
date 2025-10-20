import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load trained model
try:
    # Try loading from HuggingFace first
    model = AutoModelForSequenceClassification.from_pretrained("SiemonCha/thai-sentiment-phayabert")
    tokenizer = AutoTokenizer.from_pretrained("SiemonCha/thai-sentiment-phayabert")
    print(">>> Loaded model from HuggingFace")
except Exception as e:
    # Fallback to local model
    print(f">>>  Could not load from HuggingFace: {e}")
    print("Loading from local ./model/")
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    

LABELS = {0: "Positive 😊", 1: "Neutral 😐", 2: "Negative 😞"}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    return {
        LABELS[0]: float(probs[0]),
        LABELS[1]: float(probs[1]),
        LABELS[2]: float(probs[2])
    }

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=3,
        placeholder="พิมพ์ข้อความภาษาไทยที่นี่...",
        label="Thai Text"
    ),
    outputs=gr.Label(label="Sentiment Prediction"),
    title=">>> Thai Sentiment Analysis",
    description="Fine-tuned PhayaThaiBERT on 21k social media messages",
    examples=[
        ["อาหารอร่อยมาก บริการดีเยี่ยม"],
        ["ราคาแพงไป คุณภาพไม่คุ้ม"],
        ["ปกติครับ ไม่มีอะไรพิเศษ"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)  # Creates public link for 72hrs