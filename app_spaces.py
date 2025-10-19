import gradio as gr
from transformers import pipeline

# Simplified for Spaces
classifier = pipeline(
    "text-classification",
    model="SiemonCha/thai-sentiment-phayabert",
    return_all_scores=True
)

LABELS = {"LABEL_0": "Positive 😊", "LABEL_1": "Neutral 😐", "LABEL_2": "Negative 😞"}

def predict(text):
    results = classifier(text)[0]
    return {LABELS[r['label']]: r['score'] for r in results}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="พิมพ์ข้อความภาษาไทย..."),
    outputs=gr.Label(),
    title="🇹🇭 Thai Sentiment Analysis",
    description="Fine-tuned PhayaThaiBERT on 21k social media messages",
    examples=[
        ["อาหารอร่อยมาก บริการดีเยี่ยม"],
        ["ราคาแพงไป คุณภาพไม่คุ้ม"],
        ["ปกติครับ ไม่มีอะไรพิเศษ"],
    ]
)

demo.launch()