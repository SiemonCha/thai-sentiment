from transformers import pipeline
import time

models = {
    "PhayaThaiBERT (ours)": "SiemonCha/thai-sentiment-phayabert",
    "WangchanBERTa": "airesearch/wangchanberta-base-att-spm-uncased",
}

test_texts = [
    "อาหารอร่อยมาก บริการดีเยี่ยม",
    "ราคาแพงไป คุณภาพไม่คุ้ม",
    "ปกติครับ ไม่มีอะไรพิเศษ"
]

print("🏁 Model Benchmark\n")
for name, model_id in models.items():
    try:
        classifier = pipeline("text-classification", model=model_id)
        
        # Speed test
        start = time.time()
        for text in test_texts * 10:
            classifier(text)
        avg_time = (time.time() - start) / (len(test_texts) * 10)
        
        print(f"{name}")
        print(f"  Speed: {avg_time*1000:.1f}ms per prediction")
        print()
    except:
        print(f"{name}: Failed to load\n")