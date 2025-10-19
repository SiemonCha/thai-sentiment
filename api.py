from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List

app = FastAPI(title="Thai Sentiment API", version="1.0")

# Load model once at startup
model = AutoModelForSequenceClassification.from_pretrained("SiemonCha/thai-sentiment-phayabert")
tokenizer = AutoTokenizer.from_pretrained("SiemonCha/thai-sentiment-phayabert")
LABELS = {0: "positive", 1: "neutral", 2: "negative"}

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict

@app.get("/")
def root():
    return {"message": "Thai Sentiment API", "docs": "/docs"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    try:
        inputs = tokenizer(input.text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            pred = torch.argmax(probs).item()
        
        return {
            "text": input.text,
            "sentiment": LABELS[pred],
            "confidence": float(probs[pred]),
            "scores": {LABELS[i]: float(probs[i]) for i in range(3)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(input: BatchInput):
    return [predict(TextInput(text=text)) for text in input.texts]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)