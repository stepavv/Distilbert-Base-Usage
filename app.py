from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

app = FastAPI()

# Путь к ONNX-модели
MODEL_PATH = "onnx/model.onnx"

# Загружаем токенизатор (оригинальный из Hugging Face)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Загружаем ONNX Runtime сессию
session = ort.InferenceSession(MODEL_PATH)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    # Токенизация
    tokens = tokenizer(input.text, return_tensors="np", padding=True, truncation=True)
    # ONNX Runtime ожидает numpy arrays
    inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }
    outputs = session.run(["logits"], inputs)
    logits = outputs[0]
    # Softmax для получения вероятностей
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    label = "POSITIVE" if np.argmax(probs) == 1 else "NEGATIVE"
    return {"label": label, "confidence": float(np.max(probs))}

@app.get("/health")
def health():
    return {"status": "ok"}