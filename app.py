from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Any

app = FastAPI(title="Shield-82M PII Protection API")

model = None
tokenizer = None
id2label = None

@app.on_event("startup")
def load_model():
    global model, tokenizer, id2label
    model_path = "onnx-community/Shield-82M-ONNX"
    print(f"Загрузка модели {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    session = ort.InferenceSession(model_path)
    print("Модель загружена.")

class TextRequest(BaseModel):
    text: str

@app.post("/redact", response_model=Dict[str, Any])
async def redact_pii(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым.")

    inputs = tokenizer(request.text, return_tensors="np", truncation=True, padding=True)

    outputs = session.run(["logits"], {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
    logits = outputs[0]

    predictions = np.argmax(logits, axis=-1)[0]

    entities = []
    for token, pred_idx in zip(inputs["input_ids"][0], predictions):
        label = id2label[pred_idx]
        if label != "O":  # "O" означает "Outside", т.е. не ПД
            token_str = tokenizer.decode([token])
            entities.append({"token": token_str, "label": label})
    
    masked_text = request.text
    for entity in entities:
        masked_text = masked_text.replace(entity['token'], f"[{entity['label']}]")
    return {"original_text": request.text, "masked_text": masked_text, "entities": entities}