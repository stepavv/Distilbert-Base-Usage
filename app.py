from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoConfig
import onnxruntime as ort
import numpy as np
from typing import Dict, Any
from huggingface_hub import snapshot_download
from pathlib import Path

app = FastAPI(title="Shield-82M PII Redaction API")

session = None
tokenizer = None
id2label = None


@app.on_event("startup")
def load_model():
    global session, tokenizer, id2label

    model_id = "onnx-community/Shield-82M-ONNX"

    print(f"Loading model: {model_id}")

    # HF cache (без local_files_only — важно!)
    model_dir = snapshot_download(repo_id=model_id)

    print(f"Model cached at: {model_dir}")

    # ONNX file selection (строго model.onnx если есть)
    onnx_files = list(Path(model_dir).rglob("*.onnx"))

    if not onnx_files:
        raise RuntimeError("No ONNX files found")

    print("\nAvailable ONNX files:")
    for f in onnx_files:
        print(f)

    # выбираем model.onnx приоритетно
    onnx_path = None
    for f in onnx_files:
        if f.name == "model.onnx":
            onnx_path = str(f)
            break

    if onnx_path is None:
        onnx_path = str(onnx_files[0])

    print(f"\nUsing ONNX: {onnx_path}")

    # tokenizer + config
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)

    id2label = config.id2label

    print("\nLabels loaded:")
    print(id2label)

    # ONNX session
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )

    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(inp.name, inp.shape, inp.type)

    print("\nModel outputs:")
    for out in session.get_outputs():
        print(out.name, out.shape, out.type)

    print("\nModel loaded successfully")


class TextRequest(BaseModel):
    text: str


@app.post("/redact")
def redact_pii(request: TextRequest):

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    inputs = tokenizer(
        request.text,
        return_tensors="np",
        truncation=True,
        padding=True,
        return_offsets_mapping=True
    )

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    offset_mapping = inputs["offset_mapping"][0]

    ort_inputs = {}

    for inp in session.get_inputs():

        if inp.name == "input_ids":
            ort_inputs[inp.name] = input_ids

        elif inp.name == "attention_mask":
            ort_inputs[inp.name] = attention_mask

        elif inp.name == "token_type_ids":
            ort_inputs[inp.name] = np.zeros_like(input_ids, dtype=np.int64)

    outputs = session.run(None, ort_inputs)

    logits = outputs[0]
    predictions = np.argmax(logits, axis=-1)[0]

    # ===== ENTITY EXTRACTION (FIXED) =====
    entities = []

    current_label = None
    start = None
    end = None

    for pred, offset in zip(predictions, offset_mapping):

        label = id2label.get(int(pred), "O")
        s, e = offset

        if s == e:
            continue

        # no entity
        if label == "O":

            if current_label is not None:
                entities.append({
                    "text": request.text[start:end],
                    "label": current_label,
                    "start": start,
                    "end": end
                })

                current_label = None
                start = None
                end = None

            continue

        # start new entity
        if current_label is None:
            current_label = label
            start = int(s)
            end = int(e)

        # same entity continues
        elif label == current_label:
            end = int(e)

        # entity changed
        else:
            entities.append({
                "text": request.text[start:end],
                "label": current_label,
                "start": start,
                "end": end
            })

            current_label = label
            start = int(s)
            end = int(e)

    # flush last
    if current_label is not None:
        entities.append({
            "text": request.text[start:end],
            "label": current_label,
            "start": start,
            "end": end
        })

    # ===== MASK TEXT =====
    masked = request.text

    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        masked = (
            masked[:ent["start"]] +
            f"[{ent['label']}]" +
            masked[ent["end"]:]
        )

    return {
        "original_text": request.text,
        "masked_text": masked,
        "entities": entities
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": session is not None
    }