from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx import export
from pathlib import Path

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Экспорт в ONNX
onnx_path = Path("model.onnx")
export(model, tokenizer, onnx_path, opset=14)

print("✅ model.onnx сохранён через transformers.onnx")