from optimum.exporters.onnx import main_export

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
main_export(model_id, output="onnx/", task="text-classification")