# One-shot job: download a public BERT classifier from Hugging Face and write a
# TensorFlow SavedModel that TF-Serving can load. Runs natively on amd64 and arm64.
FROM python:3.11-slim

WORKDIR /export
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY export_savedmodel.py .

ENTRYPOINT ["python", "export_savedmodel.py"]
