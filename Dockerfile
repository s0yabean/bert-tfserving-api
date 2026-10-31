FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app
# Dependencies are installed at build time (2021: pip install ran on every container start).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bert_api ./bert_api
COPY wsgi.py .

# Download the public tokenizer vocab into the image so startup does not need the network.
ARG HF_TOKENIZER_ID=nlptown/bert-base-multilingual-uncased-sentiment
ENV HF_TOKENIZER_ID=${HF_TOKENIZER_ID}
RUN python -m bert_api.tokenizer

RUN useradd --create-home --uid 10001 app && chown -R app /opt/hf-cache
USER app
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# --preload builds the tokenizer once in the master; workers share it copy-on-write.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:8000 --workers ${WEB_CONCURRENCY:-2} --timeout 30 --preload --access-logfile - wsgi:app"]
