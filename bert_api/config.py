"""Settings, read once from the environment. No hard-coded hosts."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


@dataclass(frozen=True)
class Settings:
    # Where TF-Serving's REST API lives, e.g. http://tfserving:8501 inside compose.
    tf_serving_url: str = "http://localhost:8501"
    model_name: str = "text_classifier"
    # Hugging Face repo whose vocab.txt the tokenizer is built from. Must match the exported model.
    hf_tokenizer_id: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    lowercase: bool = True
    # Optional JSON list of labels written by the exporter; falls back to label_0..label_n.
    labels_path: str = ""
    max_seq_length: int = 256
    max_text_chars: int = 5000
    connect_timeout_s: float = 2.0
    read_timeout_s: float = 10.0
    cors_origin: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            tf_serving_url=os.getenv("TF_SERVING_URL", d.tf_serving_url).rstrip("/"),
            model_name=os.getenv("MODEL_NAME", d.model_name),
            hf_tokenizer_id=os.getenv("HF_TOKENIZER_ID", d.hf_tokenizer_id),
            lowercase=os.getenv("TOKENIZER_LOWERCASE", "true").lower() in ("1", "true", "yes"),
            labels_path=os.getenv("LABELS_PATH", d.labels_path),
            max_seq_length=_int("MAX_SEQ_LENGTH", d.max_seq_length),
            max_text_chars=_int("MAX_TEXT_CHARS", d.max_text_chars),
            connect_timeout_s=_float("TF_SERVING_CONNECT_TIMEOUT_S", d.connect_timeout_s),
            read_timeout_s=_float("TF_SERVING_READ_TIMEOUT_S", d.read_timeout_s),
            cors_origin=os.getenv("CORS_ORIGIN", d.cors_origin),
            log_level=os.getenv("LOG_LEVEL", d.log_level).upper(),
        )
