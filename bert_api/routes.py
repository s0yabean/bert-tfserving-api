"""HTTP routes: /v1/predict, /healthz, /readyz."""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, g, jsonify, request

from .serving import ServingError, ServingUnavailable

bp = Blueprint("api", __name__)
log = logging.getLogger("bert_api")

MAX_BATCH = 32


def _error(status: int, code: str, detail: str):
    return jsonify(error=code, detail=detail), status


def _parse_texts():
    """Return (texts, None) or (None, error_response)."""
    settings = current_app.extensions["settings"]
    if not request.is_json:
        return None, _error(415, "unsupported_media_type", "send Content-Type: application/json")
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return None, _error(400, "invalid_json", "body must be a JSON object")

    if "texts" in body:
        texts = body["texts"]
        if not isinstance(texts, list) or not texts:
            return None, _error(400, "invalid_texts", "'texts' must be a non-empty list of strings")
        if len(texts) > MAX_BATCH:
            return None, _error(413, "batch_too_large", f"at most {MAX_BATCH} texts per request")
    elif "text" in body:
        texts = [body["text"]]
    else:
        return None, _error(400, "missing_text", "provide 'text' (string) or 'texts' (list of strings)")

    for t in texts:
        if not isinstance(t, str) or not t.strip():
            return None, _error(400, "invalid_text", "each text must be a non-empty string")
        if len(t) > settings.max_text_chars:
            return None, _error(413, "text_too_long", f"each text must be at most {settings.max_text_chars} characters")
    return texts, None


@bp.post("/v1/predict")
def predict():
    texts, err = _parse_texts()
    if err:
        return err
    g.n_texts, g.n_chars = len(texts), sum(len(t) for t in texts)

    tokenizer = current_app.extensions["tokenizer"]
    client = current_app.extensions["client"]
    features = [tokenizer.encode(t) for t in texts]

    try:
        rows = client.predict(features)
    except ServingUnavailable as exc:
        log.warning("tf-serving unavailable id=%s reason=%s", g.request_id, exc)
        return _error(503, "model_unavailable", "model server did not respond; retry later")
    except ServingError as exc:
        log.error("tf-serving error id=%s reason=%s", g.request_id, exc)
        return _error(502, "model_error", "model server returned an unusable response")

    labels = current_app.extensions["labels"]
    results = []
    for row, feat in zip(rows, features):
        names = labels if labels and len(labels) == len(row) else [f"label_{i}" for i in range(len(row))]
        best = max(range(len(row)), key=row.__getitem__)
        results.append({
            "label": names[best],
            "scores": {n: round(p, 4) for n, p in zip(names, row)},
            "num_tokens": feat.num_tokens,
        })
    # A prediction creates nothing on the server: 200, not 201.
    return jsonify(model=current_app.extensions["settings"].model_name, results=results), 200


@bp.get("/healthz")
def healthz():
    """Liveness: the process is up and the tokenizer is loaded."""
    return jsonify(status="ok"), 200


@bp.get("/readyz")
def readyz():
    """Readiness: TF-Serving reports the model version AVAILABLE."""
    if current_app.extensions["client"].is_ready():
        return jsonify(status="ready"), 200
    return jsonify(status="not_ready", detail="model server not available"), 503
