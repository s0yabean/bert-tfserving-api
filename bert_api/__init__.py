"""Flask API in front of TF-Serving for a BERT text classifier."""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from .config import Settings
from .serving import PredictionClient, TFServingClient
from .tokenizer import TextTokenizer, load_tokenizer

log = logging.getLogger("bert_api")


def _load_labels(path: str) -> list[str] | None:
    if path and Path(path).is_file():
        return list(json.loads(Path(path).read_text()))
    return None


def create_app(settings: Settings | None = None, tokenizer: TextTokenizer | None = None,
               client: PredictionClient | None = None) -> Flask:
    """Build the app. Tokenizer and client are created ONCE here, not per request.

    Tests pass a fake tokenizer and client; production builds the real ones from env.
    """
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    app = Flask(__name__)
    # Cap the raw body too, so a huge payload is rejected before JSON parsing.
    app.config["MAX_CONTENT_LENGTH"] = settings.max_text_chars * 4 * 32 + 4096

    t0 = time.perf_counter()
    app.extensions["settings"] = settings
    app.extensions["tokenizer"] = tokenizer or load_tokenizer(
        settings.hf_tokenizer_id, settings.max_seq_length, settings.lowercase)
    app.extensions["client"] = client or TFServingClient(
        settings.tf_serving_url, settings.model_name, settings.connect_timeout_s, settings.read_timeout_s)
    app.extensions["labels"] = _load_labels(settings.labels_path)
    log.info("startup tokenizer=%s model=%s ready_in_ms=%.0f", settings.hf_tokenizer_id, settings.model_name,
             (time.perf_counter() - t0) * 1000)

    @app.before_request
    def _start_timer():
        g.t0 = time.perf_counter()
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        g.n_texts = 0
        g.n_chars = 0

    @app.after_request
    def _log_and_headers(resp):
        resp.headers["X-Request-ID"] = g.get("request_id", "")
        if settings.cors_origin:
            resp.headers["Access-Control-Allow-Origin"] = settings.cors_origin
        # Metadata only. The request text is never logged.
        log.info("request id=%s method=%s path=%s status=%s n_texts=%s n_chars=%s latency_ms=%.1f",
                 g.get("request_id"), request.method, request.path, resp.status_code,
                 g.get("n_texts", 0), g.get("n_chars", 0), (time.perf_counter() - g.get("t0", time.perf_counter())) * 1000)
        return resp

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        return jsonify(error=exc.name.lower().replace(" ", "_"), detail=exc.description), exc.code

    from .routes import bp

    app.register_blueprint(bp)
    return app
