"""Thin client for TF-Serving's REST predict API, with timeouts and typed errors."""
from __future__ import annotations

from typing import Protocol, Sequence

import requests

from .tokenizer import Features


class ServingError(Exception):
    """TF-Serving answered, but not with usable predictions."""


class ServingUnavailable(ServingError):
    """TF-Serving could not be reached or did not answer in time."""


class PredictionClient(Protocol):
    def predict(self, features: Sequence[Features]) -> list[list[float]]: ...

    def is_ready(self) -> bool: ...


class TFServingClient:
    def __init__(self, base_url: str, model_name: str, connect_timeout_s: float = 2.0, read_timeout_s: float = 10.0,
                 session: requests.Session | None = None):
        self.predict_url = f"{base_url}/v1/models/{model_name}:predict"
        self.status_url = f"{base_url}/v1/models/{model_name}"
        self.timeout = (connect_timeout_s, read_timeout_s)
        # One pooled session per process: reuses TCP connections to TF-Serving.
        self.session = session or requests.Session()

    def predict(self, features: Sequence[Features]) -> list[list[float]]:
        # TF-Serving needs every instance in a batch to have the same length, so pad to the
        # longest text in this request (not to a fixed 256 as in 2021). Padding has mask 0.
        width = max(f.num_tokens for f in features)

        def pad(xs: list[int]) -> list[int]:
            return xs + [0] * (width - len(xs))

        body = {
            "signature_name": "serving_default",
            "instances": [
                {"input_ids": pad(f.input_ids), "attention_mask": pad(f.attention_mask),
                 "token_type_ids": pad(f.token_type_ids)}
                for f in features
            ],
        }
        try:
            resp = self.session.post(self.predict_url, json=body, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise ServingUnavailable(type(exc).__name__) from exc

        if resp.status_code != 200:
            # TF-Serving puts a short reason in {"error": ...}; it never contains the input text.
            try:
                reason = str(resp.json().get("error", ""))[:200]
            except ValueError:
                reason = ""
            raise ServingError(f"TF-Serving returned {resp.status_code}: {reason}")

        try:
            preds = resp.json()["predictions"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ServingError("TF-Serving response had no 'predictions'") from exc

        # Single named output -> list of rows; multiple outputs -> list of dicts.
        rows = [p["probabilities"] if isinstance(p, dict) else p for p in preds]
        if len(rows) != len(features):
            raise ServingError("prediction count does not match input count")
        return [[float(x) for x in row] for row in rows]

    def is_ready(self) -> bool:
        try:
            resp = self.session.get(self.status_url, timeout=self.timeout)
            states = [v.get("state") for v in resp.json().get("model_version_status", [])]
        except (requests.RequestException, ValueError):
            return False
        return resp.status_code == 200 and "AVAILABLE" in states
