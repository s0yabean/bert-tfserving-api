"""Fixtures: a real WordPiece tokenizer over a tiny synthetic vocab, and a fake TF-Serving client.

No network, no model, no TF-Serving needed.
"""
from __future__ import annotations

import pytest

from bert_api import create_app
from bert_api.config import Settings
from bert_api.serving import ServingError, ServingUnavailable
from bert_api.tokenizer import WordPieceTokenizer

# Synthetic vocab: BERT special tokens plus a handful of words.
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "service", "was", "great", "slow", "and", "##ly", "."]


@pytest.fixture
def vocab_file(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(VOCAB) + "\n")
    return str(p)


@pytest.fixture
def tokenizer(vocab_file):
    return WordPieceTokenizer(vocab_file, max_seq_length=8)


class FakeClient:
    """Stands in for TFServingClient. Records calls; can be told to fail."""

    def __init__(self, num_labels: int = 3, fail: Exception | None = None, ready: bool = True):
        self.num_labels = num_labels
        self.fail = fail
        self.ready = ready
        self.calls: list = []

    def predict(self, features):
        self.calls.append(features)
        if self.fail:
            raise self.fail
        # Deterministic: the last label wins, with probability mass growing by index.
        total = sum(range(1, self.num_labels + 1))
        return [[(i + 1) / total for i in range(self.num_labels)] for _ in features]

    def is_ready(self):
        return self.ready


@pytest.fixture
def settings():
    return Settings(max_seq_length=8, max_text_chars=100, cors_origin="")


@pytest.fixture
def make_client(settings, tokenizer):
    def _make(client=None, **overrides):
        s = Settings(**{**settings.__dict__, **overrides})
        fake = client or FakeClient()
        app = create_app(settings=s, tokenizer=tokenizer, client=fake)
        app.testing = True
        return app.test_client(), fake

    return _make


__all__ = ["FakeClient", "ServingError", "ServingUnavailable"]
