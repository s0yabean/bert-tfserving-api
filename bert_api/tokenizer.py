"""Tokenizer: built once at startup from a public Hugging Face vocab.

The 2021 version unpickled a tokenizer file from disk on every request. Here the
vocab is downloaded (and cached) once, the tokenizer is constructed once, and every
request only pays for `encode`.

Run `python -m bert_api.tokenizer` to pre-download the vocab (used in the Dockerfile).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tokenizers import BertWordPieceTokenizer


@dataclass(frozen=True)
class Features:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)


class TextTokenizer(Protocol):
    def encode(self, text: str) -> Features: ...


class WordPieceTokenizer:
    """BERT WordPiece tokenizer with truncation to max_seq_length ([CLS] ... [SEP])."""

    def __init__(self, vocab_path: str, max_seq_length: int = 256, lowercase: bool = True):
        self._tok = BertWordPieceTokenizer.from_file(vocab_path, lowercase=lowercase)
        self._tok.enable_truncation(max_length=max_seq_length)

    def encode(self, text: str) -> Features:
        enc = self._tok.encode(text)
        return Features(
            input_ids=list(enc.ids),
            attention_mask=list(enc.attention_mask),
            token_type_ids=list(enc.type_ids),
        )


def download_vocab(hf_tokenizer_id: str) -> str:
    """Return a local path to the repo's vocab.txt, downloading it on first use."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=hf_tokenizer_id, filename="vocab.txt")


def load_tokenizer(hf_tokenizer_id: str, max_seq_length: int, lowercase: bool = True) -> WordPieceTokenizer:
    return WordPieceTokenizer(download_vocab(hf_tokenizer_id), max_seq_length=max_seq_length, lowercase=lowercase)


if __name__ == "__main__":
    from .config import Settings

    s = Settings.from_env()
    print(download_vocab(s.hf_tokenizer_id))
