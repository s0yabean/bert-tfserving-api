"""Measure what the 2021 per-request tokenizer load cost, versus loading once.

    python scripts/bench_tokenizer_load.py [--n 200]

"before": every request constructs the tokenizer from the vocab file, then encodes
          (what models/ai.py did, there via pickle.load of a 1.4 MB tokenizer file).
"after":  the tokenizer is constructed once at startup; each request only encodes.

Uses the same public vocab as the API (downloaded once into the Hugging Face cache).
The sample texts are synthetic.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bert_api.config import Settings  # noqa: E402
from bert_api.tokenizer import WordPieceTokenizer, download_vocab  # noqa: E402

TEXTS = [  # synthetic
    "The delivery arrived two days late and nobody answered the phone.",
    "Great service, the team explained every step and finished early.",
    "It was fine. Nothing special, but nothing went wrong either.",
    "I asked for a quote three times before anyone replied.",
]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    s = Settings.from_env()
    vocab = download_vocab(s.hf_tokenizer_id)

    before, after = [], []
    for i in range(args.n):
        t0 = time.perf_counter()
        WordPieceTokenizer(vocab, s.max_seq_length, s.lowercase).encode(TEXTS[i % len(TEXTS)])
        before.append((time.perf_counter() - t0) * 1000)

    tok = WordPieceTokenizer(vocab, s.max_seq_length, s.lowercase)
    for i in range(args.n):
        t0 = time.perf_counter()
        tok.encode(TEXTS[i % len(TEXTS)])
        after.append((time.perf_counter() - t0) * 1000)

    print(f"vocab: {s.hf_tokenizer_id} ({sum(1 for _ in open(vocab, encoding='utf-8'))} tokens), n={args.n}")
    for name, xs in (("load per request", before), ("load once", after)):
        print(f"{name:>17}: median {statistics.median(xs):8.3f} ms   p95 {pct(xs, 0.95):8.3f} ms")
    print(f"median speed-up: {statistics.median(before) / statistics.median(after):.0f}x")


if __name__ == "__main__":
    main()
