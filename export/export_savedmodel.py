"""Export a public Hugging Face BERT classifier to a TF-Serving SavedModel.

Writes:
    <out>/<model-name>/<version>/            SavedModel with a `serving_default` signature
    <out>/<model-name>/labels.json           index -> label, read by the API

Signature inputs (int32, shape [batch, seq_len]): input_ids, attention_mask, token_type_ids
Signature output: probabilities (float32, shape [batch, num_labels])

The script is idempotent: if the version directory already has a saved_model.pb it exits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=os.getenv("HF_MODEL_ID", "nlptown/bert-base-multilingual-uncased-sentiment"))
    p.add_argument("--model-name", default=os.getenv("MODEL_NAME", "text_classifier"))
    p.add_argument("--version", default=os.getenv("MODEL_VERSION", "1"))
    p.add_argument("--out", default=os.getenv("EXPORT_DIR", "/models"))
    p.add_argument("--force", action="store_true", help="Re-export even if the version exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.out) / args.model_name
    version_dir = model_dir / str(args.version)

    if (version_dir / "saved_model.pb").exists() and not args.force:
        print(f"SavedModel already present at {version_dir}, skipping export.")
        return 0

    import tensorflow as tf
    from transformers import TFAutoModelForSequenceClassification

    print(f"Downloading {args.model_id} ...")
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model_id)

    spec = {
        name: tf.TensorSpec(shape=[None, None], dtype=tf.int32, name=name)
        for name in ("input_ids", "attention_mask", "token_type_ids")
    }

    @tf.function(input_signature=[spec])
    def serve(inputs):
        logits = model(inputs, training=False).logits
        return {"probabilities": tf.nn.softmax(logits, axis=-1)}

    version_dir.parent.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(version_dir), signatures={"serving_default": serve})

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels = [id2label[i] for i in range(len(id2label))]
    (model_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"Exported {args.model_id} -> {version_dir} with labels {labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
