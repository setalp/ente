#!/usr/bin/env python3
"""Offline index builder for the Ensu Wikipedia-retrieval spike.

Pulls Simple English Wikipedia, extracts the lead paragraph of each article,
embeds it with an on-device-class embedding model (EmbeddingGemma by default),
int8-quantizes the unit vectors, and writes a flat index + metadata.

This is a *build-time* tool — it runs once on a workstation, not on-device.
The on-device app ships the resulting index as a downloadable asset.

Outputs (in --out dir):
  vectors.int8.npy   (count, dim) int8   unit vectors * 127
  meta.jsonl         one {id,title,url,text} per row, aligned to vectors
  manifest.json      model, dim, count, quant scale, dataset

See README.md. Decisions encoded here: lead-only passages, 768-dim int8,
EmbeddingGemma (Decisions 1-3 in ../retrieval-design.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def extract_lead(text: str, max_chars: int) -> str:
    """First paragraph(s) of an article up to a char budget.

    The wikimedia/wikipedia `text` field is cleaned plain text with paragraphs
    separated by blank lines and section structure flattened, so 'lead' here is
    a pragmatic char-budgeted prefix of the leading paragraphs.
    """
    lead = ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if lead and len(lead) + len(para) + 1 > max_chars:
            break
        lead = para if not lead else f"{lead}\n{para}"
        if len(lead) >= max_chars:
            break
    return lead[:max_chars].strip()


def make_encoders(model_name: str, device: str | None):
    """Return (encode_docs, encode_query) closures.

    Uses sentence-transformers encode_document/encode_query when the model
    registers retrieval prompts (EmbeddingGemma does); falls back to plain
    encode otherwise (e.g. MiniLM for a no-gating smoke test).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    has_prompts = bool(getattr(model, "prompts", None)) and {
        "query",
        "document",
    } <= set(getattr(model, "prompts", {}) or {})

    def encode_docs(texts, batch_size):
        kw = dict(batch_size=batch_size, show_progress_bar=True,
                  normalize_embeddings=True, convert_to_numpy=True)
        if has_prompts:
            return model.encode(texts, prompt_name="document", **kw)
        return model.encode(texts, **kw)

    def encode_query(text):
        kw = dict(normalize_embeddings=True, convert_to_numpy=True)
        if has_prompts:
            return model.encode([text], prompt_name="query", **kw)[0]
        return model.encode([text], **kw)[0]

    return encode_docs, encode_query, dim, model_name


def quantize_int8(vecs: np.ndarray) -> np.ndarray:
    """Unit vectors (components in [-1,1]) -> int8 via fixed scale 127."""
    return np.clip(np.round(vecs * 127.0), -127, 127).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("index"))
    ap.add_argument("--model", default="google/embeddinggemma-300m",
                    help="HF model id. Gated; use a non-gated model to smoke-test.")
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.simple")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N articles (0 = all).")
    ap.add_argument("--max-chars", type=int, default=1200,
                    help="Char budget for the lead passage.")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="Skip stubs whose lead is shorter than this.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"[1/4] loading {args.dataset}:{args.config} ...", flush=True)
    split = "train" if not args.limit else f"train[:{args.limit}]"
    ds = load_dataset(args.dataset, args.config, split=split)

    print(f"[2/4] extracting leads from {len(ds):,} articles ...", flush=True)
    rows, docs = [], []
    for r in ds:
        lead = extract_lead(r["text"], args.max_chars)
        if len(lead) < args.min_chars:
            continue
        rows.append({"id": r["id"], "title": r["title"], "url": r["url"], "text": lead})
        # Prepend the title so it influences the embedding (cheap title boost).
        docs.append(f"{r['title']}. {lead}")
    print(f"      kept {len(rows):,} passages (dropped stubs).", flush=True)

    print(f"[3/4] embedding with {args.model} ...", flush=True)
    encode_docs, _, dim, model_name = make_encoders(args.model, args.device)
    vecs = encode_docs(docs, args.batch_size).astype(np.float32)
    q = quantize_int8(vecs)

    print(f"[4/4] writing index to {args.out}/ ...", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "vectors.int8.npy", q)
    with (args.out / "meta.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "model": model_name,
        "dim": int(dim),
        "count": len(rows),
        "quant": "int8",
        "scale": 127,
        "dataset": f"{args.dataset}:{args.config}",
        "granularity": "lead-only",
        "max_chars": args.max_chars,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    mb = q.nbytes / 1e6
    print(f"done. {len(rows):,} vectors x {dim}d int8 = {mb:.1f} MB "
          f"(+ meta.jsonl).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
