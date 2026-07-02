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


def make_encoder(model_name: str, device: str | None, fp16: bool):
    """Return (encode_docs, dim).

    Uses sentence-transformers encode with the model's "document" retrieval
    prompt when present (EmbeddingGemma registers it); falls back to plain
    encode otherwise (e.g. MiniLM for a no-gating smoke test). fp16 roughly
    halves embedding time on MPS with no meaningful retrieval-quality loss.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    dtype = torch.float16 if fp16 else None
    model = SentenceTransformer(model_name, device=device,
                                model_kwargs={"torch_dtype": dtype} if dtype else {})
    dim = model.get_sentence_embedding_dimension()
    has_prompts = {"query", "document"} <= set(getattr(model, "prompts", {}) or {})

    def encode_docs(texts, batch_size):
        kw = dict(batch_size=batch_size, show_progress_bar=False,
                  normalize_embeddings=True, convert_to_numpy=True)
        if has_prompts:
            return model.encode(texts, prompt_name="document", **kw)
        return model.encode(texts, **kw)

    return encode_docs, dim


def quantize_int8(vecs: np.ndarray) -> np.ndarray:
    """Unit vectors (components in [-1,1]) -> int8 via fixed scale 127.

    nan_to_num guards the astype(int8) cast: a non-finite component would cast to
    a platform-defined garbage int8 (0 on arm64) and silently corrupt the row.
    shard_embed already hard-fails on non-finite output; this is belt-and-suspenders.
    """
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(vecs * 127.0), -127, 127).astype(np.int8)


def shard_embed(out: Path, docs, model_name: str, device, fp16: bool,
                batch_size: int, shard_size: int):
    """Sharded, resumable embedding -> merged int8 (count, dim) array.

    Each shard's int8 vectors are written to out/parts/ immediately, so progress
    is visible and a restart skips finished shards. The model is loaded lazily,
    only if some shard is missing. Shared by build_index.py (lead-only) and
    build_wikivoyage.py (section-chunk) so both encode/quantize identically.
    Returns (q, dim).
    """
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    total = len(docs)
    n_shards = (total + shard_size - 1) // shard_size
    encode_docs = dim = None
    print(f"[embed] {model_name} (fp16={fp16}) in {n_shards} shards "
          f"of {shard_size} ...", flush=True)
    for s in range(n_shards):
        part = parts / f"vec_{s:04d}.npy"
        if part.exists():
            print(f"      shard {s+1}/{n_shards}: cached, skipping.", flush=True)
            continue
        if encode_docs is None:  # lazy model load
            encode_docs, dim = make_encoder(model_name, device, fp16)
        lo, hi = s * shard_size, min((s + 1) * shard_size, total)
        vecs = encode_docs(docs[lo:hi], batch_size).astype(np.float32)
        # Fail loudly on non-finite embeddings. EmbeddingGemma in fp16 overflows
        # to all-NaN on MPS (build with --no-fp16 / fp32); catching it here beats
        # silently baking a zero-vector index that never clears the retrieval gate.
        finite_rows = np.isfinite(vecs).all(axis=1)
        if not finite_rows.all():
            bad = int((~finite_rows).sum())
            raise RuntimeError(
                f"shard {s}: {bad}/{len(vecs)} embeddings non-finite "
                f"(NaN/inf). If using EmbeddingGemma, rebuild in fp32 (--no-fp16)."
            )
        np.save(part, quantize_int8(vecs))
        print(f"      shard {s+1}/{n_shards}: embedded {hi:,}/{total:,} "
              f"({100*hi//total}%).", flush=True)
    q = np.concatenate([np.load(parts / f"vec_{s:04d}.npy") for s in range(n_shards)])
    if dim is None:  # fully resumed from cache; infer dim from shards
        dim = q.shape[1]
    return q, dim


def write_index(out: Path, rows, q: np.ndarray, dim: int, extra: dict) -> None:
    """Write the on-device index format: vectors.i8 + meta.jsonl + manifest.json.

    `rows` are the meta dicts (aligned to q); `extra` supplies dataset-specific
    manifest fields (dataset, granularity, ...). Computed count/dim/scale/quant
    always win over `extra`. `vectors.int8.npy` is also written for search.py.
    """
    np.save(out / "vectors.int8.npy", q)
    # Raw int8 (row-major count*dim) for the on-device Rust retrieval crate,
    # which reads manifest.json + vectors.i8 + meta.jsonl.
    q.tofile(out / "vectors.i8")
    with (out / "meta.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {**extra, "dim": int(dim), "count": len(rows),
                "quant": "int8", "scale": 127}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done. {len(rows):,} vectors x {dim}d int8 = {q.nbytes/1e6:.1f} MB "
          f"(+ meta.jsonl).", flush=True)


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
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=10000,
                    help="Passages per checkpoint shard (resumable, observable).")
    ap.add_argument("--fp16", action="store_true", default=False,
                    help="Run the model in float16 (~2x faster on MPS). OFF by default: "
                         "EmbeddingGemma overflows to all-NaN in fp16 on MPS. Only enable "
                         "for fp16-safe models (e.g. the MiniLM smoke).")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
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
    total = len(rows)
    print(f"      kept {total:,} passages (dropped stubs).", flush=True)

    q, dim = shard_embed(args.out, docs, args.model, args.device, args.fp16,
                         args.batch_size, args.shard_size)
    print(f"[4/4] merging shards -> {args.out}/ ...", flush=True)
    write_index(args.out, rows, q, dim, {
        "model": args.model, "fp16": args.fp16,
        "dataset": f"{args.dataset}:{args.config}",
        "granularity": "lead-only", "max_chars": args.max_chars,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
