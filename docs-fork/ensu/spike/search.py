#!/usr/bin/env python3
"""Query the spike index: embed a query, cosine top-k over the int8 vectors.

Mirrors the on-device query path (embed query -> cosine search -> threshold
gate) so we can eyeball retrieval quality and tune the similarity threshold
(Decision 4 in ../retrieval-design.md) before touching app code.

Usage:
  python search.py "how tall is mount everest" --k 5
  python search.py --interactive
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_index(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    vecs = np.load(path / "vectors.int8.npy")  # int8 (count, dim)
    meta = [json.loads(l) for l in (path / "meta.jsonl").read_text().splitlines()]
    # Dequantize to unit float32 once (transient ~4x int8 size; fine for a spike).
    mat = vecs.astype(np.float32) / manifest["scale"]
    return manifest, mat, meta


def search(model, manifest, mat, meta, query, k, has_prompts):
    kw = dict(normalize_embeddings=True, convert_to_numpy=True)
    if has_prompts:
        qv = model.encode([query], prompt_name="query", **kw)[0]
    else:
        qv = model.encode([query], **kw)[0]
    scores = mat @ qv.astype(np.float32)  # cosine (both unit-normalized)
    top = np.argpartition(-scores, range(min(k, len(scores))))[:k]
    top = top[np.argsort(-scores[top])]
    return [(float(scores[i]), meta[i]) for i in top]


def fmt(results, threshold):
    out = []
    for score, m in results:
        gate = "✓" if score >= threshold else "·"
        snippet = m["text"].replace("\n", " ")[:160]
        out.append(f"  {gate} {score:.3f}  {m['title']}\n      {snippet}…")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="?", help="Query string (omit with --interactive)")
    ap.add_argument("--index", type=Path, default=Path("index"))
    ap.add_argument("--model", default=None,
                    help="Override; defaults to the model recorded in manifest.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Similarity-gate preview: mark hits below this with '·'")
    ap.add_argument("--device", default=None)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    manifest, mat, meta = load_index(args.index)
    model_name = args.model or manifest["model"]
    print(f"index: {manifest['count']:,} x {manifest['dim']}d  model: {model_name}",
          flush=True)
    model = SentenceTransformer(model_name, device=args.device)
    has_prompts = {"query", "document"} <= set(getattr(model, "prompts", {}) or {})

    if args.interactive:
        print("Enter queries (blank to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            res = search(model, manifest, mat, meta, q, args.k, has_prompts)
            print(fmt(res, args.threshold))
        return 0

    if not args.query:
        ap.error("provide a query or use --interactive")
    res = search(model, manifest, mat, meta, args.query, args.k, has_prompts)
    print(fmt(res, args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
