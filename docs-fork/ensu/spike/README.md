# Ensu retrieval spike

Offline harness to validate Simple-English-Wikipedia retrieval quality before
touching Ensu app code. See [`../retrieval-design.md`](../retrieval-design.md) for
the decisions this encodes (lead-only passages, EmbeddingGemma, 768-dim int8,
similarity-threshold gate).

This is a **build-time research tool**, not shipped code. Nothing here runs
on-device; it produces the index asset the app would later download.

## Setup

```bash
cd docs-fork/ensu/spike
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test (no gating, ~minutes)

EmbeddingGemma is gated on Hugging Face, so validate the *pipeline* first with a
small subset and a non-gated model:

```bash
python build_index.py --model sentence-transformers/all-MiniLM-L6-v2 \
    --limit 3000 --out index-smoke
python search.py "how tall is mount everest" --index index-smoke --k 5
```

`all-MiniLM-L6-v2` is 384-dim and weaker — it's only to prove the plumbing.
Quality judgements should use the real model below.

## Real run (EmbeddingGemma)

1. Accept the license on https://huggingface.co/google/embeddinggemma-300m
2. Authenticate: `pip install huggingface_hub && huggingface-cli login`
3. Build (full Simple Wikipedia ≈ 240k articles):

```bash
python build_index.py --out index            # add --limit 20000 to iterate faster
python search.py --interactive               # eyeball quality, tune --threshold
```

Expected full index: ~240k × 768d int8 ≈ **180 MB** of vectors + `meta.jsonl`.

## Files

| File | Role |
|---|---|
| `build_index.py` | dataset → lead extraction → embed → int8 → `index/` |
| `search.py` | embed query → cosine top-k → threshold-gate preview |
| `requirements.txt` | sentence-transformers, datasets, numpy |

## What we're validating

- Does lead-only retrieval surface the right article for factual queries?
- What similarity threshold cleanly separates relevant hits from noise
  (the `✓`/`·` markers in `search.py` preview the gate)?
- Where does it fail (facts buried past the lead) → informs whether to revisit
  the chunking decision.
