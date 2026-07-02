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

## Wikivoyage (second corpus)

`build_wikivoyage.py` builds a **section-chunked** index from the Wikimedia
`enwikivoyage` dump (auto-downloaded, ~123 MB). Unlike Wikipedia (lead-only),
Wikivoyage's value is in the sections (See / Do / Eat / Sleep / Get in) and in
its **listing templates** (`{{see}}`, `{{eat}}`, …), which carry each POI's
name + description — so we chunk per section and extract listings explicitly.
It reuses `build_index.py`'s embed + int8 + index-writing, so the output is the
same format (`search.py` and the Rust `RetrievalIndex` read it unchanged; meta
rows just gain an extra ignored `section` field).

```bash
# fast plumbing smoke (non-gated model, first N articles)
python build_wikivoyage.py --limit 4000 --out index-wv-smoke \
    --model sentence-transformers/all-MiniLM-L6-v2
# full curated build (EmbeddingGemma; destinations-only, ~26k articles)
python build_wikivoyage.py --out index-wikivoyage
python search.py --index index-wikivoyage --interactive
```

Curated build (destinations-only, `max_chars=900`, 768-dim int8): **26,145
articles → 357,890 chunks → ~441 MB** (275 MB vectors + 167 MB meta). Article-type
filtering (`keep_article`) drops region/country/topic/itinerary/phrasebook/
disambiguation pages; the full un-curated corpus is ~609k chunks / ~700 MB.
See `../retrieval-design.md` → "Additional corpora".

## Files

| File | Role |
|---|---|
| `build_index.py` | Wikipedia: dataset → lead extraction → embed → int8 → `index/`; shared `shard_embed`/`write_index` helpers |
| `build_wikivoyage.py` | Wikivoyage: dump → section chunking + listing extraction → `index-wikivoyage/` |
| `search.py` | embed query → cosine top-k → threshold-gate preview (any index dir) |
| `requirements.txt` | sentence-transformers, datasets, numpy, mwparserfromhell |

## What we're validating

- Does lead-only retrieval surface the right article for factual queries?
- What similarity threshold cleanly separates relevant hits from noise
  (the `✓`/`·` markers in `search.py` preview the gate)?
- Where does it fail (facts buried past the lead) → informs whether to revisit
  the chunking decision.
