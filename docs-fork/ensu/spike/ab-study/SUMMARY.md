# A/B Test Summary — On-device Wikipedia RAG (Ensu, Pixel 10)

**What we did:** Asked the same 20 factual questions to the on-device model (LFM 2.5)
twice — once with Wikipedia retrieval **off**, once **on** — logging each Q&A, retrieved
passages, and tok/s, then compared the pairs.

**Retrieval setup:**

- **Corpus:** Simple English Wikipedia (`wikimedia/wikipedia`, 2023-11 snapshot),
  **lead-only** (~first 600 chars/article) → **235,368 passages**.
- **Embedding model:** **EmbeddingGemma-300M**, **GGUF Q8_0**
  (`embeddinggemma-300M-Q8_0.gguf`, ~318 MB, from ggml-org), 768-dim.
- **Embedding engine:** **llama.cpp** via the **`llama-cpp-2`** Rust crate (v0.1.144) in
  embedding mode (mean pooling) — the same engine that runs the chat model. Query embedded
  on-device; index built offline with the same model.
- **Index:** 768-dim vectors **int8-quantized** (`vectors.i8`, ~181 MB) + passage metadata;
  brute-force **cosine** top-k with a **0.45 similarity gate**. Loaded via a Rust
  `ensu-retrieval` crate → uniffi → Kotlin. Assets sideloaded to the device.

**How we compared:** Identical questions/phrasing; only the Settings toggle changed. Graded
each answer for factual accuracy against the retrieved Wikipedia article.

| Dimension | OFF (baseline) | ON (RAG) |
|---|---|---|
| **Factual correctness** | 9/20 clean | **~18/20 clean** — fixed 11/11 errors; 1 regression (Mansa Musa, distractor bleed-in) |
| **Tonality** | Confident, authoritative — *confidently wrong* | Same confident tone, now **grounded**; occasionally cited the source |
| **Answer length** | Often long, padded with bullet lists | Similar-to-slightly-shorter; pulled toward the encyclopedic lead |
| **Details** | Specific but frequently fabricated (dates, prizes, places) | Specifics now **accurate** (e.g. Ramanujan 1887, Voynich→Yale, Astana) |
| **Response time / TPS** | ~10.5 tok/s mean | **~6.6 tok/s** (~37% slower) — bigger prefill + memory pressure |

**Outcome:** RAG **substantially improved factual accuracy** at a **~37% speed cost**. The
one failure (Mansa Musa) and the slowdown both have clear fixes: raise the gate to ~0.50 /
cap k=1–2, and mmap the index.

See also: `analysis-comparison.md` (per-question), `analysis-wiki-off.md` (baseline), and
the raw logs `run-wiki-off.log` / `run-wiki-on.log`.
