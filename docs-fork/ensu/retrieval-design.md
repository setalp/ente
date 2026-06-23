# Ensu Retrieval — Design Note

> Fork-local design for adding **retrieval-augmented answers** to Ensu, starting with
> Wikipedia and generalizing toward a multi-source, on-device knowledge engine (and,
> eventually, RAG over the user's private notes). Companion to [`dev-guide.md`](./dev-guide.md).
>
> Status: **design / pre-implementation.** Decisions still open are tracked at the bottom.

## Goal

Give Ensu the ability to ground its answers in external knowledge and **cite sources**,
rather than relying only on what the small on-device model memorized in its weights.

Long-term target: a "web search"-style feature backed by open datasets
(**Wikipedia, Wikidata, OpenLibrary, …**). Near-term: a basic on-device version over
**Simple English Wikipedia**, built so the same machinery extends to the other sources
and to private documents.

## Framing: this is a knowledge engine, not web search

True web search = live, fresh, arbitrary pages, query leaves the device. What we're
building is a **federated retrieval engine over curated datasets**: frozen snapshots,
structured/semi-structured, bounded coverage — but **private and on-device**. The win is
*verifiable factual grounding + citations*, not recency or open-web breadth. That is a
coherent and differentiated feature for a privacy-first app.

### Why retrieval helps a small model

The model's parametric knowledge is lossy compression — good for gist, unreliable for
specifics (dates, figures, names). Retrieval puts the **exact text in front of it at
answer time** so it copies facts instead of reconstructing them. Value is concentrated in:
factual specifics, hallucination reduction, and **verifiability (shown sources)** — the
last being worthwhile even when accuracy is unchanged.

Caveat already acknowledged: Simple English Wikipedia overlaps heavily with what the model
already knows, so the *knowledge* gain on common topics is modest. Payoff scales with the
corpus — biggest for sources the model does **not** know (full Wikipedia, structured facts,
and especially private notes).

## Architecture

### Pipeline (prototype, inline injection)

```
user message
   │
   ├─(code gate: is retrieval worthwhile?)──no──► normal generation
   │ yes
   ▼
[reformulate query]  (code; raw message for v1, optional LLM rewrite later)
   ▼
Retriever.retrieve(query) ─► { passages[], sources[] }
   ▼
inject top-k passages + titles into the prompt context
   ▼
local model generates ─► answer + rendered source citations
```

No model tool-calling in the prototype — see decision below. The retrieval *decision* and
*routing* live in code, which is reliable on small models; the LLM only consumes the result.

### Core abstraction: `Retriever` + router

Every source implements one interface, so the chat flow never hardcodes Wikipedia:

```ts
interface Retriever {
  retrieve(query: string, k: number): Promise<{ passages: Passage[]; sources: Source[] }>;
}
```

A thin **router** (code, not the LLM) decides whether to retrieve and which retriever(s)
to use. This is the seam that lets us:
- add Wikidata / OpenLibrary later as sibling retrievers,
- add **private notes** as just another retriever,
- and later expose `retrieve()` as a **tool** for the cloud route without changing the
  retrieval implementation — only *who pulls the trigger* moves from code to the model.

Different sources need different retrieval modes (important):

| Source | Shape | Retriever mode |
|---|---|---|
| Wikipedia | unstructured prose | semantic / vector |
| Wikidata | knowledge graph (entity→property→value) | entity linking + structured lookup (**not** vectors) |
| OpenLibrary | catalog (books/authors/ISBN) | keyword/field lookup or API |
| Private notes | unstructured user docs | semantic / vector |

### On-device embedding — the linchpin

Running an embedding model locally is the hardest shared primitive. The Wikipedia path
builds it to embed **queries**; pointed at **documents** it also indexes notes. Query and
corpus must share one embedding model, so one choice covers every vector source.

- Native: llama.cpp embeddings (already in the Rust core).
- Web: `wllama.createEmbedding` (already available in Ensu's browser engine).

## Decisions made

- **On-device first, cloud later.** Prototype is fully local; a cloud route is expected
  eventually (for bigger corpora/models and broader sources) but is out of scope for v1.
- **Inline context injection, not tool calling, for v1.** Small models (LFM 1.6B,
  Gemma 4B) are unreliable at function calling. Keep the search decision/routing in code.
  Move to tool calling when on the cloud route with a capable model.
- **`Retriever` interface + code router** from day one, so Wikipedia work generalizes to
  other sources and to notes, and lifts cleanly into a tool later.
- **Corpus as a dial:** Simple English Wikipedia (v1) → structured sources
  (Wikidata/OpenLibrary) → private notes (highest payoff, on-device mandatory).

## How this extends to private-notes RAG

Reused as-is: query embedding, vector search, chunking, context injection, citation UI,
the `Retriever`/router seam. New work specific to notes:

1. **On-device write-side pipeline** — parse → chunk → embed → store, on the device
   (Wikipedia ships a prebuilt read-only index; notes must be built locally).
2. **Incremental updates** — upsert on edit, remove on delete.
3. **Encryption at rest** — notes are private; reuse Ensu's existing crypto
   (`services/chat/chatKey.ts`, `crypto.ts`). Wikipedia (public) needs none.

Notes is where retrieval is non-marginal: the model has zero parametric coverage of the
user's documents, and on-device is mandatory (private text can't go to a cloud embedder).

## Code landing zones

- Web: new `web/apps/ensu/src/services/retrieval/` (index load, embed, search, router),
  wired into `services/llm/inference.ts` + `services/chat/` prompt assembly; composer
  toggle in `components/chat/ChatComposer.tsx`; feature flag in `services/featureFlags.ts`.
- Native (later): `rust/crates/ensu/retrieval` mirroring it on the Rust core.

## Android wiring plan (native path)

Codebase findings (verified by reading source):

- **Embedding is feasible on the existing engine.** `inference_rs` (crate at
  `rust/crates/ensu/inference/`) is built on `llama-cpp-2` v0.1.144, which **supports
  embeddings** — `LlamaContext::embeddings_seq_ith(i)` + `with_embeddings(true)` /
  `with_pooling_type(..)` on context params. Today the crate exposes only generation; we
  add an `embed()` path. No new runtime needed. (Note: `:rust` Android module also bundles
  `onnxruntime-android` — a fallback embedding option, but llama.cpp keeps one engine.)
- **Kotlin↔Rust via uniffi.** Generated packages `io.ente.labs.inference_rs` and
  `io.ente.labs.ensu_db`. Pattern: core crate → `bindings/uniffi/ensu/<x>` (`#[uniffi::export]`,
  `setup_scaffolding!`) → Kotlin. Adding a crate = core + uniffi binding + workspace member.
- **Prompt-assembly injection point:** `domain/.../store/ChatStoreActions.kt` —
  `sendMessage()` → `startGeneration()` → `buildPrompt()` / `buildHistorySelection()` →
  `llmProvider.generateChat()`. Retrieved context is injected in `buildPrompt()` before the
  `LlmMessage` list is built, within the existing token budget.
- **Toggle + storage:** `AdvancedSettingsDataStore` (`ensu_advanced_settings`) for a
  `rag_enabled` flag; `FilePathManager` for an index dir; model/index download mirrors
  `InferenceRsProvider.downloadLlmModelFiles`. Composer toggle in `app-ui/.../chat/ChatInputBar.kt`.

Build order (each layer compiled before the next; build/test deferred while the full index
bakes to avoid CPU contention):

1. ✅ *written (compile-pending)* **Rust core embed()** — `inference_rs`: `ContextParams.embeddings`
   + `embed(context, texts) -> Vec<Vec<f32>>` (tokenize → decode → `embeddings_seq_ith(0)` → L2-normalize).
2. ✅ *written (compile-pending)* **Retrieval core crate** — `rust/crates/ensu/retrieval`:
   loads manifest.json + vectors.i8 + meta.jsonl; cosine top-k with threshold gate.
3. ✅ *written (compile-pending)* **uniffi bindings** — `embed` exposed on the inference binding;
   new `retrieval` binding (`io.ente.labs.retrieval`, `RetrievalIndex.open/search`).
4. ✅ *written (compile-pending)* **Kotlin provider** — `:domain` `RetrievalProvider`/`RetrievedPassage`;
   `:data` `RustRetrievalProvider` (lazy EmbeddingGemma embed-context + index; query prompt; best-effort).
5. ✅ *written (compile-pending)* **Injection** — `ChatStoreActions.retrieveWikipediaContext()` injects a
   system message before the user turn; `RetrievalProvider` threaded (nullable) through `AppStore`.

**Verified (compile + runtime):** All four Rust crates `cargo check` clean (llama.cpp builds
on macOS/Metal). `embed()` runtime smoke test (`inference/tests/embed.rs`, gated on
`ENSU_EMBED_GGUF`) passes against `embeddinggemma-300M-Q8_0.gguf`: 768-dim unit vectors,
cosine(query, relevant)=0.487 vs (query, unrelated)=−0.035. Confirms `llama-cpp-2` v0.1.144
supports the EmbeddingGemma architecture — no engine bump needed.

**Android build verified.** Toolchain: JDK 17 (Homebrew keg-only; registered via
`~/.gradle/gradle.properties` `org.gradle.java.installations.paths`), Android SDK at
`/opt/homebrew/share/android-commandlinetools`, NDK 27.3.13750724, `local.properties`
`sdk.dir`. `retrieval` wired into `build-rust.sh` CRATES + `codegen/main.rs`; bindings are a
build prerequisite — run `cargo codegen native` (not committed/gradle-generated).
`./gradlew :domain:compileKotlin :data:compileDebugKotlin` **BUILD SUCCESSFUL** — generated
`retrieval.kt`/`inference.kt`, `RustRetrievalProvider`, `ChatStoreActions` injection, and
`AppStore` DI all compile.

**Remaining to activate (default-off today):**
- `AppViewModel`: construct `RustRetrievalProvider(embeddingModelPath, indexDir)` and pass to `AppStore`.
- Provision assets: EmbeddingGemma GGUF + index dir via `FilePathManager` + download.
- Per-conversation toggle: `AdvancedSettingsDataStore` flag + state field + `ChatInputBar` control.
- Full app assemble (`:app-ui:assembleDebug` → triggers `buildRustJni` NDK cross-compile) + on-device test.

## Source selection (v1)

- Corpus: **Simple English Wikipedia** — [`wikimedia/wikipedia` → `20231101.simple`](https://huggingface.co/datasets/wikimedia/wikipedia)
  (raw text, CC-BY-SA; attribution satisfied by showing sources).
- Considered but rejected for v1: [`Cohere/wikipedia-22-12-simple-embeddings`](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings)
  — pre-embedded, but tied to Cohere's cloud model → query embedding would require a
  cloud call, breaking the on-device promise.

## Phasing

1. **Spike** — build the Simple-Wiki lead-only index offline (EmbeddingGemma, 768-dim int8);
   validate retrieval quality and tune the similarity threshold on sample queries.
2. **Web v1** — load index (OPFS), embed query via wllama, cosine top-k, threshold-gate,
   inject + render sources; behind a feature flag.
3. **Native parity** — `rust/crates/ensu/retrieval` (llama.cpp embeddings + cosine search)
   + uniffi bindings + Swift/Kotlin UI, reusing the identical prebuilt index.
4. **Extend** — chunking/quantization tuning; then structured sources (Wikidata/OpenLibrary
   via the router); then private-notes RAG (on-device write-side indexing + encryption).

## Spike results (Simple-Wiki, EmbeddingGemma)

Validated on a 19.5k-article subset (`docs-fork/ensu/spike/`), real EmbeddingGemma-300m,
768-dim int8, lead-only:

- **Index size** ~14 MB / 19.5k → **~170 MB at full 240k** (matches Decision 3 estimate).
- **Retrieval quality** strong: returns the correct article #1 and relevant neighbours
  (e.g. "what causes the seasons" → Season / Axial tilt / Earth's orbit).
- **Cross-lingual confirmed**: a German query ("warum ist der himmel blau") retrieved the
  right English articles — validates the multilingual model choice (Decision 1).
- **Threshold gate**: factual queries score **~0.48–0.58**; conversational chit-chat
  ("thanks that was helpful") scores **~0.26**. → start the gate at **~0.45** (Decision 4),
  tune on the full index.
- **Full index built** (235,368 lead-only passages, EmbeddingGemma 768-dim int8, max_chars
  600): **180.8 MB** vectors + 113 MB meta.jsonl. **Threshold 0.45 confirmed at full scale** —
  factual queries 0.51–0.58 (inject), chit-chat 0.37 and coding requests 0.41 (correctly
  gated out). Cross-lingual (DE→EN) holds. Index lives at `spike/index/` (gitignored).
- **Data caveat**: cleaned `wikimedia/wikipedia` text strips infobox-templated figures
  (e.g. Everest's height is missing from the lead) → reinforces routing exact numbers/dates
  to **Wikidata** rather than prose Wikipedia.

## Open decisions

Being resolved one at a time; this section updates as each is settled.

1. **Embedding model** — ✅ **EmbeddingGemma-300M.** Only option that is both multilingual
   (100+ languages, enables cross-lingual query→corpus) and on-device-optimized; Matryoshka
   dims; reuses the Gemma family. Multilingual matters most for private notes (user's own
   language) and cross-lingual Wikipedia queries.
2. **Index granularity** — ✅ **Lead/abstract only** (one passage per article ≈ **240k
   passages**). Smallest, simplest build (no section parsing); Simple-Wiki articles are short
   so the lead carries most key facts. Revisit toward section/fixed-size chunking in the
   polish phase or when pointing at full Wikipedia, if retrieval misses buried details.
3. **Embedding dimension + quantization** — ✅ **768-dim, int8.** At ~240k lead-only
   passages → ~180MB of vectors (+ ~40-60MB compressed passage text for injection). Full
   embedding fidelity, int8 keeps it on-device. With lead-only each article gets one
   retrieval shot, so we banked the headroom into precision rather than a smaller download.
4. **Retrieval gate** — ✅ **Similarity-threshold gate.** Always run retrieval, but inject
   passages only if the top match's cosine similarity clears a tuned threshold. Nearly free
   (reuses the query embedding + top result), prevents irrelevant passages from derailing
   non-factual turns (greetings/coding/reasoning). Threshold tuned empirically during the spike.
5. **v1 platform scope** — ✅ **Android-first** (user override of the web-first
   recommendation; mobile is the priority target, and it validates the hardest path —
   Rust core + uniffi + Kotlin — first). Web/desktop/iOS follow later reusing the same Rust
   retrieval crate + index. Trade-off accepted: slower first iteration than web, but no
   throwaway web work.
6. **Per-source on-device/online boundary** (relevant once structured sources land) — _open, deferred._
