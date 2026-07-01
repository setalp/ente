# A/B Study — Comparison (Wikipedia context OFF vs ON)

On-device LFM 2.5, Pixel 10. Same 20 questions. Raw: `run-wiki-off.log`, `run-wiki-on.log`.

## Headline

- **Accuracy: RAG fixed all 11 baseline factual errors.** Off = 9/20 clean; On ≈ 18/20 clean.
- **One regression introduced by retrieval** (Mansa Musa) — a borderline distractor passage contaminated the answer.
- **Speed cost: ~6.6 tok/s on vs ~10.5 off** (~37% slower), worst on short answers (bigger prefill from injected passages).

## Per-question

| # | Question | OFF | ON | Change |
|---|---|---|---|---|
| 1 | Kazakhstan capital | ❌ "Nur-Sultan" | ✅ **Astana** | **FIXED** (some fabricated pop/date noise remains) |
| 2 | Myanmar capital | ⚠️ "moved from Bagan 2017" | ✅ "from Yangon, 2005" | **FIXED** backstory |
| 3 | Turkey capital | ✅ Ankara | ✅ Ankara | same |
| 4 | Australia capital | ✅ Canberra | ✅ Canberra | same |
| 5 | Mansa Musa | ✅~ mostly right | ⚠️ **contaminated** | **REGRESSED** — "studied in Europe 1908 / Zagazig" bled in from **Salama Moussa** (0.459, barely over gate) |
| 6 | Hypatia | ❌ father "Aristarchus of Samos" | ✅ no father error, matches lead | **FIXED** |
| 7 | Emmy Noether | ✅ | ✅ + correct dates (1882–1935) | improved |
| 8 | Ramanujan | ❌ born 1890, fake "Tait Prize" | ✅ born **1887**, fabrications gone | **FIXED** |
| 9 | Grace Hopper | ❌ hallucinated ARPANET | ✅ accurate, no ARPANET | **FIXED** |
| 10 | Antikythera | ❌ "University of Athens" | ✅ avoids the wrong-museum claim | **FIXED** |
| 11 | Treaty of Tordesillas | ❌ city in "Portugal" | ✅ correct (minor Westphalia tail) | **FIXED** |
| 12 | Defenestration of Prague | ❌❌ wrong date/leader/place | ✅ correct 1618 names + event | **FIXED** (minor 1419 slip) |
| 13 | Rosetta Stone | ✅ | ✅ + British Museum | improved |
| 14 | Voynich manuscript | ❌❌ "2 leaves / British Library / Vienna" | ✅ **240pp / Yale / Italy** | **FIXED** (strong) |
| 15 | Tardigrade | ✅ | ✅ | same |
| 16 | Dunning–Kruger | ✅ | ✅ | same |
| 17 | Sapir–Whorf | ✅ | ⚠️ new error: "Whorf's 1942 *The Language Instinct*" (that's Pinker, 1994) | model padding, not from retrieval |
| 18 | Atacama Desert | ❌ "Ecuador / 10,000 km²" | ✅ Chile/Peru/Argentina | **FIXED** |
| 19 | Strait of Malacca | ❌ "155 km" | ✅ "~800 km" | **FIXED** (minor sea-name quibble) |
| 20 | Largest planet | ✅ Jupiter | ✅ + mass 318× Earth | improved |

**Errors fixed by RAG: 11/11.** **New errors: 1 clear (Mansa Musa), 1 padding (Sapir–Whorf).**

## The one regression: Mansa Musa
The gate (0.45) admitted two weak passages: **Salama Moussa (0.459)** — an Egyptian writer — and **Mansa Sakura (0.451)**. The model fused Salama Moussa's biography into Mansa Musa's ("studied in Europe (1908)", "Zagazig"). Classic borderline-distractor failure: the top hit (Mansa Musa, 0.658) was great, but the near-gate extras hurt.

## Speed
| | OFF | ON |
|---|---|---|
| mean tok/s | ~10.5 | **6.6** |
| short answers | 5–7 | 2.6–4.9 |
| long answers | 11–14.5 | 7–9.9 |

Two compounding causes: injected passages enlarge the **prefill** (most visible on short answers), and the embedding model + index add **memory pressure**.

## Recommendations
1. **Raise the gate to ~0.50** (or only keep passages within ~0.08 of the top hit) — directly kills the Mansa Musa contamination; from the baseline data, real matches sit at 0.55–0.70 while distractors clustered at 0.45–0.47.
2. **Reduce k to 1–2** — the top passage carries the answer; extras mostly add prefill cost + contamination risk.
3. Revisit the **mmap index** fix to recover tok/s (memory half of the slowdown).
