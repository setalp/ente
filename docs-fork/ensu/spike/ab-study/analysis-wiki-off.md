# A/B Study — Baseline (Wikipedia context OFF)

Model: LFM 2.5 (on-device, Pixel 10). 20 questions, `rag=false`. Raw: `run-wiki-off.log`.

## Accuracy: ~11/20 have notable factual errors

| # | Question | Verdict | Error (what RAG should fix) |
|---|---|---|---|
| 1 | Kazakhstan capital | ⚠️ outdated | "Nur-Sultan" — renamed back to **Astana** in 2022 |
| 2 | Myanmar capital | ⚠️ partial | Capital right (Naypyidaw) but invented backstory: "moved from **Bagan** in **2017**" (was Yangon, ~2005) |
| 3 | Turkey capital | ✅ | Ankara |
| 4 | Australia capital | ✅ | Canberra |
| 5 | Mansa Musa | ✅~ | Mostly right; "treasury covered the entire land" embellishment |
| 6 | Hypatia | ⚠️ wrong | Father = "**Aristarchus of Samos**" → actually **Theon of Alexandria** |
| 7 | Emmy Noether | ✅ | Correct |
| 8 | Ramanujan | ⚠️⚠️ | Born "**1890**" (→1887); "son of a farmer and a **Hindu widow**"; fabricated "**G.B. Tait Prize**" |
| 9 | Grace Hopper | ⚠️ | Hallucinated "key figure in **ARPANET**"; dubious "first female rear admiral" |
| 10 | Antikythera mechanism | ⚠️ | Housed at "**University of Athens**" → **National Archaeological Museum** |
| 11 | Treaty of Tordesillas | ⚠️ | "city of Tordesillas, **Portugal**" → Tordesillas is in **Spain** |
| 12 | Defenestration of Prague | ⚠️⚠️ | Wrong date ("May 9" → May 23); wrong leader ("**Maximilian I of Bavaria**"); wrong place ("St. Peter's Church") |
| 13 | Rosetta Stone | ✅ | Correct (1799, 3 scripts, Champollion, Ptolemy V) |
| 14 | Voynich manuscript | ⚠️⚠️ | "**two leaves**" (→~240pp); "**British Library** 1952" (→Yale Beinecke 1969); "discovered in **Vienna**" (→Italy) |
| 15 | Tardigrade | ✅ | Correct |
| 16 | Dunning–Kruger | ✅ | Correct |
| 17 | Sapir–Whorf | ✅ | Correct |
| 18 | Atacama Desert | ⚠️ | "Chile and **Ecuador**" (→Chile/Peru); size "**10,000 km²**" (→~105,000) |
| 19 | Strait of Malacca | ⚠️ | Length "**155 km**" (→~800 km) |
| 20 | Largest planet | ✅ | Jupiter |

**Clean: 9** (3,4,5,7,13,15,16,17,20) · **Errors: 11** (1,2,6,8,9,10,11,12,14,18,19)

The wrong answers are stated with full confidence (bold, structured, authoritative) — the dangerous failure mode RAG is meant to counter.

## Speed baseline (for the slowness comparison)
Sustained tok/s with RAG off: **~10–14.5 tok/s** on longer answers (5–7 tok/s on very short ones, where prefill dominates). This is the number to compare against the RAG-on run.

## What to watch in the on-run
RAG can only fix an error if the correct fact is in the article **lead** (we index lead-only). Strong candidates to flip ✅: Kazakhstan(1), Hypatia's father(6), Tordesillas country(11), Atacama(18). Less certain (specific buried facts): Antikythera museum(10), Voynich library(14), Defenestration date(12).
