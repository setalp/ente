#!/usr/bin/env python3
"""Vet candidate A/B questions against the real index in one model load.

Prints the top hit + score per question so we can pick ones that retrieve the
right article comfortably above the 0.45 gate.
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer
from search import load_index, search

QUESTIONS = [
    "What is the capital of Kazakhstan?",
    "What is the capital of Myanmar?",
    "What is the capital of Turkey?",
    "What is the capital of Australia?",
    "Who was Mansa Musa?",
    "Who was Hypatia?",
    "Who was Emmy Noether?",
    "Who was Srinivasa Ramanujan?",
    "Who was Grace Hopper?",
    "What is the Antikythera mechanism?",
    "What was the Treaty of Tordesillas?",
    "What was the Defenestration of Prague?",
    "What is the Rosetta Stone?",
    "What is the Voynich manuscript?",
    "What is a tardigrade?",
    "What is the Dunning-Kruger effect?",
    "What is the Sapir-Whorf hypothesis?",
    "Where is the Atacama Desert?",
    "What is the Strait of Malacca?",
    "What is the largest planet in the solar system?",
]

manifest, mat, meta = load_index(Path("index"))
model = SentenceTransformer(manifest["model"])
has_prompts = {"query", "document"} <= set(getattr(model, "prompts", {}) or {})

for q in QUESTIONS:
    res = search(model, manifest, mat, meta, q, 3, has_prompts)
    top_score, top_meta = res[0]
    gate = "OK " if top_score >= 0.45 else "LOW"
    second = f"  | 2nd: {res[1][1]['title']} ({res[1][0]:.2f})" if len(res) > 1 else ""
    print(f"{gate} {top_score:.3f}  {top_meta['title']:34.34s} <- {q}{second}")
