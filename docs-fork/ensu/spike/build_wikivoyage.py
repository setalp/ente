#!/usr/bin/env python3
"""Offline index builder for the Ensu **Wikivoyage** retrieval spike.

Second corpus after Simple-English-Wikipedia (see build_index.py). Wikivoyage's
value — specific POIs, transport, practical tips — lives in *sections*
(See / Do / Eat / Sleep / Get in), not the lead, so this is **section-chunked**,
not lead-only. It also extracts Wikivoyage **listing templates** ({{see}},
{{do}}, {{eat}}, {{sleep}}, {{listing}}, {{marker}}, {{vcard}}): those hold each
POI's name + description, which a plain wikitext strip would drop on the floor.

Same embed model + int8 quantization + on-device index format as build_index.py
(it reuses `shard_embed` / `write_index` from there), so `search.py` and the Rust
`RetrievalIndex` read the output unchanged. Meta rows carry an extra `section`
field; the on-device `Passage` deserializer ignores unknown fields, so the
shipped index stays byte-compatible.

Data source: the Wikimedia enwikivoyage dump — no clean HF text dataset exists,
and the dump preserves the section headings + listing templates we need:
  https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2
It is auto-downloaded to --dump if absent (it's small — tens of MB).

Outputs (in --out dir): identical layout to build_index.py.

  python build_wikivoyage.py --out index-wikivoyage            # full (bake offline)
  python build_wikivoyage.py --limit 300 --out index-wv-smoke \
      --model sentence-transformers/all-MiniLM-L6-v2           # fast plumbing smoke
  python search.py --index index-wikivoyage --interactive
"""
from __future__ import annotations

import argparse
import bz2
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote
from xml.etree.ElementTree import iterparse

import numpy as np

# Reuse the exact encode + int8 + index-writing path from the Wikipedia spike so
# both corpora are embedded and stored identically.
from build_index import shard_embed, write_index

DUMP_URL = ("https://dumps.wikimedia.org/enwikivoyage/latest/"
            "enwikivoyage-latest-pages-articles.xml.bz2")

# Templates whose name + description carry the concrete POI facts. Extracted
# explicitly because strip_code() discards template contents.
LISTING_TEMPLATES = {"see", "do", "buy", "eat", "drink", "sleep",
                     "listing", "marker", "vcard"}

# Navigation / boilerplate sections with no standalone factual value.
SKIP_SECTIONS = {"go next", "nearby", "routes", "references", "external links",
                 "see also", "further reading", "related", "other destinations",
                 "notes", "cite", "citations"}

# Article-type filtering. Wikivoyage status templates encode status+type, e.g.
# {{guidecity}}, {{outlineregion}}. We index actual *destinations* and drop
# overview / non-destination article types (their value is navigation, and they
# inflate the index disproportionately). Reject-list, not allow-list, so new /
# unclassified destination types stay indexed (recall-safe).
_STATUS_RE = re.compile(r"^(?:stub|outline|usable|guide|star)(.+)$")
DISAMBIG_TEMPLATES = {"disamb", "disambig", "geodis"}
DROP_TYPES = {"region", "country", "continent", "topic", "itinerary",
              "phrasebook", "nomination", "event"}


def _template_names(code) -> list[str]:
    return [re.sub(r"\s+", "", str(tp.name).strip().lower())
            for tp in code.filter_templates()]


def keep_article(names: list[str]) -> bool:
    """Index destinations; drop disambiguation + non-destination overview types."""
    if any(n in DISAMBIG_TEMPLATES for n in names):
        return False
    for n in names:
        m = _STATUS_RE.match(n)
        if m:
            return m.group(1) not in DROP_TYPES
    return True  # no status template -> keep (new / unclassified destination)


def _clean(value) -> str:
    """Strip wiki markup from a template value / node to plain text."""
    import mwparserfromhell as mw
    return mw.parse(str(value)).strip_code().strip()


def _normalize_ws(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _pack(text: str, max_chars: int):
    """Greedy paragraph packing into <= max_chars chunks; hard-split monster paras."""
    chunks: list[str] = []
    cur = ""
    for para in (p.strip() for p in re.split(r"\n+", text)):
        if not para:
            continue
        if cur and len(cur) + len(para) + 1 > max_chars:
            chunks.append(cur)
            cur = para
        else:
            cur = para if not cur else f"{cur}\n{para}"
        while len(cur) > max_chars:
            chunks.append(cur[:max_chars])
            cur = cur[max_chars:].strip()
    if cur:
        chunks.append(cur)
    return chunks


def section_chunks(code, max_chars: int, min_chars: int):
    """Yield (section_name, chunk_text) for a parsed Wikivoyage article.

    `code` is an mwparserfromhell Wikicode (parsed once by the caller, shared
    with the article-type filter). Lead section -> section_name "". Each
    section's prose (headings + listing templates removed) is concatenated with
    one "Name: description" line per listing, then packed into <= max_chars
    chunks. Mutates `code` (strips file links / ref tags in place).
    """
    # Strip noise that strip_code() would otherwise leak as text: image/file
    # wikilinks leave "thumb|300px|caption" fragments, and <ref>/<gallery> tags
    # leave citation/filename junk. Drop them wholesale before chunking.
    for link in list(code.filter_wikilinks()):
        if str(link.title).strip().lower().startswith(("file:", "image:")):
            try:
                code.remove(link)
            except ValueError:
                pass
    for tag in list(code.filter_tags()):
        if str(tag.tag).strip().lower() in ("ref", "gallery"):
            try:
                code.remove(tag)
            except ValueError:
                pass
    for sec in code.get_sections(flat=True, include_lead=True, include_headings=True):
        headings = sec.filter_headings()
        name = _clean(headings[0].title) if headings else ""
        if name.lower() in SKIP_SECTIONS:
            continue

        listing_lines: list[str] = []
        for tpl in sec.filter_templates():
            if str(tpl.name).strip().lower() not in LISTING_TEMPLATES:
                continue
            label = next((_clean(tpl.get(p).value) for p in ("name", "alt")
                          if tpl.has(p) and _clean(tpl.get(p).value)), "")
            desc = next((_clean(tpl.get(p).value) for p in ("content", "description")
                         if tpl.has(p) and _clean(tpl.get(p).value)), "")
            line = ": ".join(x for x in (label, desc) if x)
            if line:
                listing_lines.append(line)

        # Drop headings so the section title isn't embedded in the body (we add
        # it back as context at embed time); strip_code drops the now-extracted
        # listing templates too.
        for h in list(sec.filter_headings()):
            sec.remove(h)
        prose = _normalize_ws(sec.strip_code())

        body = "\n".join(x for x in (prose, "\n".join(listing_lines)) if x)
        for chunk in _pack(body, max_chars):
            if len(chunk) >= min_chars:
                yield name, chunk


def iter_articles(dump_path: Path):
    """Stream (pageid, title, wikitext) for main-namespace, non-redirect pages."""
    with bz2.open(dump_path, "rt", encoding="utf-8") as fh:
        for _, elem in iterparse(fh, events=("end",)):
            if elem.tag.rsplit("}", 1)[-1] != "page":
                continue
            ns = elem.findtext("{*}ns")
            redirect = elem.find("{*}redirect")
            text = elem.findtext("{*}revision/{*}text")
            if ns == "0" and redirect is None and text:
                yield (elem.findtext("{*}id"), elem.findtext("{*}title"), text)
            elem.clear()  # keep memory flat over the whole dump


def ensure_dump(path: Path) -> Path:
    if not path.exists():
        print(f"downloading {DUMP_URL}\n         -> {path} ...", flush=True)
        urllib.request.urlretrieve(DUMP_URL, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("index-wikivoyage"))
    ap.add_argument("--model", default="google/embeddinggemma-300m",
                    help="HF model id. Gated; use a non-gated model to smoke-test.")
    ap.add_argument("--dump", type=Path,
                    default=Path("enwikivoyage-latest-pages-articles.xml.bz2"),
                    help="Wikimedia dump path (auto-downloaded if missing).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N articles (0 = all).")
    ap.add_argument("--max-chars", type=int, default=900,
                    help="Char budget per section chunk.")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="Skip chunks shorter than this.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--fp16", action="store_true", default=False,
                    help="float16 (~2x faster). OFF by default: EmbeddingGemma "
                         "overflows to all-NaN in fp16 on MPS. Enable only for "
                         "fp16-safe models (MiniLM smoke).")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    args = ap.parse_args()

    ensure_dump(args.dump)

    import mwparserfromhell as mw

    print(f"[1/3] parsing + chunking {args.dump} (destinations only) ...", flush=True)
    rows, docs = [], []
    n_articles = n_dropped = 0
    for pageid, title, wikitext in iter_articles(args.dump):
        code = mw.parse(wikitext)
        if not keep_article(_template_names(code)):
            n_dropped += 1
            continue
        chunks = list(section_chunks(code, args.max_chars, args.min_chars))
        if not chunks:
            continue
        n_articles += 1
        slug = quote(title.replace(" ", "_"))
        for si, (section, text) in enumerate(chunks):
            url = f"https://en.wikivoyage.org/wiki/{slug}"
            if section:
                url += f"#{quote(section.replace(' ', '_'))}"
            rows.append({"id": f"{pageid}-{si}", "title": title, "url": url,
                         "section": section, "text": text})
            # Prepend title (+ section) so it steers the embedding — mirrors the
            # "Title. lead" title boost in build_index.py.
            ctx = f"{title} — {section}. {text}" if section else f"{title}. {text}"
            docs.append(ctx)
        if n_articles % 2000 == 0:
            print(f"      {n_articles:,} articles -> {len(rows):,} chunks ...", flush=True)
        if args.limit and n_articles >= args.limit:
            break
    print(f"      kept {len(rows):,} chunks from {n_articles:,} destinations "
          f"({len(rows)/max(n_articles,1):.1f} chunks/article); "
          f"dropped {n_dropped:,} non-destination articles.", flush=True)

    print(f"[2/3] embedding ...", flush=True)
    q, dim = shard_embed(args.out, docs, args.model, args.device, args.fp16,
                         args.batch_size, args.shard_size)
    print(f"[3/3] writing -> {args.out}/ ...", flush=True)
    write_index(args.out, rows, q, dim, {
        "model": args.model, "fp16": args.fp16,
        "dataset": "enwikivoyage-latest", "granularity": "section-chunk",
        "max_chars": args.max_chars,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
