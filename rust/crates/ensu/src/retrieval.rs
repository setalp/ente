//! On-device retrieval index for Ensu knowledge RAG (Wikipedia, Wikivoyage, …).
//!
//! Loads a prebuilt, read-only index (int8 vectors + passage metadata) and does
//! a brute-force cosine top-k with a similarity-threshold gate. Embedding the
//! query is done separately by `llm::embed` (EmbeddingGemma); this module only
//! takes an already-normalized query vector and ranks passages.
//!
//! Index layout (a directory shipped/downloaded as an asset):
//!   manifest.json  { model, dim, count, scale, .. }
//!   vectors.i8     raw int8, row-major `count * dim` (unit vectors * scale)
//!   meta.jsonl     one {id,title,url,text} per line, aligned to vector rows
//!
//! Both `vectors.i8` and `meta.jsonl` are **memory-mapped**, not read onto the
//! heap. Vectors are scanned per query; passage metadata is parsed lazily — only
//! the top-k hit rows are deserialized, from byte offsets recorded at open. This
//! keeps heap flat regardless of corpus size, which matters once several corpora
//! (Wikipedia ~118 MB + Wikivoyage ~219 MB of text) are open at once alongside
//! the chat model; eagerly parsing every Passage would cost hundreds of MB of
//! resident heap.
//!
//! Brute force is fine for v1: a few hundred k passages × 768 dims of int8 is a
//! single linear scan per query; no ANN structure needed yet.

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model: String,
    pub dim: usize,
    pub count: usize,
    pub scale: f32,
    #[serde(default)]
    pub granularity: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub title: String,
    pub url: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub score: f32,
    pub passage: Passage,
}

pub struct RetrievalIndex {
    dim: usize,
    inv_scale: f32,
    // Memory-mapped int8 vectors (row-major count*dim). mmap keeps these pages
    // clean/OS-evictable instead of ~hundreds of MB of dirty heap, which matters
    // when the chat model is also resident on-device. Raw bytes are two's-complement
    // int8 read as u8.
    vectors: Mmap,
    // Memory-mapped meta.jsonl. Passages are parsed on demand (see `rows`).
    meta: Mmap,
    // Byte range [start, end) of each non-empty meta.jsonl line, aligned to the
    // vector rows. We record offsets at open (one cheap scan for newlines) and
    // deserialize a Passage only when it lands in a query's top-k, so the passage
    // text stays memory-mapped rather than parsed onto the heap.
    rows: Vec<(usize, usize)>,
}

impl RetrievalIndex {
    /// Load an index directory. Validates that vector and metadata counts agree
    /// with the manifest so a truncated download fails loudly rather than
    /// returning garbage.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();

        let manifest_text = fs::read_to_string(dir.join("manifest.json"))
            .map_err(|err| format!("Failed to read manifest.json: {err}"))?;
        let manifest: Manifest = serde_json::from_str(&manifest_text)
            .map_err(|err| format!("Failed to parse manifest.json: {err}"))?;
        // Require a finite, strictly-positive scale. `!is_finite()` rejects NaN
        // and infinities; `<= 0.0` rejects zero and negatives. A bare `!= 0.0`
        // check would let NaN/negative through and silently produce NaN/inverted
        // similarities so nothing ever clears the gate — defeating the intent
        // that a bad index fails loudly rather than returning empty forever.
        if manifest.dim == 0 || !manifest.scale.is_finite() || manifest.scale <= 0.0 {
            return Err(
                "Manifest dim must be non-zero and scale must be finite and > 0".to_string(),
            );
        }

        let file = File::open(dir.join("vectors.i8"))
            .map_err(|err| format!("Failed to open vectors.i8: {err}"))?;
        // SAFETY: the index is a read-only asset; we never mutate the mapping,
        // and concurrent external truncation of a shipped asset isn't expected.
        let vectors = unsafe { Mmap::map(&file) }
            .map_err(|err| format!("Failed to mmap vectors.i8: {err}"))?;
        let expected = manifest
            .count
            .checked_mul(manifest.dim)
            .ok_or_else(|| "manifest count*dim overflows usize".to_string())?;
        if vectors.len() != expected {
            return Err(format!(
                "vectors.i8 has {} bytes, expected count*dim = {}",
                vectors.len(),
                expected
            ));
        }

        let meta_file = File::open(dir.join("meta.jsonl"))
            .map_err(|err| format!("Failed to open meta.jsonl: {err}"))?;
        // SAFETY: read-only asset, same as vectors above.
        let meta = unsafe { Mmap::map(&meta_file) }
            .map_err(|err| format!("Failed to mmap meta.jsonl: {err}"))?;
        let rows = line_ranges(&meta);
        // Validate row count without parsing content — a truncated meta.jsonl
        // still fails loudly here, matching the eager-parse behaviour it replaces.
        if rows.len() != manifest.count {
            return Err(format!(
                "meta.jsonl has {} rows, expected {}",
                rows.len(),
                manifest.count
            ));
        }

        Ok(Self {
            dim: manifest.dim,
            inv_scale: 1.0 / manifest.scale,
            vectors,
            meta,
            rows,
        })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Parse the passage at `row` from the mmap'd meta.jsonl on demand.
    fn passage(&self, row: usize) -> Result<Passage, String> {
        let (start, end) = self.rows[row];
        serde_json::from_slice(&self.meta[start..end])
            .map_err(|err| format!("Failed to parse meta.jsonl row {row}: {err}"))
    }

    /// Cosine top-k over the index, keeping only hits at or above `threshold`
    /// (the similarity gate). `query` must be L2-normalized — `llm::embed`
    /// returns normalized vectors, and stored vectors were normalized before
    /// int8 quantization, so the dot product approximates cosine similarity.
    ///
    /// Returns hits sorted by descending score; empty when nothing clears the
    /// gate, which the caller treats as "don't inject retrieved context".
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchHit>, String> {
        if query.len() != self.dim {
            return Err(format!(
                "query has {} dims, index expects {}",
                query.len(),
                self.dim
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(f32, usize)> = Vec::new();
        for (row, chunk) in self.vectors.chunks_exact(self.dim).enumerate() {
            let mut dot = 0.0f32;
            for d in 0..self.dim {
                // mmap bytes are u8; reinterpret as two's-complement int8.
                dot += query[d] * f32::from(chunk[d] as i8);
            }
            // Apply the constant int8 dequant scale once per row rather than once
            // per element (fewer multiplies per query); the stored vectors were
            // unit-normalized before quantization so this approximates cosine.
            let dot = dot * self.inv_scale;
            if dot >= threshold {
                scored.push((dot, row));
            }
        }

        // Partial-select the top-k instead of fully sorting every above-threshold
        // hit, then sort just those k.
        let by_score_desc =
            |a: &(f32, usize), b: &(f32, usize)| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal);
        if scored.len() > k {
            scored.select_nth_unstable_by(k - 1, by_score_desc);
            scored.truncate(k);
        }
        scored.sort_unstable_by(by_score_desc);

        // Parse only the surviving top-k passages from the mmap (not all rows).
        // A row that fails to parse (corrupt sideload / bit-rot — downloads are
        // SHA-256 verified) is skipped rather than failing the whole query, so one
        // bad row can't drop the other top-k hits.
        Ok(scored
            .into_iter()
            .filter_map(|(score, row)| {
                self.passage(row)
                    .ok()
                    .map(|passage| SearchHit { score, passage })
            })
            .collect())
    }
}

/// Byte ranges [start, end) of each non-empty (non-whitespace-only) line in
/// `bytes`, excluding the trailing newline. Mirrors the old `lines().filter(!empty)`
/// so blank/trailing lines don't create phantom rows.
fn line_ranges(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut rows = Vec::new();
    let mut start = 0usize;
    // memchr is SIMD-accelerated; scanning a ~200 MB meta.jsonl byte-by-byte at
    // open would cost hundreds of ms and undercut the lazy-parse win.
    for nl in memchr::memchr_iter(b'\n', bytes) {
        push_if_nonempty(bytes, start, nl, &mut rows);
        start = nl + 1;
    }
    // Final line if the file doesn't end in a newline.
    push_if_nonempty(bytes, start, bytes.len(), &mut rows);
    rows
}

fn push_if_nonempty(bytes: &[u8], start: usize, end: usize, rows: &mut Vec<(usize, usize)>) {
    if end > start && !bytes[start..end].iter().all(|b| b.is_ascii_whitespace()) {
        rows.push((start, end));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a tiny 2-row index to a per-test temp dir and return its path.
    /// `name` must be unique per test — cargo runs tests as parallel threads of
    /// one process, so a shared path would race.
    fn write_index(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("ensu_retr_test_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("manifest.json"),
            r#"{"model":"test","dim":2,"count":2,"scale":127.0}"#,
        )
        .unwrap();
        // Row 0 = unit (1,0)*127, row 1 = unit (0,1)*127.
        fs::write(dir.join("vectors.i8"), [127i8 as u8, 0, 0, 127i8 as u8]).unwrap();
        // Trailing newline + a blank line must NOT create phantom rows.
        let mut meta = File::create(dir.join("meta.jsonl")).unwrap();
        writeln!(meta, r#"{{"id":"a","title":"Alpha","url":"ua","text":"ta"}}"#).unwrap();
        writeln!(meta, r#"{{"id":"b","title":"Beta","url":"ub","text":"tb"}}"#).unwrap();
        writeln!(meta).unwrap();
        dir
    }

    #[test]
    fn opens_and_counts_rows_ignoring_blank_lines() {
        let dir = write_index("count");
        let index = RetrievalIndex::open(&dir).unwrap();
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_gates_and_lazily_parses_top_row() {
        let dir = write_index("search");
        let index = RetrievalIndex::open(&dir).unwrap();
        // Query aligned with row 0; row 1 (orthogonal) scores 0 and is gated out.
        let hits = index.search(&[1.0, 0.0], 5, 0.5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].passage.title, "Alpha");
        assert!((hits[0].score - 1.0).abs() < 1e-4);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_meta_fails_to_open() {
        let dir = write_index("truncated");
        // Manifest says count=2 but leave only one row.
        fs::write(
            dir.join("meta.jsonl"),
            "{\"id\":\"a\",\"title\":\"Alpha\",\"url\":\"ua\",\"text\":\"ta\"}\n",
        )
        .unwrap();
        assert!(RetrievalIndex::open(&dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
