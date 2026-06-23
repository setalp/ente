//! On-device retrieval index for Ensu Wikipedia RAG.
//!
//! Loads a prebuilt, read-only index (int8 vectors + passage metadata) and does
//! a brute-force cosine top-k with a similarity-threshold gate. Embedding the
//! query is done separately by `inference_rs::embed` (EmbeddingGemma); this
//! crate only takes an already-normalized query vector and ranks passages.
//!
//! Index layout (a directory shipped/downloaded as an asset):
//!   manifest.json  { model, dim, count, scale, .. }
//!   vectors.i8     raw int8, row-major `count * dim` (unit vectors * scale)
//!   meta.jsonl     one {id,title,url,text} per line, aligned to vector rows
//!
//! Brute force is fine for v1: ~240k passages × 768 dims is a few hundred MB of
//! int8 and a single linear scan per query; no ANN structure needed yet.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
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
    vectors: Vec<i8>,
    passages: Vec<Passage>,
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
        if manifest.dim == 0 || manifest.scale == 0.0 {
            return Err("Manifest dim/scale must be non-zero".to_string());
        }

        let raw = fs::read(dir.join("vectors.i8"))
            .map_err(|err| format!("Failed to read vectors.i8: {err}"))?;
        // Raw file bytes are u8; reinterpret as two's-complement int8.
        let vectors: Vec<i8> = raw.into_iter().map(|byte| byte as i8).collect();
        let expected = manifest.count * manifest.dim;
        if vectors.len() != expected {
            return Err(format!(
                "vectors.i8 has {} values, expected count*dim = {}",
                vectors.len(),
                expected
            ));
        }

        let meta_text = fs::read_to_string(dir.join("meta.jsonl"))
            .map_err(|err| format!("Failed to read meta.jsonl: {err}"))?;
        let passages: Vec<Passage> = meta_text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line)
                    .map_err(|err| format!("Failed to parse meta.jsonl row: {err}"))
            })
            .collect::<Result<_, _>>()?;
        if passages.len() != manifest.count {
            return Err(format!(
                "meta.jsonl has {} rows, expected {}",
                passages.len(),
                manifest.count
            ));
        }

        Ok(Self {
            dim: manifest.dim,
            inv_scale: 1.0 / manifest.scale,
            vectors,
            passages,
        })
    }

    pub fn len(&self) -> usize {
        self.passages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    /// Cosine top-k over the index, keeping only hits at or above `threshold`
    /// (the similarity gate). `query` must be L2-normalized — `inference_rs::embed`
    /// returns normalized vectors, and stored vectors were normalized before
    /// int8 quantization, so the dot product approximates cosine similarity.
    ///
    /// Returns hits sorted by descending score; empty when nothing clears the
    /// gate, which the caller treats as "don't inject retrieved context".
    pub fn search(&self, query: &[f32], k: usize, threshold: f32) -> Result<Vec<SearchHit>, String> {
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
                dot += query[d] * (f32::from(chunk[d]) * self.inv_scale);
            }
            if dot >= threshold {
                scored.push((dot, row));
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(score, row)| SearchHit {
                score,
                passage: self.passages[row].clone(),
            })
            .collect())
    }
}
