use ente_ensu::retrieval as core;
use thiserror::Error;

#[derive(Debug, Error, uniffi::Error)]
pub enum RetrievalError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for RetrievalError {
    fn from(value: String) -> Self {
        Self::Message(value)
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct RetrievalPassage {
    pub id: String,
    pub title: String,
    pub url: String,
    pub text: String,
}

impl From<core::Passage> for RetrievalPassage {
    fn from(value: core::Passage) -> Self {
        Self {
            id: value.id,
            title: value.title,
            url: value.url,
            text: value.text,
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct RetrievalSearchHit {
    pub score: f32,
    pub passage: RetrievalPassage,
}

impl From<core::SearchHit> for RetrievalSearchHit {
    fn from(value: core::SearchHit) -> Self {
        Self {
            score: value.score,
            passage: value.passage.into(),
        }
    }
}

#[derive(uniffi::Object)]
pub struct RetrievalIndex {
    inner: core::RetrievalIndex,
}

#[uniffi::export]
impl RetrievalIndex {
    /// Open a prebuilt index directory (manifest.json + vectors.i8 + meta.jsonl).
    #[uniffi::constructor]
    pub fn open(dir: String) -> Result<Self, RetrievalError> {
        Ok(Self {
            inner: core::RetrievalIndex::open(dir)?,
        })
    }

    pub fn len(&self) -> u32 {
        u32::try_from(self.inner.len()).unwrap_or(u32::MAX)
    }

    /// Cosine top-k over the index, gated at `threshold`. `query` must be the
    /// L2-normalized embedding from `llm_embed`. Empty result => the gate
    /// rejected everything (caller injects no retrieved context).
    pub fn search(
        &self,
        query: Vec<f32>,
        k: u32,
        threshold: f32,
    ) -> Result<Vec<RetrievalSearchHit>, RetrievalError> {
        let hits = self.inner.search(&query, k as usize, threshold)?;
        Ok(hits.into_iter().map(Into::into).collect())
    }
}
