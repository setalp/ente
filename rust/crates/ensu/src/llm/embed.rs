use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;

use super::context::ContextHandle;
use super::format_error;

/// Compute a pooled, L2-normalized embedding for each input text.
///
/// Requires a context created with `ContextParams::embeddings = Some(true)`
/// (mean pooling). Each returned vector has length `n_embd` (768 for
/// EmbeddingGemma). Inputs must fit within both `n_ctx` and `n_batch`, since a
/// pooled embedding needs the whole sequence decoded in a single batch; callers
/// embed short texts (queries / lead passages), so this is not a real limit.
pub fn embed(context: &ContextHandle, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
    context.with_context_mut(|ctx| -> Result<Vec<Vec<f32>>, String> {
        let n_batch = ctx.n_batch() as usize;
        if n_batch == 0 {
            return Err("Context batch size is 0".to_string());
        }
        let n_ctx = ctx.n_ctx() as usize;

        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let tokens = ctx
                .model
                .str_to_token(&text, AddBos::Always)
                .map_err(|err| format_error("Tokenize failed", err))?;
            if tokens.is_empty() {
                return Err("Input produced no tokens".to_string());
            }
            if tokens.len() > n_ctx || tokens.len() > n_batch {
                return Err(format!(
                    "Input length {} exceeds context/batch size (n_ctx={n_ctx}, n_batch={n_batch})",
                    tokens.len()
                ));
            }

            ctx.clear_kv_cache();

            let mut batch = LlamaBatch::new(tokens.len(), 1);
            for (idx, token) in tokens.iter().enumerate() {
                // logits=true marks the token's output for pooling.
                batch
                    .add(*token, idx as i32, &[0], true)
                    .map_err(|err| format_error("Failed to add token", err))?;
            }
            ctx.decode(&mut batch)
                .map_err(|err| format_error("Embedding decode failed", err))?;

            let embedding = ctx
                .embeddings_seq_ith(0)
                .map_err(|err| format_error("Failed to read embeddings", err))?;

            let mut vector = embedding.to_vec();
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            // Reject a zero/NaN-norm embedding rather than emit an un-normalized
            // (e.g. all-zero) vector that would silently match nothing in search.
            if !(norm > 0.0) {
                return Err("Embedding has zero or invalid norm".to_string());
            }
            for value in &mut vector {
                *value /= norm;
            }
            out.push(vector);
        }
        Ok(out)
    })
}
