//! Runtime smoke test for `embed()`.
//!
//! Gated on `ENSU_EMBED_GGUF` pointing at an EmbeddingGemma GGUF, so it's a
//! no-op in CI / for anyone without the model. Run with:
//!   ENSU_EMBED_GGUF=/path/to/embeddinggemma-300M-Q8_0.gguf \
//!     cargo test -p inference_rs --test embed -- --nocapture

use inference_rs::{ContextParams, ModelLoadParams, create_context, embed, init_backend, load_model};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn embed_is_normalized_768d_and_ranks_similarity() {
    let Ok(model_path) = std::env::var("ENSU_EMBED_GGUF") else {
        eprintln!("skipping: set ENSU_EMBED_GGUF to an EmbeddingGemma GGUF to run");
        return;
    };

    init_backend().expect("init backend");
    let model = load_model(ModelLoadParams {
        model_path,
        n_gpu_layers: Some(0),
        use_mmap: Some(true),
        use_mlock: Some(false),
    })
    .expect("load model");

    let context = create_context(
        model,
        ContextParams {
            context_size: Some(512),
            n_threads: None,
            n_batch: Some(512),
            embeddings: Some(true),
        },
    )
    .expect("create embedding context");

    // EmbeddingGemma retrieval prompts: a query and two documents, one relevant
    // and one not.
    let texts = vec![
        "task: search result | query: how tall is mount everest".to_string(),
        "title: none | text: Mount Everest is the highest mountain on Earth.".to_string(),
        "title: none | text: Bananas are a yellow fruit rich in potassium.".to_string(),
    ];
    let vectors = embed(&context, texts).expect("embed");

    assert_eq!(vectors.len(), 3);
    for vector in &vectors {
        assert_eq!(vector.len(), 768, "EmbeddingGemma should yield 768 dims");
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit-normalized, got {norm}");
    }

    let related = cosine(&vectors[0], &vectors[1]);
    let unrelated = cosine(&vectors[0], &vectors[2]);
    println!("cosine(query, everest)={related:.3}  cosine(query, banana)={unrelated:.3}");
    assert!(
        related > unrelated,
        "the Everest doc must rank above the banana doc for an Everest query"
    );
}
