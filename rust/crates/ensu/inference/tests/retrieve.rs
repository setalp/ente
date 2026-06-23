//! End-to-end on-device retrieval test: embed a query with EmbeddingGemma, then
//! cosine-search the real prebuilt index. Exercises the full Rust path that the
//! Android `RustRetrievalProvider` drives.
//!
//! Gated on both env vars (no-op otherwise). Run with:
//!   ENSU_EMBED_GGUF=/path/to/embeddinggemma-300M-Q8_0.gguf \
//!   ENSU_INDEX_DIR=/path/to/spike/index \
//!     cargo test -p inference_rs --test retrieve -- --nocapture

use ensu_retrieval::RetrievalIndex;
use inference_rs::{ContextParams, ModelLoadParams, create_context, embed, init_backend, load_model};

#[test]
fn end_to_end_retrieval_on_real_index() {
    let (Ok(model_path), Ok(index_dir)) =
        (std::env::var("ENSU_EMBED_GGUF"), std::env::var("ENSU_INDEX_DIR"))
    else {
        eprintln!("skipping: set ENSU_EMBED_GGUF and ENSU_INDEX_DIR to run");
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

    let index = RetrievalIndex::open(&index_dir).expect("open index");
    println!("index loaded: {} passages", index.len());

    let query = "task: search result | query: how tall is mount everest".to_string();
    let query_vector = embed(&context, vec![query]).expect("embed").remove(0);
    let hits = index.search(&query_vector, 3, 0.45).expect("search");

    for hit in &hits {
        println!("  {:.3}  {}", hit.score, hit.passage.title);
    }
    assert!(!hits.is_empty(), "expected at least one hit above the 0.45 gate");
    assert!(
        hits[0].passage.title.contains("Everest"),
        "top hit should be about Everest, got '{}'",
        hits[0].passage.title
    );
}
