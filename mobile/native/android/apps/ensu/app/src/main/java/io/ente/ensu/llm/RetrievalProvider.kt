package io.ente.ensu.llm

/**
 * On-device retrieval over one or more prebuilt knowledge corpora (Wikipedia +
 * Wikivoyage for v1).
 *
 * Implementations embed the query locally and return the top passages that
 * clear a similarity threshold, merged across every available corpus and tagged
 * with their [RetrievedPassage.source]. An empty result means the gate rejected
 * everything and no context should be injected into the prompt. See
 * docs-fork/ensu/retrieval-design.md.
 */
interface RetrievalProvider {
    /** True when the shared embedding model is present (retrieval prerequisite). */
    val isEmbeddingModelReady: Boolean

    /**
     * Bytes still needed to fetch the embedding model (0 if already present).
     * Added to the chat-model estimate so the first-run "Download" size reflects
     * the embedding model bundled alongside it.
     */
    fun embeddingDownloadBytesRemaining(): Long

    /**
     * Download just the shared embedding model — the retrieval prerequisite bundled
     * alongside the chat model at first run. Datasets are downloaded on demand
     * ([downloadCorpus]). [onProgress] receives 0..100; throws on failure.
     */
    suspend fun downloadEmbeddingModel(onProgress: (percent: Int) -> Unit)

    /** Per-corpus on-device status, for the Knowledge settings screen. */
    fun corpora(): List<CorpusInfo>

    /**
     * Download the shared embedding model (if missing) + the given corpus's index,
     * on user demand. Throws on failure / unknown / non-downloadable corpus.
     */
    suspend fun downloadCorpus(corpusId: String, onProgress: (percent: Int) -> Unit)

    /**
     * Embed [query] and search only the corpora whose id is in [enabledCorpusIds]
     * (and that are present on-device), returning up to [k] passages with cosine
     * score >= [threshold], merged and sorted by descending score. Empty if no
     * enabled corpus is ready or nothing clears the gate.
     */
    suspend fun search(
        query: String,
        enabledCorpusIds: Set<String>,
        k: Int = DEFAULT_K,
        threshold: Float = DEFAULT_THRESHOLD,
    ): List<RetrievedPassage>

    companion object {
        const val DEFAULT_K = 2

        /**
         * Raised 0.45 -> 0.50 after the A/B study: real matches landed at
         * 0.55-0.70 while distractors that contaminated answers (e.g. the
         * Mansa Musa case) clustered at 0.45-0.47. 0.50 drops them cleanly.
         */
        const val DEFAULT_THRESHOLD = 0.50f
    }
}

data class RetrievedPassage(
    /** Human-readable corpus this passage came from, e.g. "Wikipedia" / "Wikivoyage". */
    val source: String,
    val title: String,
    val url: String,
    val text: String,
    val score: Float,
)

/** On-device status of one knowledge corpus, for the Settings data rows. */
data class CorpusInfo(
    val id: String,
    val label: String,
    val ready: Boolean,
    /** Total download size in bytes (index files; the shared model is separate). */
    val downloadBytes: Long,
    /** False for sideload-only corpora whose assets aren't hosted yet. */
    val downloadable: Boolean,
)
