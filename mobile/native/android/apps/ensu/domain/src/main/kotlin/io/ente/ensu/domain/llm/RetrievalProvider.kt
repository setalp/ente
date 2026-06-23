package io.ente.ensu.domain.llm

/**
 * On-device retrieval over a prebuilt knowledge index (Wikipedia for v1).
 *
 * Implementations embed the query locally and return the top passages that
 * clear a similarity threshold. An empty result means the gate rejected
 * everything and no context should be injected into the prompt. See
 * docs-fork/ensu/retrieval-design.md.
 */
interface RetrievalProvider {
    /** True when the embedding model and index are present on-device. */
    val isReady: Boolean

    /**
     * Embed [query] and return up to [k] passages with cosine score >= [threshold],
     * sorted by descending score. Returns empty if not ready or nothing clears the gate.
     */
    suspend fun search(
        query: String,
        k: Int = DEFAULT_K,
        threshold: Float = DEFAULT_THRESHOLD,
    ): List<RetrievedPassage>

    companion object {
        const val DEFAULT_K = 3

        /** Empirical from the spike: factual queries ~0.5, chit-chat ~0.26. */
        const val DEFAULT_THRESHOLD = 0.45f
    }
}

data class RetrievedPassage(
    val title: String,
    val url: String,
    val text: String,
    val score: Float,
)
