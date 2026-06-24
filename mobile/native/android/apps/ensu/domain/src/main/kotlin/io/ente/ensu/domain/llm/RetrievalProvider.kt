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
     * Download the embedding model + index assets onto the device (for shared
     * builds where they aren't sideloaded). [onProgress] receives 0..100.
     * Skips files already present at the expected size; throws on failure.
     */
    suspend fun downloadAssets(onProgress: (percent: Int) -> Unit)

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
    val title: String,
    val url: String,
    val text: String,
    val score: Float,
)
