package io.ente.ensu.data.llm

import android.util.Log
import io.ente.ensu.domain.llm.RetrievalProvider
import io.ente.ensu.domain.llm.RetrievedPassage
import io.ente.labs.inference_rs.ContextHandle
import io.ente.labs.inference_rs.ContextParams
import io.ente.labs.inference_rs.ModelHandle
import io.ente.labs.inference_rs.ModelLoadParams
import io.ente.labs.inference_rs.createContext
import io.ente.labs.inference_rs.embed
import io.ente.labs.inference_rs.initBackend
import io.ente.labs.inference_rs.loadModel
import io.ente.labs.inference_rs.uniffiEnsureInitialized
import io.ente.labs.retrieval.RetrievalIndex
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * On-device Wikipedia retrieval.
 *
 * Embeds the query with EmbeddingGemma — loaded into the same llama.cpp engine
 * as the chat model but in embedding mode (mean pooling) — and runs cosine
 * top-k over a prebuilt index via the Rust `ensu-retrieval` crate. Both the
 * embedding model and the index ship/download as on-device assets.
 *
 * Lazily loads on first search so the embedding model (~200 MB) and index don't
 * cost anything until the user turns retrieval on. See
 * docs-fork/ensu/retrieval-design.md.
 */
class RustRetrievalProvider(
    private val embeddingModelPath: File,
    private val indexDir: File,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : RetrievalProvider {

    private val loadMutex = Mutex()

    @Volatile private var index: RetrievalIndex? = null
    @Volatile private var embeddingModel: ModelHandle? = null
    @Volatile private var embeddingContext: ContextHandle? = null

    init {
        uniffiEnsureInitialized()
    }

    override val isReady: Boolean
        get() = embeddingModelPath.exists() && File(indexDir, "manifest.json").exists()

    override suspend fun search(
        query: String,
        k: Int,
        threshold: Float,
    ): List<RetrievedPassage> = withContext(ioDispatcher) {
        if (query.isBlank() || !isReady) return@withContext emptyList()

        try {
            ensureLoaded()
            val idx = index ?: return@withContext emptyList()
            val ctx = embeddingContext ?: return@withContext emptyList()

            // Query must use EmbeddingGemma's retrieval query prompt so it lands
            // in the same space as the index, which was built with the matching
            // document prompt (see build_index.py).
            val embedded = embed(ctx, listOf(QUERY_PROMPT_PREFIX + query))
            val queryVector = embedded.firstOrNull() ?: return@withContext emptyList()

            idx.search(queryVector, k.toUInt(), threshold).map { hit ->
                RetrievedPassage(
                    title = hit.passage.title,
                    url = hit.passage.url,
                    text = hit.passage.text,
                    score = hit.score,
                )
            }
        } catch (error: Throwable) {
            // Retrieval is best-effort: never break a chat turn because the
            // index or embedding model failed to load.
            Log.w("RustRetrievalProvider", "Retrieval failed; continuing without context", error)
            emptyList()
        }
    }

    private suspend fun ensureLoaded() {
        if (index != null && embeddingContext != null) return
        loadMutex.withLock {
            if (index == null) {
                index = RetrievalIndex.open(indexDir.absolutePath)
            }
            if (embeddingContext == null) {
                initBackend()
                val model = loadModel(
                    ModelLoadParams(
                        modelPath = embeddingModelPath.absolutePath,
                        nGpuLayers = 0,
                        useMmap = true,
                        useMlock = false,
                    )
                )
                embeddingModel = model
                // n_batch >= n_ctx so a whole query fits in one decode for pooling.
                embeddingContext = createContext(
                    model,
                    ContextParams(
                        contextSize = EMBED_CONTEXT_SIZE,
                        nThreads = null,
                        nBatch = EMBED_CONTEXT_SIZE,
                        embeddings = true,
                    ),
                )
            }
        }
    }

    /** Free the embedding model/context/index (e.g. when retrieval is toggled off). */
    fun release() {
        embeddingContext?.destroy()
        embeddingContext = null
        embeddingModel?.destroy()
        embeddingModel = null
        index?.destroy()
        index = null
    }

    companion object {
        private const val QUERY_PROMPT_PREFIX = "task: search result | query: "
        private const val EMBED_CONTEXT_SIZE = 512
    }
}
