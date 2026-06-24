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
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException

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

            // Free EmbeddingGemma immediately: it's only needed for this one
            // embedding, and the chat model generation that follows is heavy on
            // RAM. Holding both models resident thrashes swap on-device. The
            // index stays loaded (cheap to keep, expensive to reload).
            releaseEmbeddingModel()

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

    override suspend fun downloadAssets(onProgress: (Int) -> Unit): Unit = withContext(ioDispatcher) {
        indexDir.mkdirs()
        embeddingModelPath.parentFile?.mkdirs()

        // url filename == on-device filename, so derive both from the targets.
        val targets = listOf(
            embeddingModelPath,
            File(indexDir, "manifest.json"),
            File(indexDir, "vectors.i8"),
            File(indexDir, "meta.jsonl")
        )
        val client = OkHttpClient()
        val sizes = targets.map { headContentLength(client, assetUrl(it.name)) }
        val total = sizes.sum().coerceAtLeast(1L)
        var done = 0L

        for ((i, target) in targets.withIndex()) {
            if (target.exists() && sizes[i] > 0 && target.length() == sizes[i]) {
                done += sizes[i]
                onProgress(((done * 100) / total).toInt().coerceIn(0, 100))
                continue
            }
            val tmp = File(target.parentFile, "${target.name}.part")
            client.newCall(Request.Builder().url(assetUrl(target.name)).build()).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("Download failed (${resp.code}) for ${target.name}")
                val body = resp.body ?: throw IOException("Empty body for ${target.name}")
                tmp.outputStream().use { out ->
                    body.byteStream().use { input ->
                        val buf = ByteArray(1 shl 16)
                        while (true) {
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            done += n
                            onProgress(((done * 100) / total).toInt().coerceIn(0, 100))
                        }
                    }
                }
            }
            if (!tmp.renameTo(target)) {
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
        }
        onProgress(100)
    }

    private fun assetUrl(fileName: String) = "$ASSET_BASE_URL/$fileName"

    private fun headContentLength(client: OkHttpClient, url: String): Long = try {
        client.newCall(Request.Builder().url(url).head().build()).execute().use { resp ->
            resp.header("Content-Length")?.toLongOrNull() ?: 0L
        }
    } catch (_: Throwable) {
        0L
    }

    /** Free just the embedding model + context, keeping the index loaded. */
    private suspend fun releaseEmbeddingModel() {
        loadMutex.withLock {
            embeddingContext?.destroy()
            embeddingContext = null
            embeddingModel?.destroy()
            embeddingModel = null
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
        private const val ASSET_BASE_URL =
            "https://github.com/setalp/ensu-rag-assets/releases/download/v1"
    }
}
