package io.ente.ensu.llm

import android.util.Log
import io.ente.ensu.bindings.LlmContextHandle
import io.ente.ensu.bindings.LlmContextParams
import io.ente.ensu.bindings.LlmModelHandle
import io.ente.ensu.bindings.LlmModelLoadParams
import io.ente.ensu.bindings.RetrievalIndex
import io.ente.ensu.bindings.llmCreateContext
import io.ente.ensu.bindings.llmEmbed
import io.ente.ensu.bindings.llmInitBackend
import io.ente.ensu.bindings.llmLoadModel
import io.ente.ensu.bindings.uniffiEnsureInitialized
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest

/**
 * On-device Wikipedia retrieval.
 *
 * Embeds the query with EmbeddingGemma (loaded into the same llama.cpp engine as
 * the chat model, in embedding mode) and runs cosine top-k over a prebuilt index
 * via the Rust `ente-ensu` retrieval module. Both the embedding model and the
 * index ship/download as on-device assets. See docs-fork/ensu/retrieval-design.md.
 */
class RustRetrievalProvider(
    private val embeddingModelPath: File,
    private val indexDir: File,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : RetrievalProvider {

    private data class RagAsset(val target: File, val size: Long, val sha256: String)

    // Expected on-device assets with verified size + SHA-256 (the source of truth
    // for isReady, the download skip/resume logic, and integrity verification).
    private val assets: List<RagAsset> = listOf(
        RagAsset(embeddingModelPath, 333_590_944L, "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"),
        RagAsset(File(indexDir, "manifest.json"), 230L, "e20da369ffc98a2777d72fe618dc39b5cdbe4c6a5585219550aa7c0db357b9f4"),
        RagAsset(File(indexDir, "vectors.i8"), 180_762_624L, "dac1102d01b164bd2481fc94b2921c96c462f585b17dbfaf5b356c1c62b5b50e"),
        RagAsset(File(indexDir, "meta.jsonl"), 118_052_053L, "43170744112562f8098a0c2d218706bf31a4306b56d43554976ae5b886b8c2de"),
    )

    // Serializes load/embed/search (single-tenant chat) and prevents an
    // overlapping search from using a just-released embedding context.
    private val loadMutex = Mutex()
    // Single-flight: the Settings download and the bundled model download both
    // route through downloadAssets; this stops two writers racing the same files.
    private val downloadMutex = Mutex()

    @Volatile private var index: RetrievalIndex? = null
    @Volatile private var embeddingModel: LlmModelHandle? = null
    @Volatile private var embeddingContext: LlmContextHandle? = null

    init {
        uniffiEnsureInitialized()
    }

    override val isReady: Boolean
        get() = assets.all { it.target.exists() && it.target.length() == it.size }

    override suspend fun search(
        query: String,
        k: Int,
        threshold: Float,
    ): List<RetrievedPassage> = withContext(ioDispatcher) {
        if (query.isBlank() || !isReady) return@withContext emptyList()
        try {
            // Hold the lock across load + embed + release so a concurrent search
            // can't embed on a context this one just destroyed.
            loadMutex.withLock {
                ensureLoadedLocked()
                val idx = index ?: return@withLock emptyList<RetrievedPassage>()
                val ctx = embeddingContext ?: return@withLock emptyList<RetrievedPassage>()

                // Query uses EmbeddingGemma's retrieval query prompt so it lands in
                // the same space as the index (built with the matching document prompt).
                val embedded = llmEmbed(ctx, listOf(QUERY_PROMPT_PREFIX + query))
                // Free EmbeddingGemma right away — only needed for this embedding,
                // and the chat generation that follows is RAM-heavy on-device.
                releaseEmbeddingLocked()

                val queryVector = embedded.firstOrNull() ?: return@withLock emptyList<RetrievedPassage>()
                idx.search(queryVector, k.toUInt(), threshold).map { hit ->
                    RetrievedPassage(
                        title = hit.passage.title,
                        url = hit.passage.url,
                        text = hit.passage.text,
                        score = hit.score,
                    )
                }
            }
        } catch (error: Throwable) {
            // Best-effort: never break a chat turn because retrieval failed.
            Log.w("RustRetrievalProvider", "Retrieval failed; continuing without context", error)
            emptyList()
        }
    }

    /** Load index + embedding context. Caller must hold [loadMutex]. */
    private fun ensureLoadedLocked() {
        if (index == null) {
            // Note: `isReady` only checks file length, so a right-size-but-corrupt
            // index (bad sideload / bit-rot) passes it yet fails to open here. We
            // deliberately do NOT delete the files on failure — open() can also fail
            // transiently under memory pressure (mmap ENOMEM, OOM reading meta.jsonl),
            // and deleting would destroy sideloaded data and force a network
            // re-download. The failure propagates to search()'s best-effort catch, so
            // retrieval degrades to off (logged) rather than corrupting state.
            index = RetrievalIndex.open(indexDir.absolutePath)
        }
        if (embeddingContext == null) {
            llmInitBackend()
            val model = llmLoadModel(
                LlmModelLoadParams(
                    modelPath = embeddingModelPath.absolutePath,
                    nGpuLayers = 0,
                    useMmap = true,
                    useMlock = false,
                )
            )
            embeddingModel = model
            // n_batch >= n_ctx so a whole query fits in one decode for pooling.
            embeddingContext = llmCreateContext(
                model,
                LlmContextParams(
                    contextSize = EMBED_CONTEXT_SIZE,
                    nThreads = null,
                    nBatch = EMBED_CONTEXT_SIZE,
                    embeddings = true,
                ),
            )
        }
    }

    /** Free embedding model + context, keeping the index. Caller must hold [loadMutex]. */
    private fun releaseEmbeddingLocked() {
        embeddingContext?.destroy()
        embeddingContext = null
        embeddingModel?.destroy()
        embeddingModel = null
    }

    override suspend fun downloadAssets(onProgress: (Int) -> Unit): Unit = withContext(ioDispatcher) {
        downloadMutex.withLock {
            if (isReady) {
                onProgress(100)
                return@withLock
            }
            indexDir.mkdirs()
            embeddingModelPath.parentFile?.mkdirs()

            val total = assets.sumOf { it.size }.coerceAtLeast(1L)
            val needed = assets.filterNot { complete(it) }.sumOf { it.size }
            val available = indexDir.usableSpace
            if (available in 1 until (needed + FREE_SPACE_MARGIN)) {
                throw IOException(
                    "Not enough free space: need ~${(needed + FREE_SPACE_MARGIN) / 1_000_000} MB, " +
                        "${available / 1_000_000} MB available"
                )
            }

            val client = OkHttpClient()
            try {
                var done = assets.sumOf { if (complete(it)) it.size else 0L }
                onProgress(((done * 100) / total).toInt().coerceIn(0, 100))

                for (asset in assets) {
                    if (complete(asset)) continue
                    val tmp = partFile(asset)
                    // Delete a partial that is >= the full size: an over-length one
                    // is corrupt, and an exactly-full one (written but killed before
                    // verify/rename) would otherwise send `Range: bytes=<size>-` and
                    // get a permanent HTTP 416, wedging the download forever. Start it
                    // fresh instead.
                    if (tmp.exists() && tmp.length() >= asset.size) tmp.delete()
                    var existing = if (tmp.exists()) tmp.length() else 0L

                    val request = Request.Builder().url(assetUrl(asset.target.name))
                    if (existing > 0) request.header("Range", "bytes=$existing-")
                    client.newCall(request.build()).execute().use { resp ->
                        if (!resp.isSuccessful) {
                            throw IOException("Download failed (${resp.code}) for ${asset.target.name}")
                        }
                        // Resume only if the server honored Range (206); else restart.
                        val resuming = existing > 0 && resp.code == 206
                        if (!resuming) existing = 0L
                        done += existing
                        val body = resp.body ?: throw IOException("Empty body for ${asset.target.name}")
                        FileOutputStream(tmp, resuming).use { out ->
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

                    // Verify size + checksum before publishing the file.
                    if (tmp.length() != asset.size) {
                        tmp.delete()
                        throw IOException("Size mismatch for ${asset.target.name}: ${tmp.length()} != ${asset.size}")
                    }
                    if (!sha256(tmp).equals(asset.sha256, ignoreCase = true)) {
                        tmp.delete()
                        throw IOException("Checksum mismatch for ${asset.target.name}")
                    }
                    if (!tmp.renameTo(asset.target)) {
                        tmp.copyTo(asset.target, overwrite = true)
                        tmp.delete()
                    }
                }
                onProgress(100)
            } finally {
                client.dispatcher.executorService.shutdown()
                client.connectionPool.evictAll()
            }
        }
    }

    /** Free the embedding model/context/index (e.g. when retrieval is toggled off). */
    suspend fun release() {
        loadMutex.withLock {
            releaseEmbeddingLocked()
            index?.destroy()
            index = null
        }
    }

    private fun complete(asset: RagAsset) = asset.target.exists() && asset.target.length() == asset.size

    private fun partFile(asset: RagAsset) = File(asset.target.parentFile, "${asset.target.name}.part")

    private fun assetUrl(fileName: String) = "$ASSET_BASE_URL/$fileName"

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val QUERY_PROMPT_PREFIX = "task: search result | query: "
        private const val EMBED_CONTEXT_SIZE = 512
        private const val FREE_SPACE_MARGIN = 200L * 1024 * 1024
        private const val ASSET_BASE_URL =
            "https://github.com/setalp/ensu-rag-assets/releases/download/v1"
    }
}
