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
 * On-device retrieval over one or more prebuilt knowledge corpora (Wikipedia +
 * Wikivoyage for v1).
 *
 * Embeds the query with EmbeddingGemma (loaded into the same llama.cpp engine as
 * the chat model, in embedding mode) *once*, then runs cosine top-k over every
 * ready corpus via the Rust `ente-ensu` retrieval module and merges the results.
 * All corpora share the one embedding space, int8 scale, and cosine metric, so
 * scores are directly comparable across indexes — merge-sort by score and take a
 * global top-k. The embedding model + each corpus index ship/download as on-device
 * assets. See docs-fork/ensu/retrieval-design.md ("Additional corpora").
 */
class RustRetrievalProvider(
    private val embeddingModelPath: File,
    private val retrievalDir: File,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) : RetrievalProvider {

    // remoteName: file name under ASSET_BASE_URL, or null for a sideload-only
    // asset (present via `adb push` but not yet auto-downloaded — Wikivoyage
    // hosting is a follow-up). Sideload-only assets still count for isReady/search.
    private data class RagAsset(
        val target: File,
        val size: Long,
        val sha256: String,
        val remoteName: String?,
    )

    /** One knowledge corpus: an index directory (manifest.json + vectors.i8 + meta.jsonl). */
    private data class Corpus(
        val id: String,
        val label: String,
        val dir: File,
        val files: List<RagAsset>,
    ) {
        // Right size on disk for all three files. `isReady` uses length only; a
        // right-size-but-corrupt file passes here yet fails RetrievalIndex.open,
        // handled best-effort in ensureLoadedLocked.
        val isReady: Boolean get() = files.all { it.target.exists() && it.target.length() == it.size }
    }

    // Shared embedding model — embeds the query once for all corpora.
    private val embeddingAsset = RagAsset(
        embeddingModelPath, 333_590_944L,
        "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
        remoteName = embeddingModelPath.name,
    )

    // remotePrefix: prefix for this corpus's files under ASSET_BASE_URL (GitHub
    // release assets are a flat namespace, so distinct corpora need distinct
    // names). null => sideload-only (not downloadable). "" => flat names.
    private fun corpus(id: String, label: String, subDir: String, remotePrefix: String?,
                       entries: List<Triple<String, Long, String>>): Corpus {
        val dir = File(retrievalDir, subDir)
        return Corpus(id, label, dir, entries.map { (name, size, sha) ->
            RagAsset(File(dir, name), size, sha, remoteName = remotePrefix?.let { "$it$name" })
        })
    }

    private val corpora: List<Corpus> = listOf(
        // Wikipedia (primary) — flat file names, hosted as today.
        corpus("wikipedia", "Wikipedia", "index", remotePrefix = "", entries = listOf(
            Triple("manifest.json", 230L, "e20da369ffc98a2777d72fe618dc39b5cdbe4c6a5585219550aa7c0db357b9f4"),
            Triple("vectors.i8", 180_762_624L, "dac1102d01b164bd2481fc94b2921c96c462f585b17dbfaf5b356c1c62b5b50e"),
            Triple("meta.jsonl", 118_052_053L, "43170744112562f8098a0c2d218706bf31a4306b56d43554976ae5b886b8c2de"),
        )),
        // Wikivoyage — section-chunked, built fp32. Sizes + SHA-256 are from the
        // canonical built index. Downloadable under "wikivoyage-*" remote names;
        // the assets must be uploaded to the release under those names (until then
        // downloadCorpus 404s and sideloading to index-wikivoyage/ still works).
        corpus("wikivoyage", "Wikivoyage", "index-wikivoyage", remotePrefix = "wikivoyage-", entries = listOf(
            Triple("manifest.json", 218L, "cedbe0a413d1e7c0aea7f5599781ff2f2701d554dfe6716670cb022b589bcda5"),
            Triple("vectors.i8", 274_859_520L, "adca352e3e7772f2ae14184638dcd34b920c7ccd239c77f836d2e55e1f5510d6"),
            Triple("meta.jsonl", 218_542_893L, "ecb530aa3f4785734faebd633e60777256bdf77a6bd63bc66380fda04d8277f1"),
        )),
    )

    // Serializes load/embed/search (single-tenant chat) and prevents an
    // overlapping search from using a just-released embedding context.
    private val loadMutex = Mutex()
    // Single-flight: the Settings download and the bundled model download both
    // route through downloadAssets; this stops two writers racing the same files.
    private val downloadMutex = Mutex()

    // Opened index per ready corpus id. Guarded by loadMutex.
    private val indexes = mutableMapOf<String, RetrievalIndex>()
    @Volatile private var embeddingModel: LlmModelHandle? = null
    @Volatile private var embeddingContext: LlmContextHandle? = null

    init {
        uniffiEnsureInitialized()
    }

    override val isEmbeddingModelReady: Boolean
        get() = embeddingModelPath.exists() && embeddingModelPath.length() == embeddingAsset.size

    override fun embeddingDownloadBytesRemaining(): Long =
        if (isEmbeddingModelReady) 0L else embeddingAsset.size

    override suspend fun search(
        query: String,
        enabledCorpusIds: Set<String>,
        k: Int,
        threshold: Float,
    ): List<RetrievedPassage> = withContext(ioDispatcher) {
        if (query.isBlank() || !isEmbeddingModelReady || enabledCorpusIds.isEmpty()) {
            return@withContext emptyList()
        }
        try {
            // Hold the lock across load + embed + release so a concurrent search
            // can't embed on a context this one just destroyed.
            loadMutex.withLock {
                ensureLoadedLocked()
                val ctx = embeddingContext ?: return@withLock emptyList<RetrievedPassage>()
                // Only the enabled + ready corpora participate.
                val active = indexes.filterKeys { it in enabledCorpusIds }
                if (active.isEmpty()) return@withLock emptyList<RetrievedPassage>()

                // Query uses EmbeddingGemma's retrieval query prompt so it lands in
                // the same space as the indexes (built with the matching document prompt).
                val embedded = llmEmbed(ctx, listOf(QUERY_PROMPT_PREFIX + query))
                // Free EmbeddingGemma right away — only needed for this embedding,
                // and the chat generation that follows is RAM-heavy on-device.
                releaseEmbeddingLocked()

                val queryVector = embedded.firstOrNull() ?: return@withLock emptyList<RetrievedPassage>()
                val labels = corpora.associate { it.id to it.label }

                // One embedding, fanned across every enabled index. Each returns its
                // own top-k above the shared gate; merge, sort by comparable score,
                // and keep the global top-k. Source tag drives citations + injection.
                active.entries.flatMap { (id, idx) ->
                    // Isolate per-corpus failures: with lazy meta.jsonl parsing a
                    // corrupt row can throw during search. Skip just that corpus so
                    // the others still serve, rather than letting the outer catch
                    // swallow every corpus's hits for this turn.
                    val hits = try {
                        idx.search(queryVector, k.toUInt(), threshold)
                    } catch (error: Throwable) {
                        Log.w("RustRetrievalProvider", "Search failed for corpus $id; skipping it", error)
                        emptyList()
                    }
                    hits.map { hit ->
                        RetrievedPassage(
                            source = labels[id] ?: id,
                            title = hit.passage.title,
                            url = hit.passage.url,
                            text = hit.passage.text,
                            score = hit.score,
                        )
                    }
                }.sortedByDescending { it.score }.take(k)
            }
        } catch (error: Throwable) {
            // Best-effort: never break a chat turn because retrieval failed.
            Log.w("RustRetrievalProvider", "Retrieval failed; continuing without context", error)
            emptyList()
        }
    }

    /** Open indexes for ready corpora + the embedding context. Caller must hold [loadMutex]. */
    private fun ensureLoadedLocked() {
        for (corpus in corpora) {
            if (!corpus.isReady || indexes.containsKey(corpus.id)) continue
            try {
                // Note: `isReady` only checks file length, so a right-size-but-corrupt
                // index (bad sideload / bit-rot) passes it yet fails to open here. We
                // deliberately do NOT delete the files on failure — open() can also fail
                // transiently under memory pressure (mmap ENOMEM, OOM reading meta.jsonl),
                // and deleting would destroy sideloaded data and force a network
                // re-download. Skip just this corpus; others still serve.
                indexes[corpus.id] = RetrievalIndex.open(corpus.dir.absolutePath)
            } catch (error: Throwable) {
                Log.w("RustRetrievalProvider", "Failed to open ${corpus.id} index; skipping it", error)
            }
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

    /** Free embedding model + context, keeping the indexes. Caller must hold [loadMutex]. */
    private fun releaseEmbeddingLocked() {
        embeddingContext?.destroy()
        embeddingContext = null
        embeddingModel?.destroy()
        embeddingModel = null
    }

    // Retrieval prerequisite bundled with the chat model at first run; datasets
    // are downloaded on demand via downloadCorpus.
    override suspend fun downloadEmbeddingModel(onProgress: (Int) -> Unit): Unit =
        withContext(ioDispatcher) { downloadFiles(listOf(embeddingAsset), onProgress) }

    override fun corpora(): List<CorpusInfo> = corpora.map { c ->
        CorpusInfo(
            id = c.id,
            label = c.label,
            ready = c.isReady,
            // Include the shared embedding model when it's still missing, since
            // downloading this corpus fetches it too — otherwise the size shown
            // would understate the actual download by ~333 MB.
            downloadBytes = c.files.sumOf { it.size } + embeddingDownloadBytesRemaining(),
            downloadable = c.files.all { it.remoteName != null },
        )
    }

    override suspend fun downloadCorpus(corpusId: String, onProgress: (Int) -> Unit): Unit =
        withContext(ioDispatcher) {
            val corpus = corpora.firstOrNull { it.id == corpusId }
                ?: throw IOException("Unknown corpus: $corpusId")
            if (corpus.files.any { it.remoteName == null }) {
                throw IOException("Corpus ${corpus.id} is sideload-only (assets not hosted)")
            }
            // Shared embedding model (once) + this corpus's index files.
            downloadFiles(listOf(embeddingAsset) + corpus.files, onProgress)
        }

    /**
     * Download a set of assets (shared model + one corpus's files), verifying
     * size + SHA-256 before publishing each. Single-flighted via [downloadMutex]
     * so concurrent corpus downloads can't race the shared model file.
     */
    private suspend fun downloadFiles(assets: List<RagAsset>, onProgress: (Int) -> Unit) {
        downloadMutex.withLock {
            if (assets.all { complete(it) }) {
                onProgress(100)
                return@withLock
            }
            retrievalDir.mkdirs()
            assets.forEach { it.target.parentFile?.mkdirs() }

            val needed = assets.filterNot { complete(it) }.sumOf { it.size }
            val available = retrievalDir.usableSpace
            if (available in 1 until (needed + FREE_SPACE_MARGIN)) {
                throw IOException(
                    "Not enough free space: need ~${(needed + FREE_SPACE_MARGIN) / 1_000_000} MB, " +
                        "${available / 1_000_000} MB available"
                )
            }

            val client = OkHttpClient()
            try {
                // Progress is measured against `needed` (bytes still to fetch), not the
                // whole set — otherwise an already-present shared embedding model would
                // make a corpus download start at a large non-zero percent.
                val denom = needed.coerceAtLeast(1L)
                var done = 0L
                onProgress(((done * 100) / denom).toInt().coerceIn(0, 100))

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

                    val request = Request.Builder().url(assetUrl(asset.remoteName!!))
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
                                    onProgress(((done * 100) / denom).toInt().coerceIn(0, 100))
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

    /** Free the embedding model/context and all open indexes (e.g. when retrieval is toggled off). */
    suspend fun release() {
        loadMutex.withLock {
            releaseEmbeddingLocked()
            indexes.values.forEach { it.destroy() }
            indexes.clear()
        }
    }

    private fun complete(asset: RagAsset) = asset.target.exists() && asset.target.length() == asset.size

    private fun partFile(asset: RagAsset) = File(asset.target.parentFile, "${asset.target.name}.part")

    private fun assetUrl(remoteName: String) = "$ASSET_BASE_URL/$remoteName"

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
