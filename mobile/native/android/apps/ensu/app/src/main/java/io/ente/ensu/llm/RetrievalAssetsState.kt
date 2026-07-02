package io.ente.ensu.llm

import io.ente.ensu.AppState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update

/** On-device status of one knowledge corpus's assets (embedding model + index). */
data class RetrievalAssetsState(
    val ready: Boolean = false,
    val downloading: Boolean = false,
    val percent: Int = 0,
    val error: String? = null
)

// Per-corpus retrievalAssets transitions, keyed by corpus id. Every download
// entry point (a Settings row, the bundled-with-model download) reports
// progress/ready/error through this one state machine instead of divergent copies.
private inline fun Map<String, RetrievalAssetsState>.withCorpus(
    corpusId: String,
    transform: (RetrievalAssetsState) -> RetrievalAssetsState,
): Map<String, RetrievalAssetsState> =
    this + (corpusId to transform(this[corpusId] ?: RetrievalAssetsState()))

internal fun MutableStateFlow<AppState>.setRetrievalDownloading(corpusId: String, percent: Int) =
    update { it.copy(retrievalAssets = it.retrievalAssets.withCorpus(corpusId) { c -> c.copy(downloading = true, percent = percent, error = null) }) }

internal fun MutableStateFlow<AppState>.setRetrievalReady(corpusId: String) =
    update { it.copy(retrievalAssets = it.retrievalAssets.withCorpus(corpusId) { c -> c.copy(ready = true, downloading = false, percent = 100, error = null) }) }

internal fun MutableStateFlow<AppState>.setRetrievalError(corpusId: String, message: String) =
    update { it.copy(retrievalAssets = it.retrievalAssets.withCorpus(corpusId) { c -> c.copy(downloading = false, error = message) }) }
