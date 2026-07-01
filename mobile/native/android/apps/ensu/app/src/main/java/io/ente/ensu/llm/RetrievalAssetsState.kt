package io.ente.ensu.llm

import io.ente.ensu.AppState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update

/** On-device status of the Wikipedia retrieval assets (embedding model + index). */
data class RetrievalAssetsState(
    val ready: Boolean = false,
    val downloading: Boolean = false,
    val percent: Int = 0,
    val error: String? = null
)

// Shared retrievalAssets transitions so every download entry point (the Settings
// row and the bundled-with-model download) reports progress/ready/error through
// one identical state machine instead of divergent copies.
internal fun MutableStateFlow<AppState>.setRetrievalDownloading(percent: Int) =
    update { it.copy(retrievalAssets = it.retrievalAssets.copy(downloading = true, percent = percent, error = null)) }

internal fun MutableStateFlow<AppState>.setRetrievalReady() =
    update { it.copy(retrievalAssets = it.retrievalAssets.copy(ready = true, downloading = false, percent = 100, error = null)) }

internal fun MutableStateFlow<AppState>.setRetrievalError(message: String) =
    update { it.copy(retrievalAssets = it.retrievalAssets.copy(downloading = false, error = message)) }
