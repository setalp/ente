package io.ente.ensu.domain.state

/** On-device status of the Wikipedia retrieval assets (embedding model + index). */
data class RetrievalAssetsState(
    val ready: Boolean = false,
    val downloading: Boolean = false,
    val percent: Int = 0,
    val error: String? = null
)
