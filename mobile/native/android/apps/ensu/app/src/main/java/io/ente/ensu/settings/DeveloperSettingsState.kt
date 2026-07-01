package io.ente.ensu.settings

data class DeveloperSettingsState(
    val isAdvancedUnlocked: Boolean = false,
    val systemPrompt: String = "",
    // When true (and the index/embedding assets are present), factual queries are
    // augmented with on-device Wikipedia retrieval. Toggle off to compare answers.
    val wikipediaRetrievalEnabled: Boolean = true
)
