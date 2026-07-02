package io.ente.ensu.settings

data class DeveloperSettingsState(
    val isAdvancedUnlocked: Boolean = false,
    val systemPrompt: String = "",
    // Knowledge corpora the user has turned OFF. A downloaded corpus is used for
    // retrieval unless its id is here (i.e. downloaded => on by default). Retrieval
    // is on/off per dataset now; there is no global toggle.
    val disabledCorpora: Set<String> = emptySet()
)
