package io.ente.ensu.settings

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.stringSetPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import io.ente.ensu.settings.DeveloperSettingsState
import io.ente.ensu.llm.ModelSettingsState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

private val Context.advancedSettingsPreferences by preferencesDataStore("ensu_advanced_settings")

data class AdvancedSettingsSnapshot(
    val developerSettings: DeveloperSettingsState = DeveloperSettingsState(),
    val modelSettings: ModelSettingsState = ModelSettingsState()
)

class AdvancedSettingsDataStore(private val context: Context) {
    private val persistenceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    val settingsFlow: Flow<AdvancedSettingsSnapshot> = context.advancedSettingsPreferences.data.map { prefs ->
        AdvancedSettingsSnapshot(
            developerSettings = DeveloperSettingsState(
                isAdvancedUnlocked = prefs[Keys.advancedUnlocked] ?: false,
                systemPrompt = prefs[Keys.systemPrompt].orEmpty(),
                // Migration: honor a legacy global "Wikipedia retrieval OFF" as
                // Wikipedia disabled, so upgrading users who opted out aren't
                // silently re-enabled. Only applies until disabled_corpora is written.
                disabledCorpora = prefs[Keys.disabledCorpora]
                    ?: if (prefs[Keys.legacyWikipediaEnabled] == false) setOf("wikipedia") else emptySet()
            ),
            modelSettings = ModelSettingsState(
                useCustomModel = prefs[Keys.useCustomModel] ?: false,
                modelUrl = prefs[Keys.modelUrl].orEmpty(),
                mmprojUrl = prefs[Keys.mmprojUrl].orEmpty(),
                contextLength = prefs[Keys.contextLength].orEmpty(),
                maxTokens = prefs[Keys.maxTokens].orEmpty(),
                temperature = prefs[Keys.temperature].orEmpty()
            )
        )
    }

    suspend fun unlockAdvancedSettings() {
        context.advancedSettingsPreferences.edit { prefs ->
            prefs[Keys.advancedUnlocked] = true
        }
    }

    fun persistUnlockAdvancedSettings() {
        persistenceScope.launch {
            unlockAdvancedSettings()
        }
    }

    suspend fun saveSystemPrompt(value: String) {
        context.advancedSettingsPreferences.edit { prefs ->
            prefs[Keys.systemPrompt] = value
        }
    }

    fun persistSystemPrompt(value: String) {
        persistenceScope.launch {
            saveSystemPrompt(value)
        }
    }

    fun persistDisabledCorpora(disabled: Set<String>) {
        persistenceScope.launch {
            context.advancedSettingsPreferences.edit { prefs ->
                prefs[Keys.disabledCorpora] = disabled
            }
        }
    }

    suspend fun saveModelSettings(settings: ModelSettingsState) {
        context.advancedSettingsPreferences.edit { prefs ->
            prefs[Keys.useCustomModel] = settings.useCustomModel
            prefs[Keys.modelUrl] = settings.modelUrl
            prefs[Keys.mmprojUrl] = settings.mmprojUrl
            prefs[Keys.contextLength] = settings.contextLength
            prefs[Keys.maxTokens] = settings.maxTokens
            prefs[Keys.temperature] = settings.temperature
        }
    }

    fun persistModelSettings(settings: ModelSettingsState) {
        persistenceScope.launch {
            saveModelSettings(settings)
        }
    }

    suspend fun resetModelSettings() {
        context.advancedSettingsPreferences.edit { prefs ->
            prefs[Keys.useCustomModel] = false
            prefs[Keys.modelUrl] = ""
            prefs[Keys.mmprojUrl] = ""
            prefs[Keys.contextLength] = ""
            prefs[Keys.maxTokens] = ""
            prefs[Keys.temperature] = ""
        }
    }

    fun persistResetModelSettings() {
        persistenceScope.launch {
            resetModelSettings()
        }
    }

    companion object {
        private object Keys {
            val advancedUnlocked = booleanPreferencesKey("advanced_unlocked")
            val systemPrompt = stringPreferencesKey("system_prompt")
            val disabledCorpora = stringSetPreferencesKey("disabled_corpora")
            // Legacy global toggle, read only for one-time migration to disabledCorpora.
            val legacyWikipediaEnabled = booleanPreferencesKey("wikipedia_retrieval_enabled")
            val useCustomModel = booleanPreferencesKey("use_custom_model")
            val modelUrl = stringPreferencesKey("model_url")
            val mmprojUrl = stringPreferencesKey("mmproj_url")
            val contextLength = stringPreferencesKey("context_length")
            val maxTokens = stringPreferencesKey("max_tokens")
            val temperature = stringPreferencesKey("temperature")
        }
    }
}
