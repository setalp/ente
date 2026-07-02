package io.ente.ensu

import android.app.Application
import android.content.pm.PackageManager
import android.os.Build
import android.os.Environment
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.ente.ensu.settings.AdvancedSettingsDataStore
import io.ente.ensu.settings.AdvancedSettingsSnapshot
import io.ente.ensu.device.AndroidDeviceCapabilityProvider
import io.ente.ensu.settings.SessionPreferencesDataStore
import io.ente.ensu.chat.RustChatRepository
import io.ente.ensu.config.RustDefaults
import io.ente.ensu.llm.RustLlmProvider
import io.ente.ensu.llm.RustRetrievalProvider
import io.ente.ensu.logging.FileLogRepository
import io.ente.ensu.storage.CredentialStore
import io.ente.ensu.logging.LogLevel
import io.ente.ensu.AppStore
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File

class AppViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionPreferences = SessionPreferencesDataStore(application)
    val advancedSettingsDataStore = AdvancedSettingsDataStore(application)
    private val credentialStore = CredentialStore(application)
    val appVersion = runCatching { getAppVersion(application) }.getOrDefault("unknown")
    private val deviceCapabilityProvider = AndroidDeviceCapabilityProvider(application)

    val logRepository = FileLogRepository(application)
    private val llmProvider = RustLlmProvider(
        context = application,
        modelDir = resolveModelDir(application),
        legacyModelDir = File(application.filesDir, "llm"),
        deviceCapabilityProvider = deviceCapabilityProvider
    )
    // On-device Wikipedia retrieval. No-ops until the embedding model + index are
    // present under the rag/ dir (sideload via adb push, or in-app download). The
    // similarity gate inside the provider decides when context is actually injected.
    private val retrievalProvider = RustRetrievalProvider(
        embeddingModelPath = File(resolveRetrievalDir(application), EMBEDDING_MODEL_FILE),
        retrievalDir = resolveRetrievalDir(application)
    )
    private val chatRepository = RustChatRepository(application, credentialStore)
    val configDefaults = RustDefaults.load()

    val store = AppStore(
        sessionPreferences = sessionPreferences,
        chatRepository = chatRepository,
        llmProvider = llmProvider,
        retrievalProvider = retrievalProvider,
        deviceCapabilityProvider = deviceCapabilityProvider,
        configDefaults = configDefaults,
        logRepository = logRepository,
        // Debug-only: enables full Q&A logging for RAG analysis (never in release).
        verboseQaLogging = BuildConfig.DEBUG
    )
    init {
        val launchMessage = "App launched app=$appVersion device=${Build.MANUFACTURER} ${Build.MODEL} os=${Build.VERSION.RELEASE} (sdk=${Build.VERSION.SDK_INT})"
        logRepository.log(LogLevel.Info, launchMessage, tag = "App")

        viewModelScope.launch {
            val initialSettings = runCatching {
                advancedSettingsDataStore.settingsFlow.first()
            }.getOrDefault(AdvancedSettingsSnapshot())
            store.applyPersistedSettings(
                developerSettings = initialSettings.developerSettings,
                modelSettings = initialSettings.modelSettings
            )
            store.hydrateModelDownloadRequested(sessionPreferences.modelDownloadRequested.first())
            store.bootstrap(viewModelScope)

            advancedSettingsDataStore.settingsFlow.drop(1).collectLatest { settings ->
                store.applyPersistedSettings(
                    developerSettings = settings.developerSettings,
                    modelSettings = settings.modelSettings
                )
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun getAppVersion(application: Application): String {
        val packageManager = application.packageManager
        val packageName = application.packageName
        val packageInfo = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            packageManager.getPackageInfo(packageName, PackageManager.PackageInfoFlags.of(0))
        } else {
            packageManager.getPackageInfo(packageName, 0)
        }
        val versionName = packageInfo.versionName ?: "unknown"
        val versionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            packageInfo.longVersionCode
        } else {
            packageInfo.versionCode.toLong()
        }
        return "$versionName+$versionCode"
    }

    private fun resolveModelDir(application: Application): File {
        val root = application.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
            ?: application.getExternalFilesDir(null)
            ?: application.filesDir
        return File(root, "llm")
    }

    // Sideload target: adb push the embedding GGUF + index/ here. Resolves to
    // <external-files>/Download/rag/ (Android/data/<pkg>/files/Download/rag).
    private fun resolveRetrievalDir(application: Application): File {
        val root = application.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
            ?: application.getExternalFilesDir(null)
            ?: application.filesDir
        return File(root, "rag")
    }

    companion object {
        private const val EMBEDDING_MODEL_FILE = "embeddinggemma-300M-Q8_0.gguf"
    }

}
