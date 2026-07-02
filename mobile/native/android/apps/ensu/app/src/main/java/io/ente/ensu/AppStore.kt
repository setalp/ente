package io.ente.ensu
import io.ente.ensu.llm.ModelSettingsActions
import io.ente.ensu.chat.AttachmentStoreActions
import io.ente.ensu.chat.ChatStoreActions

import io.ente.ensu.chat.RustChatRepository
import io.ente.ensu.device.ChatDeviceCapability
import io.ente.ensu.device.AndroidDeviceCapabilityProvider
import io.ente.ensu.llm.RustLlmProvider
import io.ente.ensu.llm.CorpusInfo
import io.ente.ensu.llm.RetrievalAssetsState
import io.ente.ensu.llm.RetrievalProvider
import io.ente.ensu.llm.setRetrievalDownloading
import io.ente.ensu.llm.setRetrievalReady
import io.ente.ensu.llm.setRetrievalError
import io.ente.ensu.logging.FileLogRepository
import io.ente.ensu.chat.Attachment
import io.ente.ensu.chat.ChatMessage
import io.ente.ensu.config.ConfigDefaults
import io.ente.ensu.logging.LogLevel
import io.ente.ensu.settings.SessionPreferencesDataStore
import io.ente.ensu.AppState
import io.ente.ensu.settings.DeveloperSettingsState
import io.ente.ensu.llm.ModelSettingsState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class AppStore(
    private val sessionPreferences: SessionPreferencesDataStore,
    private val chatRepository: RustChatRepository,
    private val llmProvider: RustLlmProvider,
    private val retrievalProvider: RetrievalProvider? = null,
    private val deviceCapabilityProvider: AndroidDeviceCapabilityProvider,
    val configDefaults: ConfigDefaults,
    private val logRepository: FileLogRepository,
    private val clock: () -> Long = { System.currentTimeMillis() },
    private val verboseQaLogging: Boolean = false
) {
    private val _state = MutableStateFlow(AppState())
    val state: StateFlow<AppState> = _state.asStateFlow()

    private val messageStore = mutableMapOf<String, MutableList<ChatMessage>>()
    private val attachmentActions = AttachmentStoreActions(_state, messageStore)
    private val modelSettingsActions =
        ModelSettingsActions(_state, sessionPreferences, llmProvider, logRepository, configDefaults, retrievalProvider)
    private val chatActions = ChatStoreActions(
        state = _state,
        sessionPreferences = sessionPreferences,
        chatRepository = chatRepository,
        llmProvider = llmProvider,
        clock = clock,
        logRepository = logRepository,
        messageStore = messageStore,
        attachmentActions = attachmentActions,
        modelSettingsActions = modelSettingsActions,
        configDefaults = configDefaults,
        retrievalProvider = retrievalProvider,
        verboseQaLogging = verboseQaLogging
    )

    private var appScope: CoroutineScope? = null

    fun bootstrap(scope: CoroutineScope) {
        appScope = scope
        chatActions.setScope(scope)
        attachmentActions.setScope(scope)
        modelSettingsActions.setScope(scope)
        refreshDeviceCapability(scope)
        chatActions.bootstrap(scope)
        modelSettingsActions.refreshModelDownloadInfo()
        refreshRetrievalReady()
    }

    /** Per-corpus retrieval status, for the Settings data rows. */
    fun retrievalCorpora(): List<CorpusInfo> = retrievalProvider?.corpora() ?: emptyList()

    /** Reflect per-corpus retrieval asset readiness on-device. */
    fun refreshRetrievalReady() {
        val provider = retrievalProvider ?: return
        // corpora() stats every corpus file — keep it off the main thread and out
        // of the update{} lambda (which can re-run under CAS contention).
        val scope = appScope
        if (scope == null) {
            applyRetrievalReadiness(provider.corpora())
        } else {
            scope.launch(Dispatchers.IO) { applyRetrievalReadiness(provider.corpora()) }
        }
    }

    private fun applyRetrievalReadiness(info: List<CorpusInfo>) {
        _state.update { st ->
            st.copy(retrievalAssets = info.associate { c ->
                c.id to (st.retrievalAssets[c.id] ?: RetrievalAssetsState()).copy(ready = c.ready)
            })
        }
    }

    /** Download a corpus's assets (shared embedding model + that corpus's index). */
    fun downloadCorpusAssets(corpusId: String) {
        val provider = retrievalProvider ?: return
        val scope = appScope ?: return
        if (_state.value.retrievalAssets[corpusId]?.downloading == true) return
        _state.setRetrievalDownloading(corpusId, 0)
        scope.launch {
            try {
                provider.downloadCorpus(corpusId) { percent -> _state.setRetrievalDownloading(corpusId, percent) }
                _state.setRetrievalReady(corpusId)
            } catch (error: Throwable) {
                logRepository.log(
                    LogLevel.Error,
                    "Retrieval data download failed",
                    details = "corpus=$corpusId: ${error.message}",
                    tag = "Retrieval"
                )
                _state.setRetrievalError(corpusId, error.message ?: "Download failed")
            }
        }
    }

    fun refreshDeviceCapability(scope: CoroutineScope? = null) {
        val capability = deviceCapabilityProvider.chatCapability()
        val unsupported = capability is ChatDeviceCapability.UnsupportedLowMemory
        _state.value = _state.value.copy(
            chat = _state.value.chat.copy(
                deviceCapability = capability,
                showUnsupportedDeviceDialog = unsupported || _state.value.chat.showUnsupportedDeviceDialog,
                isDownloading = if (unsupported) false else _state.value.chat.isDownloading,
                downloadPercent = if (unsupported) null else _state.value.chat.downloadPercent,
                downloadStatus = if (unsupported) null else _state.value.chat.downloadStatus,
                hasRequestedModelDownload = if (unsupported) false else _state.value.chat.hasRequestedModelDownload,
                editingMessageId = if (unsupported) null else _state.value.chat.editingMessageId,
                messageText = if (unsupported) "" else _state.value.chat.messageText,
                attachments = if (unsupported) emptyList() else _state.value.chat.attachments
            )
        )
        if (unsupported) {
            scope?.let { coroutineScope ->
                coroutineScope.launch {
                    sessionPreferences.setModelDownloadRequested(false)
                }
            }
        }
        logRepository.log(
            LogLevel.Info,
            "Chat device capability evaluated",
            details = "capability=$capability",
            tag = "App"
        )
    }

    fun dismissUnsupportedDeviceDialog() {
        _state.value = _state.value.copy(
            chat = _state.value.chat.copy(showUnsupportedDeviceDialog = false)
        )
    }

    fun hydrateModelDownloadRequested(requested: Boolean) {
        if (!requested) return
        _state.value = _state.value.copy(
            chat = _state.value.chat.copy(
                hasRequestedModelDownload = true,
                isDownloading = true,
                downloadStatus = "Resuming download..."
            )
        )
    }

    fun createNewSession(): String = chatActions.createNewSession()

    fun startNewSessionDraft() = chatActions.startNewSessionDraft()

    fun selectSession(sessionId: String) = chatActions.selectSession(sessionId)

    fun deleteSession(sessionId: String) = chatActions.deleteSession(sessionId)

    fun persistSelectedSession(scope: CoroutineScope, sessionId: String?) =
        chatActions.persistSelectedSession(scope, sessionId)

    fun updateMessageText(value: String) = chatActions.updateMessageText(value)

    fun updateBranchSelection(messageId: String, selectedIndex: Int) =
        chatActions.updateBranchSelection(messageId, selectedIndex)

    fun beginEditing(messageId: String) = chatActions.beginEditing(messageId)

    fun cancelEditing() = chatActions.cancelEditing()

    fun setAttachmentProcessing(isProcessing: Boolean) =
        attachmentActions.setAttachmentProcessing(isProcessing)

    fun addAttachment(attachment: Attachment) = attachmentActions.addAttachment(attachment)

    fun removeAttachment(attachment: Attachment) = attachmentActions.removeAttachment(attachment)

    fun stopGeneration() = chatActions.stopGeneration()

    fun startModelDownload(userInitiated: Boolean = true) =
        modelSettingsActions.startModelDownload(userInitiated)

    fun prewarmImageInferenceIfDownloaded() =
        modelSettingsActions.prewarmImageInferenceIfDownloaded()

    fun refreshModelDownloadInfo() = modelSettingsActions.refreshModelDownloadInfo()

    fun cancelDownload() {
        chatActions.cancelGenerationForDownload()
        modelSettingsActions.cancelModelDownload()
    }

    fun retryAssistantMessage(messageId: String) = chatActions.retryAssistantMessage(messageId)

    fun sendMessage() = chatActions.sendMessage()

    fun confirmOverflowTrim() = chatActions.confirmOverflowTrim()

    fun cancelOverflowDialog() = chatActions.cancelOverflowDialog()

    fun updateModelSettings(state: ModelSettingsState) =
        modelSettingsActions.updateModelSettings(state)

    fun resetModelSettings() = modelSettingsActions.resetModelSettings()

    fun updateDeveloperSettings(state: DeveloperSettingsState) {
        _state.value = _state.value.copy(developerSettings = state)
    }

    fun unlockAdvancedSettings() {
        updateDeveloperSettings(
            _state.value.developerSettings.copy(isAdvancedUnlocked = true)
        )
    }

    fun applyPersistedSettings(
        developerSettings: DeveloperSettingsState,
        modelSettings: ModelSettingsState
    ) {
        _state.value = _state.value.copy(developerSettings = developerSettings)
        modelSettingsActions.hydratePersistedModelSettings(modelSettings)
    }

    fun cancelAttachmentDownload(attachmentId: String) =
        attachmentActions.cancelAttachmentDownload(attachmentId)
}
