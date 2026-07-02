package io.ente.ensu.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import io.ente.ensu.designsystem.EnsuColor
import io.ente.ensu.designsystem.EnsuCornerRadius
import io.ente.ensu.designsystem.EnsuSpacing
import io.ente.ensu.designsystem.EnsuTypography
import io.ente.ensu.llm.CorpusInfo
import io.ente.ensu.llm.RetrievalAssetsState

/**
 * Manage on-device knowledge datasets. Each dataset downloads on demand (download
 * icon); once present, a switch turns its use in retrieval on/off. Retrieval draws
 * on whichever datasets are downloaded and switched on — there is no global toggle.
 */
@Composable
fun KnowledgeSettingsScreen(
    corpora: List<CorpusInfo>,
    assets: Map<String, RetrievalAssetsState>,
    disabledCorpora: Set<String>,
    onDownload: (String) -> Unit,
    onToggleEnabled: (corpusId: String, enabled: Boolean) -> Unit,
) {
    Column(modifier = Modifier.padding(EnsuSpacing.pageHorizontal.dp)) {
        Text(
            text = "Ground answers in on-device reference datasets. Download the ones " +
                "you want, then switch each on or off. Answers cite the source used.",
            style = EnsuTypography.small,
            color = EnsuColor.textMuted()
        )
        Spacer(modifier = Modifier.height(EnsuSpacing.lg.dp))

        LazyColumn(verticalArrangement = Arrangement.spacedBy(EnsuSpacing.sm.dp)) {
            items(corpora, key = { it.id }) { corpus ->
                DatasetRow(
                    corpus = corpus,
                    assets = assets[corpus.id],
                    enabled = corpus.id !in disabledCorpora,
                    onDownload = { onDownload(corpus.id) },
                    onToggleEnabled = { on -> onToggleEnabled(corpus.id, on) }
                )
            }
        }
    }
}

@Composable
private fun DatasetRow(
    corpus: CorpusInfo,
    assets: RetrievalAssetsState?,
    enabled: Boolean,
    onDownload: () -> Unit,
    onToggleEnabled: (Boolean) -> Unit,
) {
    // Live state (downloading/error/ready) from the per-corpus asset slice; fall
    // back to the provider's on-disk readiness before the first refresh.
    val ready = assets?.ready ?: corpus.ready
    val downloading = assets?.downloading == true
    val sizeMb = corpus.downloadBytes / 1_000_000L
    val status = when {
        downloading -> "Downloading… ${assets?.percent ?: 0}%"
        assets?.error != null -> "Failed: ${assets.error}"
        ready -> if (enabled) "On" else "Off"
        !corpus.downloadable -> "Not available yet"
        else -> "~$sizeMb MB"
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(EnsuCornerRadius.card.dp))
            .background(EnsuColor.fillFaint())
            .padding(horizontal = EnsuSpacing.lg.dp, vertical = EnsuSpacing.lg.dp),
        horizontalArrangement = Arrangement.spacedBy(EnsuSpacing.md.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = corpus.label, style = EnsuTypography.body, color = EnsuColor.textPrimary())
            Text(text = status, style = EnsuTypography.small, color = EnsuColor.textMuted())
        }
        when {
            downloading -> CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                strokeWidth = 2.dp,
                color = EnsuColor.action()
            )
            ready -> Switch(checked = enabled, onCheckedChange = onToggleEnabled)
            corpus.downloadable -> IconButton(onClick = onDownload) {
                Icon(
                    imageVector = Icons.Outlined.Download,
                    contentDescription = "Download ${corpus.label}",
                    tint = EnsuColor.textPrimary()
                )
            }
        }
    }
}
