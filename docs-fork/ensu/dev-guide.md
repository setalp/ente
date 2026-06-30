# Ensu — Fork Developer Guide

> Fork-local notes for working on **Ensu**, Ente's local-first, on-device AI chat app.
> This file lives under `docs-fork/` to keep it separate from upstream's user-facing
> `docs/`. Safe to edit freely; it won't conflict with upstream merges.

## What Ensu is

A chat app that runs a small language model **on the device** — no server in the loop
for inference. Shared **Rust core** (llama.cpp under the hood) wrapped per platform:

- **iOS / Android** — Rust core as a native lib, Swift / Kotlin UI on top.
- **Desktop (macOS / Windows / Linux)** — Tauri shell + Rust core + the Ensu web UI.
- **Web** — everything in-browser; model runs as WASM (`@wllama/wllama`), cached in OPFS.

App identity: web `io.ente.ensu`, desktop `io.ente.ensu.desktop`. Accent color is
yellow/gold (`#f5d93a`). Current native version line: `0.1.18-beta`.

User-facing docs (upstream): `docs/docs/ensu/` — `index.md`, `how-it-works.md`,
`faq/`, `features/`, `changelog.md`. Read `how-it-works.md` first; it's the best
mental model of the runtime.

## Repository map

### Web (`web/`)
- `web/apps/ensu/` — the Next.js app (package name `ensu`).
  - `src/pages/` — `index.tsx`, `chat.tsx` (main UI), auth pages (`login`, `credentials`,
    `verify`, `passkeys/`, `two-factor/`), `_app.tsx`.
  - `src/services/chat/` — `gateway.ts`, `sync.ts`, `store.ts`, `chatKey.ts`,
    `crypto.ts`, `attachments.ts`, `branching.ts`.
  - `src/services/llm/` — `inference.ts`, `provider.ts`, `types.ts` (local inference).
  - `src/services/` — `tauri-runtime.ts`, `secure-storage.ts`, `session.ts`, `wasm.ts`,
    `featureFlags.ts`, `app-update.ts`, `whats-new*.ts`.
  - `src/components/chat/` — `ChatComposer`, `ChatMessageList`, `ChatSidebar`,
    `ChatDialogs`, Rive logo/indicator components.
- Shared packages wiring Ensu in:
  - `web/packages/base/app.ts` — `appNames`, titles, `clientPackageName` (`io.ente.ensu`).
  - `web/packages/base/env.ts` — `NEXT_PUBLIC_ENSU_DESKTOP_VERSION`.
  - `web/packages/base/components/utils/theme.ts` — `accentEnsu`, `lightEnsu`, `darkEnsu`.
  - `web/packages/accounts*/services/redirect.ts` — `ensu` → `/` home route.
  - `web/packages/accounts*/components/LoginContents.tsx` — `isEnsu` login UI branch.

### Rust (`rust/`) — the shared core
- `rust/crates/ensu/db/` — `ensu-db`: encrypted chat DB, attachments, image compression.
- `rust/crates/ensu/sync/` — `ensu-sync`: chat synchronization.
- `rust/crates/ensu/inference/` — `inference_rs` (a.k.a. ensu-inference): LLM engine + `ensu_defaults()`.
- `rust/crates/ensu/transcription/` — `ensu-transcription`: voice transcription, VAD model.
- `rust/bindings/uniffi/ensu/{db,inference,sync,transcription}/` — bindings for mobile.
- `rust/apps/ensu/src-tauri/` — desktop Tauri backend (`tauri.conf.json`, `main.rs`,
  `commands.rs`). Dev URL `http://localhost:3010`; updater feed
  `https://ente.com/release-info/ensu-desktop.json`.

### Mobile (`mobile/native/`)
- Android: `mobile/native/android/apps/ensu/` — gradle modules `app-ui/`, `data/`,
  `domain/`, `crypto-auth-core/`. Key files: `EnsuCryptoBridge.kt`,
  `AndroidDeviceCapabilityProvider.kt`, `EnsuRustDefaults.kt`. Pkg `io.ente.ensu`.
- iOS: `mobile/native/darwin/Apps/Ensu/` — SwiftUI. `EnsuApp.swift`, `EnsuTheme.swift`,
  `EnsuAuthService.swift`, `EnsuRustDefaults.swift`. URL scheme `enteensu`.

### CI / build (`.github/workflows/`)
- `ensu-build.yml` (desktop), `ensu-android-build.yml`, `ensu-ios-build.yml`,
  `app-release.yml` (runs `.github/scripts/ensu-version.mjs`, release notes in
  `rust/apps/ensu/changes`), `warm-caches.yml`, `web-deploy.yml` (`app: ensu`).

## Running it locally

### Web
```bash
cd web
npm install            # first time
npm run dev:ensu       # builds wasm, then next dev on http://localhost:3007
npm run build:ensu     # production build
```

### Desktop (Tauri)
```bash
cd rust/apps/ensu
npm install
npm run dev            # tauri dev; expects the web dev server (see tauri.conf.json devUrl)
npm run build          # tauri build
```

### Rust core (tests / build)
```bash
cd rust
cargo build -p ensu-db -p ensu-sync -p ensu-transcription -p inference_rs
cargo test  -p ensu-sync     # crate-scoped tests
```

## Where data lives (local, on-device)
- macOS: `~/Library/Application Support/io.ente.ensu/`
- Windows: `%APPDATA%\io.ente.ensu\`
- Linux: `~/.local/share/io.ente.ensu/`

## Fork workflow reminders
- `origin` → `setalp/ente` (our fork). `upstream` → `ente-io/ente` (fetch only; push DISABLEd).
- Sync from upstream: `git fetch upstream && git merge upstream/main` (or `gh repo sync`).
- Keep fork-only notes in `docs-fork/`; avoid editing upstream `docs/` to reduce conflicts.

## Open questions / scratchpad
<!-- Track what we're actually changing here as we go. -->
- [ ] Goal of this fork's Ensu work: _TBD_
