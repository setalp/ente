# Ensu (Wikipedia RAG test build) — install & test

A test build of Ensu with **on-device Wikipedia retrieval**. Answers to factual
questions are grounded in a Simple-English-Wikipedia index that runs entirely on
the phone (no server). See `docs-fork/ensu/SUMMARY.md` for the A/B results.

## Requirements
- **Any Android phone, Android 7+** (the APK is multi-ABI: arm64, armv7, x86_64).
- **~2.5 GB free space** and **Wi-Fi** (first launch downloads ~1.8 GB).

## Install
1. Download **`ensu-test-debug.apk`** from the release page.
2. Open it. Android will ask to **allow installing unknown apps** for your
   browser/Files app — allow it, then **Install**.
3. Open **Ensu** (the icon is labelled Ensu).

## First launch
4. Tap **Download model**. One progress screen runs through:
   - the chat model (~1.2 GB), then
   - **Downloading Wikipedia data…** (~600 MB).
   Wait for **both** — chat opens only when everything is ready.

## Try it
5. Ask factual questions, e.g.:
   - *What is the capital of Kazakhstan?*
   - *Who was Mansa Musa?*
   - *What is the Antikythera mechanism?*
6. To compare with/without retrieval: **Settings → toggle "Wikipedia context"**
   off, ask the same question, then on again. (When on, factual answers are
   grounded in Wikipedia; the toggle takes effect on the next message.)

## Notes
- It's a debug build (package `io.ente.ensu.debug`) — it can coexist with the
  Play-store Ensu.
- Everything runs on-device; you can put the phone in airplane mode after the
  one-time download and it still works.
