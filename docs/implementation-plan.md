<!-- Version: 2.0 | Last updated: 2026-07-26 -->

# Implementation and architecture

## Overview

Transcribe is a Swift command-line application distributed as the `transcribe`
executable. Swift Argument Parser provides subcommand dispatch and option parsing.
The application orchestrates local command-line dependencies for media processing
and transcription, a Python helper for optional speaker diarization, and Swift LLM
providers for summarization.

The deprecated `transcribe-summarize` executable name is installed as a symlink.
It preserves the original summary-oriented invocation and emits a deprecation
warning.

## Runtime architecture

```text
transcribe
├── Swift Argument Parser command dispatch
├── Config and secret resolution
├── AudioExtractor
│   ├── ffprobe media validation
│   └── ffmpeg extraction and preprocessing
├── Transcriber
│   └── whisper-cli transcription and native output
├── Diarizer
│   └── scripts/diarize.py using speechbrain or optional pyannote
├── LLM providers
│   ├── Ollama
│   ├── Claude
│   └── OpenAI
└── Output writers
    ├── Markdown and text
    ├── SRT and WebVTT
    ├── full timestamped JSON from whisper.cpp
    └── Pandoc conversion to docx, odt, pdf, or html
```

## Source layout

| Path | Responsibility |
|---|---|
| `Sources/TranscribeSummarize/main.swift` | Root command and legacy invocation routing |
| `Sources/TranscribeSummarize/HelpText.swift` | Canonical help-document discovery and parsing |
| `Sources/TranscribeSummarize/CommonOptions.swift` | Options shared by all subcommands |
| `Sources/TranscribeSummarize/Commands/` | `summarize`, `text`, `srt`, `vtt`, and `words` pipelines |
| `Sources/TranscribeSummarize/Pipeline/` | Audio extraction, transcription, and diarization orchestration |
| `Sources/TranscribeSummarize/Providers/` | Ollama, Claude, and OpenAI summary providers |
| `Sources/TranscribeSummarize/Output/` | Transcript, subtitle, document, and file writers |
| `scripts/diarize.py` | Python diarization entry point |
| `docs/transcribe-help.md` | Canonical user-facing CLI help text |
| `Tests/TranscribeSummarizeTests/` | XCTest unit and integration coverage |

## Command pipelines

### `summarize`

1. Validate and extract audio with `ffprobe` and `ffmpeg`.
2. Transcribe through `whisper-cli` and parse segment JSON.
3. Apply speaker diarization when requested.
4. Select Ollama, Claude, or OpenAI according to configuration.
5. Generate Markdown and optionally convert it through Pandoc.

### `text`

1. Extract and transcribe the audio.
2. Apply speaker diarization when requested.
3. Render plain text or Markdown with optional segment timestamps.
4. Convert through Pandoc when a document format is selected.

### `srt` and `vtt`

Without speaker names, `whisper-cli` writes the native subtitle format directly.
With speaker names, the application transcribes and diarizes segments before the
Swift subtitle writer adds speaker labels. `--max-len` controls cue line length,
and `--split-on-word` controls boundary-aware splitting.

### `words`

1. Extract audio to temporary WAV.
2. Invoke `whisper-cli` with full JSON output and dynamic time warping.
3. Copy the resulting JSON to `<input>.json` or the explicit `--output` path.
4. When speakers are requested, merge speaker labels into transcription entries.

Timestamped units are exposed in `transcription[].tokens[]`. Whisper tokens may
be complete words, punctuation, or word fragments.

## Help architecture

`docs/transcribe-help.md` is the canonical source for root, subcommand, argument,
option, and flag descriptions. Swift code refers only to stable section keys.
The loader searches the source tree and installed shared-data locations. Make and
Homebrew installations place the canonical file under the application share path,
so help does not depend on the caller's working directory.

## Configuration

Configuration sources, from highest to lowest priority, are:

1. Explicit command-line options.
2. `.transcribe.yaml` in the current directory.
3. `~/.config/transcribe-summarize/config.yaml`.
4. `~/.transcribe.yaml` for legacy compatibility.
5. Relevant environment variables and compiled defaults.

Secret values support environment variables, command-backed resolution, and plain
YAML values in that priority order.

## Dependencies

| Dependency | Purpose | Requirement |
|---|---|---|
| Swift 5.9 or later | Build and runtime application | Build time |
| swift-argument-parser | CLI parsing and generated usage text | Swift package |
| Yams | YAML configuration parsing | Swift package |
| ffmpeg and ffprobe | Media inspection, extraction, and preprocessing | Runtime |
| whisper-cli from whisper.cpp | Transcription, subtitles, and timestamped JSON | Runtime |
| Python 3 | Diarization helper environment | Optional runtime |
| speechbrain | Default local speaker diarization | Optional runtime |
| pyannote.audio | Alternative speaker diarization | Optional runtime |
| Ollama | Default local summarization provider | Summarize runtime |
| Pandoc | Converted document formats | Optional runtime |
| LaTeX engine | PDF conversion through Pandoc | Optional runtime |

Claude and OpenAI are accessed directly through their HTTP APIs when selected;
their API keys are optional configuration.

## Build, test, and installation

- `make build` creates a release executable.
- `make build-debug` creates a debug executable.
- `make test` runs the XCTest suite.
- `make install` installs executable symlinks, the diarization helper, and the
  canonical help document under the user-local prefix.
- The Homebrew formula builds from source and installs the same executable,
  compatibility symlink, helper script, and help document.

## Distribution

The release workflow tests the application, updates the version, tags the release,
updates the Homebrew formula checksum, and publishes the formula to the tap. The
version is defined in the root Swift command configuration.

## Verification boundaries

- Root and subcommand help are checked through the built executable.
- Transcriber integration tests use bundled sample audio and the installed local
  transcription dependency when available.
- Output writers are verified independently for formatting behaviour.
- Full regression verification runs through `make test`.
- Installation verification must run help from outside the repository to prove the
  external help document is installed and discoverable.

## Changelog

- 2.0: Replaced the obsolete scaffold plan with the implemented Swift subcommand architecture and runtime contracts.
