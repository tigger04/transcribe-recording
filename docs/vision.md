<!-- Version: 2.0 | Last updated: 2026-07-26 -->

# Vision: local audio transcription and summarization

## Purpose

Transcribe turns local audio or video recordings into useful, portable outputs
without requiring an account or hosted transcription service. It supports
meeting summaries, readable transcripts, subtitles, and detailed timestamp data.

## Product principles

- Local processing is the default. Audio remains on the machine.
- Cloud LLM providers are optional and receive transcript text, not audio.
- Each output has a dedicated subcommand with command-specific help.
- Common processing and speaker options behave consistently across subcommands.
- Outputs use open or widely supported formats.
- Failures identify the affected dependency or invalid input clearly.

## Command outcomes

| Command | Outcome | Default extension |
|---|---|---|
| `transcribe summarize INPUT` | Transcript and LLM-powered meeting summary | `.md` |
| `transcribe text INPUT` | Plain-text or Markdown transcript | `.txt` |
| `transcribe srt INPUT` | SRT subtitles | `.srt` |
| `transcribe vtt INPUT` | WebVTT subtitles | `.vtt` |
| `transcribe words INPUT` | Full whisper.cpp JSON with token timestamps | `.json` |

Converted `docx`, `odt`, `pdf`, and `html` output is available where documented
through Pandoc.

## Timestamped JSON

The `words` subcommand writes full whisper.cpp JSON to a file. Each transcription
segment contains a `tokens` array, and each token carries `timestamps` and numeric
`offsets`. Tokens may represent words, punctuation, or word fragments.

The JSON document is written to the default `<input>.json` path or the path
provided with `--output`. Status and progress messages are not the JSON payload
and must not be redirected into the output file.

## Input handling

- Supported media extensions include `m4a`, `mp4`, `wav`, `mp3`, `opus`, `webm`,
  `aac`, `flac`, `ogg`, and `mov`.
- Audio shorter than ten seconds is rejected.
- `ffprobe` validates duration and media readability.
- `ffmpeg` extracts mono WAV audio and can analyse or preprocess it.
- Whisper models are selected by name and downloaded locally when required.

## Speaker identification

Speaker diarization is optional for every subcommand through `--speakers`.
Speechbrain is the local default. Pyannote can be selected when its model access
and Hugging Face token are configured. Supplied names are assigned in order of
first appearance.

## Summarization

The summary pipeline supports Ollama, Claude, and OpenAI providers. Automatic
selection is local-first and uses Ollama before configured cloud providers unless
`llm_priority` overrides the order.

## Configuration

Configuration is resolved from command-line options, project and user YAML files,
environment variables, command-backed secrets, and compiled defaults. Secrets may
be supplied through environment variables or secret-manager commands rather than
stored directly in YAML.

## Accessibility and interoperability

The current interface is command-line based and scriptable. Output formats are
chosen so transcripts can be read directly, edited in common document tools, used
as subtitles, or processed as structured data.

## Success criteria

- The root help makes every output command discoverable.
- Command-specific help describes output location and relevant options.
- Clear recordings produce readable transcripts with useful timing information.
- Timestamped JSON is valid JSON and exposes token offsets and formatted times.
- Optional dependency failures do not misrepresent the output that was produced.
- Installation through Homebrew or Make preserves the same CLI behaviour.

## Changelog

- 2.0: Replaced the original summary-only vision with the current subcommand product and output contracts.
