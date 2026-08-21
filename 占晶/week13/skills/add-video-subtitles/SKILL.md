---
name: add-video-subtitles
description: Transcribe speech from a local video with a downloaded faster-whisper model, create UTF-8 SRT and transcript files, and burn the subtitles into a new MP4 with FFmpeg. Use when users ask to generate, add, or permanently burn subtitles for a local video.
---

# Add Video Subtitles

Follow this workflow sequentially. After every tool call, inspect its result and
replan instead of assuming success.

1. Obtain the local video path. Ask if it is missing or ambiguous.
2. Call `check_ffmpeg`.
3. If FFmpeg is unavailable:
   - ask whether the user wants installation guidance or automatic installation;
   - for guidance, read `references/windows-ffmpeg.md`;
   - for automatic installation, call `install_ffmpeg` only after the user asks
     for it; the Harness performs the final confirmation;
   - call `check_ffmpeg` again before continuing.
4. Use `small` as the default Whisper model unless the user requests another
   model. If the user asks about model tradeoffs, read
   `references/model-selection.md`.
5. Call `check_whisper_model`.
6. If the model is missing, explain that it must be downloaded and ask for
   consent. After consent, call `download_whisper_model`; the Harness performs a
   final confirmation.
7. Call `transcribe_video`. Use `zh` when the user explicitly requests Chinese
   speech recognition; otherwise use `null` for automatic language detection.
8. When transcription succeeds, call `burn_subtitles` using the returned SRT
   path. Pass `null` as `output_path` unless the user provides a new path.
9. Report the SRT, transcript JSON, and new video paths.
10. Call `complete_skill` only after the requested output is complete or the
    user cancels.

Never overwrite the source video. Do not claim to translate non-Chinese speech
into Chinese; this version performs transcription, not subtitle translation.
Do not retry a non-retryable tool error without changing the plan.

