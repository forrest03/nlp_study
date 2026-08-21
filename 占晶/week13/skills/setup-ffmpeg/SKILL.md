---
name: setup-ffmpeg
description: Check whether FFmpeg and FFprobe are ready for local media processing, provide Windows installation guidance, or install FFmpeg with explicit terminal confirmation. Use when users ask about FFmpeg setup or when video processing cannot continue because FFmpeg is missing or lacks subtitle support.
---

# Set Up FFmpeg

Follow this workflow in order and base every status claim on tool output.

1. Call `check_ffmpeg`.
2. If `ok` is true, report the path and version, then call `complete_skill`.
3. If FFmpeg is unavailable or incomplete, ask whether the user wants:
   - a self-installation tutorial;
   - the Agent to install it.
4. For self-installation, call `read_skill_reference` with
   `references/windows-install.md`, present the instructions, then call
   `complete_skill`.
5. For Agent installation, call `install_ffmpeg` only after the user has clearly
   requested automatic installation. The Harness performs the final terminal
   confirmation.
6. After installation, call `check_ffmpeg` again.
7. If installation succeeded but the current process still cannot find FFmpeg,
   tell the user to restart the terminal and run the check again.

Do not invent installation success. Do not construct or execute arbitrary shell
commands.

