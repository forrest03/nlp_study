# Windows FFmpeg installation

## Option 1: WinGet

Open PowerShell and run:

```powershell
winget install --id Gyan.FFmpeg --exact
```

Close and reopen the terminal, then verify:

```powershell
ffmpeg -version
ffprobe -version
```

## Option 2: Manual installation

1. Download a Windows FFmpeg build from the download links on `ffmpeg.org`.
2. Extract it to a stable directory such as `C:\Tools\ffmpeg`.
3. Add the extracted `bin` directory to the user `Path`.
4. Open a new terminal.
5. Run `ffmpeg -version` and `ffprobe -version`.

The build must include the `subtitles` filter (libass) for subtitle burning.

