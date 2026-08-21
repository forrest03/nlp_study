# FFmpeg prerequisite on Windows

Use WinGet:

```powershell
winget install --id Gyan.FFmpeg --exact
```

Then open a new terminal and verify:

```powershell
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -filters
```

The filter list must contain `subtitles`.

