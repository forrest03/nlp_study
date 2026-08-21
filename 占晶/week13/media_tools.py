from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"
SUPPORTED_MODELS = {"tiny", "base", "small", "medium", "large-v3"}


def _result(ok: bool, **values: Any) -> dict[str, Any]:
    return {"ok": ok, **values}


def _run_capture(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def check_ffmpeg() -> dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path:
        return _result(
            False,
            installed=False,
            reason="没有在 PATH 中找到 ffmpeg",
            platform=platform.system(),
        )

    version_process = _run_capture([ffmpeg_path, "-version"])
    if version_process.returncode != 0:
        return _result(
            False,
            installed=True,
            path=ffmpeg_path,
            reason="ffmpeg 存在，但无法正常运行",
            stderr=version_process.stderr[-2000:],
        )

    first_line = version_process.stdout.splitlines()[0] if version_process.stdout else ""
    filters_process = _run_capture([ffmpeg_path, "-hide_banner", "-filters"])
    has_subtitles_filter = (
        filters_process.returncode == 0
        and any(" subtitles " in line for line in filters_process.stdout.splitlines())
    )

    ready = bool(ffprobe_path and has_subtitles_filter)
    reason = None
    if not ffprobe_path:
        reason = "没有找到 ffprobe"
    elif not has_subtitles_filter:
        reason = "当前 FFmpeg 构建不包含 subtitles/libass 滤镜"

    return _result(
        ready,
        installed=True,
        ready=ready,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        version=first_line,
        has_subtitles_filter=has_subtitles_filter,
        reason=reason,
    )


def ffmpeg_install_command() -> list[str] | None:
    if platform.system() != "Windows" or not shutil.which("winget"):
        return None
    return [
        "winget",
        "install",
        "--id",
        "Gyan.FFmpeg",
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]


def install_ffmpeg() -> dict[str, Any]:
    command = ffmpeg_install_command()
    if command is None:
        return _result(
            False,
            error={
                "code": "INSTALL_UNAVAILABLE",
                "message": "自动安装目前只支持带 winget 的 Windows",
                "retryable": False,
            },
        )

    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        return _result(
            False,
            command=command,
            error={
                "code": "INSTALL_FAILED",
                "message": f"winget 返回退出码 {process.returncode}",
                "retryable": True,
            },
        )

    status = check_ffmpeg()
    return _result(
        True,
        command=command,
        verification=status,
        note=(
            "安装已完成。如果当前进程仍找不到 FFmpeg，请关闭并重新打开终端。"
            if not status.get("ok")
            else "FFmpeg 已安装并验证。"
        ),
    )


def _validate_model_name(model_name: str) -> str:
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"不支持模型 {model_name!r}；可选：{supported}")
    return model_name


def model_directory(model_name: str) -> Path:
    return MODELS_DIR / _validate_model_name(model_name)


def check_whisper_model(model_name: str) -> dict[str, Any]:
    directory = model_directory(model_name)
    required_files = [directory / "model.bin", directory / "config.json"]
    ready = all(path.is_file() for path in required_files)
    return _result(
        ready,
        model_name=model_name,
        installed=ready,
        path=str(directory),
        missing=[path.name for path in required_files if not path.is_file()],
    )


def download_whisper_model(model_name: str) -> dict[str, Any]:
    model_name = _validate_model_name(model_name)
    status = check_whisper_model(model_name)
    if status["ok"]:
        return _result(
            True,
            model_name=model_name,
            path=status["path"],
            already_present=True,
        )

    from faster_whisper.utils import download_model

    directory = model_directory(model_name)
    directory.mkdir(parents=True, exist_ok=True)
    downloaded_path = download_model(model_name, output_dir=str(directory))
    verified = check_whisper_model(model_name)
    if not verified["ok"]:
        return _result(
            False,
            model_name=model_name,
            downloaded_path=str(downloaded_path),
            error={
                "code": "MODEL_DOWNLOAD_INCOMPLETE",
                "message": "下载结束，但模型文件校验未通过",
                "retryable": True,
            },
        )
    return _result(
        True,
        model_name=model_name,
        path=str(directory),
        already_present=False,
    )


def _unique_output(stem: str, suffix: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = OUTPUTS_DIR / f"{stem}{suffix}"
    index = 2
    while candidate.exists():
        candidate = OUTPUTS_DIR / f"{stem}{index}{suffix}"
        index += 1
    return candidate


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def transcribe_video(
    video_path: str,
    model_name: str,
    language: str | None,
) -> dict[str, Any]:
    ffmpeg_status = check_ffmpeg()
    if not ffmpeg_status.get("ok"):
        return _result(
            False,
            error={
                "code": "FFMPEG_NOT_READY",
                "message": ffmpeg_status.get("reason", "FFmpeg 不可用"),
                "retryable": True,
            },
        )

    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        return _result(
            False,
            error={
                "code": "VIDEO_NOT_FOUND",
                "message": f"视频不存在：{source}",
                "retryable": False,
            },
        )

    model_status = check_whisper_model(model_name)
    if not model_status["ok"]:
        return _result(
            False,
            error={
                "code": "MODEL_NOT_FOUND",
                "message": f"Whisper 模型 {model_name} 尚未下载",
                "retryable": True,
            },
        )

    from faster_whisper import WhisperModel

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temporary:
        audio_path = Path(temporary) / "audio.wav"
        extract = subprocess.run(
            [
                str(ffmpeg_status["ffmpeg_path"]),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if extract.returncode != 0:
            return _result(
                False,
                error={
                    "code": "AUDIO_EXTRACTION_FAILED",
                    "message": extract.stderr[-3000:],
                    "retryable": False,
                },
            )

        model = WhisperModel(
            str(model_directory(model_name)),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )
        segments_iterator, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            log_progress=True,
        )
        segments = list(segments_iterator)

    srt_path = _unique_output(source.stem, ".srt")
    transcript_path = _unique_output(source.stem, ".transcript.json")
    srt_blocks: list[str] = []
    transcript_segments: list[dict[str, Any]] = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        display_index = len(transcript_segments) + 1
        srt_blocks.append(
            "\n".join(
                [
                    str(display_index),
                    f"{_format_srt_time(segment.start)} --> {_format_srt_time(segment.end)}",
                    text,
                ]
            )
        )
        transcript_segments.append(
            {"start": segment.start, "end": segment.end, "text": text}
        )

    srt_path.write_text("\n\n".join(srt_blocks) + "\n", encoding="utf-8")
    transcript_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "model": model_name,
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": transcript_segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return _result(
        True,
        video_path=str(source),
        srt_path=str(srt_path),
        transcript_path=str(transcript_path),
        language=info.language,
        language_probability=info.language_probability,
        segment_count=len(transcript_segments),
    )


def _escape_subtitle_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace("\\", "\\\\")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    value = value.replace("[", r"\[").replace("]", r"\]")
    return value


def burn_subtitles(
    video_path: str,
    subtitle_path: str,
    output_path: str | None,
) -> dict[str, Any]:
    ffmpeg_status = check_ffmpeg()
    if not ffmpeg_status.get("ok"):
        return _result(
            False,
            error={
                "code": "FFMPEG_NOT_READY",
                "message": ffmpeg_status.get("reason", "FFmpeg 不可用"),
                "retryable": True,
            },
        )

    source = Path(video_path).expanduser().resolve()
    subtitles = Path(subtitle_path).expanduser().resolve()
    if not source.is_file():
        return _result(False, error={"code": "VIDEO_NOT_FOUND", "message": str(source)})
    if not subtitles.is_file():
        return _result(
            False,
            error={"code": "SUBTITLE_NOT_FOUND", "message": str(subtitles)},
        )

    if output_path:
        output = Path(output_path).expanduser().resolve()
        if output == source:
            return _result(
                False,
                error={
                    "code": "SOURCE_OVERWRITE_BLOCKED",
                    "message": "不允许覆盖原视频",
                    "retryable": False,
                },
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            return _result(
                False,
                error={
                    "code": "OUTPUT_EXISTS",
                    "message": f"输出文件已存在：{output}",
                    "retryable": False,
                },
            )
    else:
        output = _unique_output(source.stem, ".subtitled.mp4")

    subtitle_filter = (
        f"subtitles=filename='{_escape_subtitle_filter_path(subtitles)}':charenc=UTF-8"
    )
    process = subprocess.run(
        [
            str(ffmpeg_status["ffmpeg_path"]),
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-vf",
            subtitle_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=False,
    )
    if process.returncode != 0 or not output.is_file():
        return _result(
            False,
            error={
                "code": "SUBTITLE_BURN_FAILED",
                "message": f"FFmpeg 返回退出码 {process.returncode}",
                "retryable": False,
            },
        )

    return _result(
        True,
        output_path=str(output),
        source_path=str(source),
        subtitle_path=str(subtitles),
    )

