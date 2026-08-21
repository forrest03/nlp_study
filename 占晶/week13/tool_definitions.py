from __future__ import annotations


def function_tool(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


ACTIVATE_SKILL = function_tool(
    "activate_skill",
    "Load the complete instructions for one skill from the visible skill catalog.",
    {"name": {"type": "string", "description": "Exact skill name from the catalog."}},
    ["name"],
)

READ_SKILL_REFERENCE = function_tool(
    "read_skill_reference",
    "Read one Markdown reference explicitly mentioned by the active skill.",
    {
        "relative_path": {
            "type": "string",
            "description": "Path such as references/windows-install.md.",
        }
    },
    ["relative_path"],
)

COMPLETE_SKILL = function_tool(
    "complete_skill",
    "Mark the active skill workflow as complete after all requested work has finished.",
    {},
    [],
)

CHECK_FFMPEG = function_tool(
    "check_ffmpeg",
    "Check whether ffmpeg, ffprobe, and subtitle burning support are ready.",
    {},
    [],
)

INSTALL_FFMPEG = function_tool(
    "install_ffmpeg",
    "Install FFmpeg with a fixed winget command. The harness asks for confirmation.",
    {},
    [],
)

CHECK_WHISPER_MODEL = function_tool(
    "check_whisper_model",
    "Check whether a supported local faster-whisper model is downloaded.",
    {
        "model_name": {
            "type": "string",
            "enum": ["tiny", "base", "small", "medium", "large-v3"],
            "description": "Local Whisper model size.",
        }
    },
    ["model_name"],
)

DOWNLOAD_WHISPER_MODEL = function_tool(
    "download_whisper_model",
    "Download a faster-whisper model. The harness asks for confirmation.",
    {
        "model_name": {
            "type": "string",
            "enum": ["tiny", "base", "small", "medium", "large-v3"],
            "description": "Model to download.",
        }
    },
    ["model_name"],
)

TRANSCRIBE_VIDEO = function_tool(
    "transcribe_video",
    "Extract audio, transcribe it, and write SRT plus transcript JSON.",
    {
        "video_path": {
            "type": "string",
            "description": "Absolute or relative local path to the source video.",
        },
        "model_name": {
            "type": "string",
            "enum": ["tiny", "base", "small", "medium", "large-v3"],
            "description": "Downloaded Whisper model to use.",
        },
        "language": {
            "type": ["string", "null"],
            "description": "ISO code such as zh or en, or null for auto detection.",
        },
    },
    ["video_path", "model_name", "language"],
)

BURN_SUBTITLES = function_tool(
    "burn_subtitles",
    "Burn an SRT file into a new MP4 without overwriting the source video.",
    {
        "video_path": {"type": "string", "description": "Source video path."},
        "subtitle_path": {"type": "string", "description": "UTF-8 SRT path."},
        "output_path": {
            "type": ["string", "null"],
            "description": "New MP4 path, or null to use outputs.",
        },
    },
    ["video_path", "subtitle_path", "output_path"],
)

CORE_TOOLS = [ACTIVATE_SKILL]

SKILL_TOOLS = {
    "setup-ffmpeg": [
        ACTIVATE_SKILL,
        READ_SKILL_REFERENCE,
        CHECK_FFMPEG,
        INSTALL_FFMPEG,
        COMPLETE_SKILL,
    ],
    "add-video-subtitles": [
        ACTIVATE_SKILL,
        READ_SKILL_REFERENCE,
        CHECK_FFMPEG,
        INSTALL_FFMPEG,
        CHECK_WHISPER_MODEL,
        DOWNLOAD_WHISPER_MODEL,
        TRANSCRIBE_VIDEO,
        BURN_SUBTITLES,
        COMPLETE_SKILL,
    ],
}
