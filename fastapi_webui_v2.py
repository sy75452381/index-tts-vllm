
#!/usr/bin/env python3
"""
FastAPI Web Interface for IndexTTS vLLM v2
A single-file FastAPI application that combines webui_with_presets.py functionality
with the API structure from deploy_vllm_indextts.py, using IndexTTS vLLM v2 as backend.

Features:
- IndexTTS vLLM v2 backend for ultra-fast inference
- Speaker preset management with persistent storage
- API compatibility for external integrations
- Modern web interface with Chinese support
- Parallel chunk processing for long texts
- MP3 output support
- Advanced translate/edit mode with segment editing and selective regeneration
- Gemini model selection (Flash vs Pro) with optional API key override
- Translation/transcription toggle with custom prompt support
- Per-segment generation control for efficient audio processing
"""

from __future__ import annotations

import os
import sys
import asyncio
import time
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional, Literal, Tuple, Set, Callable, Awaitable, Union
import logging
import multiprocessing

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

PERF_LOGGER = logging.getLogger("perf")
if not PERF_LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    PERF_LOGGER.addHandler(handler)
PERF_LOGGER.setLevel(logging.INFO)
PERF_LOGGER.propagate = False
from contextlib import asynccontextmanager
from io import BytesIO
import base64
import functools
import json
import math
import re
import shutil
import subprocess
import urllib.request
import hashlib
import zipfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy
from dataclasses import dataclass, field, replace

# yt_dlp is imported lazily via _ensure_yt_dlp() to prevent its compat_utils
# module patching from interfering with torch.compile / torch._inductor code
# hashing (causes ModuleNotFoundError: No module named 'no_Cryptodome').
yt_dlp = None


def _ensure_yt_dlp():
    """Lazily import yt_dlp on first use and return the module (or None)."""
    global yt_dlp
    if yt_dlp is None:
        try:
            import yt_dlp as _yt_dlp  # type: ignore[import]
            yt_dlp = _yt_dlp
        except ImportError:
            pass
    return yt_dlp


# Audio processing
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa

try:
    from clearvoice import ClearVoice  # type: ignore[import]
except ImportError:
    ClearVoice = None

try:
    from audio_separator.separator import Separator as AudioSeparator  # type: ignore[import]
except ImportError:
    AudioSeparator = None

try:
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]
except ImportError:
    genai = None
    types = None


# FastAPI and web interface
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, validator
from urllib.parse import quote

# IndexTTS v2 and speaker management
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "indextts"))


def _load_project_env_files() -> None:
    """Load simple KEY=VALUE entries from project .env files without overriding env."""
    for env_path in (Path(current_dir) / ".env", Path(current_dir) / ".env.local"):
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception as exc:
            print(f"Warning: failed to load {env_path}: {exc}")


_load_project_env_files()

from indextts.infer_vllm_v2 import IndexTTS2
from speaker_preset_manager import SpeakerPresetManager, initialize_preset_manager
from qwen3_tts_manager import Qwen3VoiceDesignManager, Qwen3TTSConfig, DesignedVoiceResult
from stable_audio3_manager import (
    STABLE_AUDIO3_DEFAULT_VARIANT,
    STABLE_AUDIO3_OUTPUT_FORMATS,
    STABLE_AUDIO3_SAMPLERS,
    StableAudio3GenerateOptions,
    StableAudio3Manager,
)

# WhisperX local transcription pipeline (optional)
try:
    from whisperx_pipeline import (
        is_whisperx_available,
        _run_whisperx_pipeline_sync,
    )
except ImportError:
    def is_whisperx_available() -> bool:
        return False
    _run_whisperx_pipeline_sync = None  # type: ignore[assignment]

# Qwen3-ASR + OmniVAD transcription pipeline (optional)
try:
    from qwen_omnivad_pipeline import (
        QWEN_OMNIVAD_MERGE_GAP_SECONDS as DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS,
        is_qwen_omnivad_available,
        _run_qwen_omnivad_pipeline_sync,
    )
except ImportError:
    DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS = 0.001
    def is_qwen_omnivad_available() -> bool:
        return False
    _run_qwen_omnivad_pipeline_sync = None  # type: ignore[assignment]

# NVIDIA Parakeet NeMo transcription pipeline (optional)
try:
    from parakeet_pipeline import (
        is_parakeet_available,
        _run_parakeet_pipeline_sync,
    )
except ImportError:
    def is_parakeet_available() -> bool:
        return False
    _run_parakeet_pipeline_sync = None  # type: ignore[assignment]

# Configuration
import argparse


ALLOWED_TRANSCRIPTION_PIPELINES = {"gemini", "whisperx", "qwen_omnivad", "parakeet"}


def _normalize_transcription_pipeline(value: Any, default: str = "gemini") -> str:
    pipeline = (str(value or "")).strip().lower()
    return pipeline if pipeline in ALLOWED_TRANSCRIPTION_PIPELINES else default


def _env_flag(name: str, default: bool) -> bool:
    """Parse boolean environment flags with sane defaults."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int = 1, max_value: Optional[int] = None) -> int:
    """Parse integer environment values with bounds."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    parsed = max(min_value, parsed)
    if max_value is not None and parsed > max_value:
        return max_value
    return parsed


def _env_ffmpeg_threads(name: str = "FFMPEG_THREADS") -> int:
    """Resolve FFmpeg worker threads; 0/auto lets FFmpeg choose."""
    cpu_threads = max(1, os.cpu_count() or 1)
    value = os.environ.get(name)
    if value is None:
        return cpu_threads
    normalized = value.strip().lower()
    if normalized in {"0", "auto"}:
        return 0
    try:
        parsed = int(normalized)
    except (TypeError, ValueError):
        return cpu_threads
    if parsed <= 0:
        return 0
    return min(parsed, cpu_threads)


@dataclass(frozen=True)
class AppSettings:
    verbose: bool = False
    port: int = 8000
    host: str = "0.0.0.0"
    model_dir: str = "checkpoints"
    is_fp16: bool = False
    use_torch_compile: bool = False
    gpu_memory_utilization: float = 0.25

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "AppSettings":
        return cls(
            verbose=bool(args.verbose),
            port=int(args.port),
            host=str(args.host),
            model_dir=str(args.model_dir),
            is_fp16=bool(args.is_fp16),
            use_torch_compile=bool(args.use_torch_compile),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
        )

# Global thread executor for blocking operations
# Use CPU count for parallel audio processing, but cap at 8 to avoid excessive context switching
_executor_workers = min(8, max(4, (os.cpu_count() or 4)))
executor = ThreadPoolExecutor(max_workers=_executor_workers, thread_name_prefix="fastapi_async")

# Global ClearVoice models (initialized lazily and reused)
_enhancement_model: Optional[Any] = None
_super_res_model: Optional[Any] = None
_current_enhancement_model_name: Optional[str] = None

# Global Qwen3-TTS Voice Design manager (initialized lazily)
_voice_design_manager: Optional[Qwen3VoiceDesignManager] = None

# Available ClearVoice speech enhancement models
# MossFormerGAN_SE_16K is the default (smaller and faster)
# MossFormer2_SE_48K is the original model (larger, higher quality at 48kHz)
# FRCRN_SE_16K is another smaller/faster option
AVAILABLE_ENHANCEMENT_MODELS = ["MossFormerGAN_SE_16K", "FRCRN_SE_16K", "MossFormer2_SE_48K"]
DEFAULT_ENHANCEMENT_MODEL = "MossFormerGAN_SE_16K"

# Audio-Separator configuration for vocal/instrumental separation
# Models: quality (best SDR), balance (good quality/speed), fast (fastest)
AUDIO_SEPARATOR_MODELS = {
    "quality": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",  # Best quality
    "balance": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # Balance of quality/speed (default)
    "fast": "UVR-MDX-NET-Inst_full_292.onnx",  # Fastest
}
DEFAULT_AUDIO_SEPARATOR_MODEL = "balance"
AUDIO_SEPARATOR_RAW_OUTPUT_FORMAT = "mp3"
AUDIO_SEPARATOR_USE_SOUNDFILE = _env_flag("AUDIO_SEPARATOR_USE_SOUNDFILE", False)
# Global audio-separator instance (initialized lazily and reused)
_audio_separator: Optional[Any] = None
_audio_separator_model_name: Optional[str] = None
_audio_separator_output_format: Optional[str] = None
_audio_separator_use_soundfile: Optional[bool] = None

AUDIO_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
}


def _audio_media_type(audio_format: Optional[str]) -> str:
    normalized = (audio_format or "").strip().lower()
    return AUDIO_MEDIA_TYPES.get(normalized, f"audio/{normalized or 'mpeg'}")


@dataclass
class AudioAssetInfo:
    duration_ms: int
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    hash_md5: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "frame_count": self.frame_count,
            "hash_md5": self.hash_md5,
        }

    @staticmethod
    def from_dict(payload: Optional[Dict[str, Any]]) -> Optional["AudioAssetInfo"]:
        if not payload:
            return None
        return AudioAssetInfo(
            duration_ms=int(payload.get("duration_ms") or 0),
            sample_rate=int(payload.get("sample_rate") or 0),
            channels=int(payload.get("channels") or 0),
            sample_width=int(payload.get("sample_width") or 0),
            frame_count=int(payload.get("frame_count") or 0),
            hash_md5=payload.get("hash_md5"),
        )


@dataclass
class AudioBuffer:
    path: str
    sample_rate: int
    channels: int
    frame_count: int
    duration_ms: int
    dtype: str = "float32"
    analysis_samples: Optional[np.ndarray] = field(default=None, repr=False)
    analysis_sample_rate: Optional[int] = None
    analysis_downsample_factor: int = 1

    def ensure_analysis_samples(
        self,
        target_sample_rate: int = 16_000,
    ) -> Tuple[np.ndarray, int, int]:
        if (
            self.analysis_samples is not None
            and self.analysis_sample_rate == target_sample_rate
        ):
            return (
                self.analysis_samples,
                self.analysis_sample_rate,
                self.analysis_downsample_factor,
            )
        samples, sample_rate, downsample_factor = _read_analysis_samples_from_path(
            self.path,
            target_sample_rate=target_sample_rate,
        )
        self.analysis_samples = samples
        self.analysis_sample_rate = sample_rate
        self.analysis_downsample_factor = downsample_factor
        return samples, sample_rate, downsample_factor


@dataclass
class SplitClearVoiceAssets:
    processed_audio_path: str
    processed_audio_info: AudioAssetInfo
    clearvoice_hash: str
    processed_mime_type: str = "audio/mpeg"
    backing_track_path: Optional[str] = None
    backing_audio_info: Optional[AudioAssetInfo] = None
    backing_track_source: str = "none"
    applied_super_resolution: bool = False


class PerfLogger:
    """Lightweight helper to print per-step timings for complex workflows."""

    def __init__(self, label: str):
        self.label = label
        self.records: List[Dict[str, Any]] = []

    def rename(self, label: str) -> None:
        self.label = label

    def mark(self, step_name: str, start_time: float, *, extra: Optional[str] = None) -> float:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.records.append({"step": step_name, "elapsed_ms": elapsed_ms, "extra": extra})
        extra_text = f" | {extra}" if extra else ""
        PERF_LOGGER.info("⏱️ [%s] %s took %.1f ms%s", self.label, step_name, elapsed_ms, extra_text)
        return elapsed_ms

    def summary(self) -> None:
        if not self.records:
            return
        total_ms = sum(record["elapsed_ms"] for record in self.records)
        PERF_LOGGER.info(
            "⏱️ [%s] summary → %s step(s), total %.1f ms",
            self.label,
            len(self.records),
            total_ms,
        )
        for record in self.records:
            extra_text = f" | {record['extra']}" if record.get("extra") else ""
            PERF_LOGGER.info("   • %s: %.1f ms%s", record["step"], record["elapsed_ms"], extra_text)

# Gemini configuration
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
GEMINI_MODEL_ENV_VAR = "GEMINI_MODEL_NAME"
DEFAULT_GEMINI_MODEL_NAME = "gemini-3.1-flash-lite-preview"
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
COERCE_SPEAKER_SEGMENTS_PATTERN = re.compile(
    r'^\s*(\[[\s\S]*?\])\s*,\s*"segments"\s*:\s*(\[[\s\S]*\])\s*$',
    re.DOTALL,
)
TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional interpreter and diarization analyst. "
    "Transcribe the speech from the provided audio and translate it into {dest_language} using natural, conversational wording. "
    "{non_speech_instruction}"
    "CRITICAL: Conversations may contain rapid handoffs without pauses. Detect every speaker change by listening for voice characteristics (pitch, timbre, accent, cadence) and never merge two speakers into one segment. "
    "Return JSON with exactly two top-level keys: \"speakers\" and \"segments\". "
    "\"speakers\" must be an array ordered by first appearance where every entry has "
    "\"id\" (deterministic labels \"speaker1\", \"speaker2\", ...), "
    "and \"description\" (≤12 words highlighting gender, approximate age range, and personality or tone). "
    "\"segments\" must be an array where each item represents a contiguous utterance with the keys "
    "\"start\" (timestamp in mm:ss.xxx format with millisecond precision), "
    "\"end\" (same format, mm:ss.xxx), "
    "\"speaker\" (one of the ids from the speakers list), "
    "\"source_text\" (original-language transcript), "
    "\"translated_text\" (translation in {dest_language}). "
    "TIMESTAMP PRECISION IS CRITICAL - these timestamps will be used to cut audio clips for voice cloning: "
    "Always use millisecond format (mm:ss.xxx). "
    "\"start\" must be the EXACT moment the speaker begins speaking (first syllable onset). "
    "\"end\" must be the EXACT moment the speaker finishes speaking (last syllable offset, before any silence or next speaker). "
    "Do NOT pad timestamps with extra silence. Do NOT round to whole seconds. "
    "Each segment's boundaries should tightly enclose only that speaker's utterance. "
    "Add a new speaker entry whenever a new voice appears. "
    "Respond with JSON only—no explanations, markdown, or additional prose."
)
TRANSCRIPTION_PROMPT_TEMPLATE = (
    "You are a meticulous transcription and diarization expert. "
    "Transcribe the provided speech audio in its original language without translating it. "
    "{non_speech_instruction}"
    "CRITICAL: Conversations may contain rapid speaker transitions without pauses. Detect every speaker change by voice characteristics and never mix two speakers in one segment. "
    "Return JSON with the top-level keys \"speakers\" and \"segments\". "
    "\"speakers\" is an ordered array of objects containing "
    "\"id\" (speaker1, speaker2, ... in order of first appearance) "
    "and \"description\" (short summary focusing on gender, approximate age, and personality/tone). "
    "\"segments\" is an array where each entry contains "
    "\"start\" (timestamp in mm:ss.xxx format with millisecond precision), "
    "\"end\" (same format, mm:ss.xxx), "
    "\"speaker\" (speaker id), "
    "\"source_text\" (transcript in the original language), "
    "\"translated_text\" (use empty string \"\" because no translation is requested). "
    "TIMESTAMP PRECISION IS CRITICAL - these timestamps will be used to cut audio clips for voice cloning: "
    "Always use millisecond format (mm:ss.xxx). "
    "\"start\" must be the EXACT moment the speaker begins speaking (first syllable onset). "
    "\"end\" must be the EXACT moment the speaker finishes speaking (last syllable offset, before any silence or next speaker). "
    "Do NOT pad timestamps with extra silence. Do NOT round to whole seconds. "
    "Each segment's boundaries should tightly enclose only that speaker's utterance. "
    "Respond with JSON only—no markdown or commentary."
)
IGNORE_NON_SPEECH_PROMPT_SUFFIX = (
    "Ignore non-speech vocalizations (laughter, shouts, chants, or crowd noise) and any background voices that are not clear speech; only capture spoken dialogue."
)
NON_SPEECH_PROMPT_PLACEHOLDER = "{non_speech_instruction}"
DEFAULT_GEMINI_TEMPERATURE = 0.2
DEFAULT_GEMINI_TOP_P = 0.9
TRANSLATE_DEFAULT_OUTPUT_FORMAT = "mp3"
TRANSLATE_DEFAULT_BITRATE = "128k"
CHUNK_SESSION_CREATION_CONCURRENCY = 4
AUDIO_GENERATION_MARGIN_MS = 20
TRANSLATION_TTS_CONCURRENCY = 100
MIN_SPEECH_DURATION_MS = 3000
MAX_MERGE_INTERVAL_MS = 0
DEFAULT_GENERATED_VOLUME_PERCENT = 100.0
MIN_GENERATED_VOLUME_PERCENT = 10.0
MAX_GENERATED_VOLUME_PERCENT = 300.0
DEFAULT_SILENCE_VOLUME_PERCENT = DEFAULT_GENERATED_VOLUME_PERCENT
DEFAULT_EMOTION_WEIGHT = 0.6


def _normalize_translate_output_format(_value: Optional[str] = None) -> str:
    """Translation outputs are always MP3; soundfile does not support M4A."""
    return TRANSLATE_DEFAULT_OUTPUT_FORMAT


ALLOWED_GEMINI_MODELS = {
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
}
GEMINI_AUDIO_EXPORT_BITRATE = "128k"
GEMINI_CACHE_VERSION = 1
SPEAKER_PREVIEW_DIR = Path("speaker_presets") / "previews"
SPEAKER_PREVIEW_BITRATE = "128k"
CHUNK_SPLIT_DEFAULT_MIN_MINUTES = 10.0
CHUNK_SPLIT_DEFAULT_MAX_MINUTES = 15.0
CHUNK_SPLIT_MIN_MINUTES = 1.0
CHUNK_SPLIT_MAX_MINUTES = 45.0
CHUNK_SPLIT_MIN_SILENCE_MS = 500
CHUNK_SPLIT_SILENCE_THRESHOLD_DB = -42.0
CHUNK_SPLIT_MAX_CHUNKS = 64
CHUNK_SPLIT_SILENCE_GRACE_MS = 45000
CHUNK_SPLIT_MIN_CHUNK_MS = 1000
CHUNK_BATCH_GENERATE_DELAY_SECONDS = 60
TRANSLATE_USE_FFMPEG_SPLIT_MERGE = _env_flag("TRANSLATE_USE_FFMPEG_SPLIT_MERGE", True)
CHUNK_SPLIT_FFMPEG_CONCURRENCY = _env_int("CHUNK_SPLIT_FFMPEG_CONCURRENCY", 2, min_value=1, max_value=8)
FFMPEG_THREADS = _env_ffmpeg_threads()
VIDEO_SUBTITLE_FONT = os.environ.get("VIDEO_SUBTITLE_FONT", "").strip()
VIDEO_SUBTITLE_FONT_FILE = os.environ.get("VIDEO_SUBTITLE_FONT_FILE", "").strip()
VIDEO_SUBTITLE_FONTS_DIR = os.environ.get("VIDEO_SUBTITLE_FONTS_DIR", "").strip()
VIDEO_SUBTITLE_FONT_SIZE = _env_int("VIDEO_SUBTITLE_FONT_SIZE", 24, min_value=8, max_value=96)

CLEARVOICE_PARALLEL_MIN_CHUNK_SECONDS = 300
CLEARVOICE_PARALLEL_MAX_CHUNK_SECONDS = 1800
CLEARVOICE_PARALLEL_MAX_WORKERS = 5
CLEARVOICE_PARALLEL_DEFAULT_CHUNK_SECONDS = _env_int(
    "CLEARVOICE_PARALLEL_CHUNK_SECONDS",
    300,
    min_value=CLEARVOICE_PARALLEL_MIN_CHUNK_SECONDS,
    max_value=CLEARVOICE_PARALLEL_MAX_CHUNK_SECONDS,
)
CLEARVOICE_PARALLEL_DEFAULT_WORKERS = _env_int(
    "CLEARVOICE_PARALLEL_WORKERS",
    5,
    min_value=1,
    max_value=CLEARVOICE_PARALLEL_MAX_WORKERS,
)
CLEARVOICE_PARALLEL_MAX_CHUNKS = 120


@dataclass
class ClearVoiceParallelConfig:
    enabled: bool = False
    chunk_seconds: int = CLEARVOICE_PARALLEL_DEFAULT_CHUNK_SECONDS
    max_workers: int = CLEARVOICE_PARALLEL_DEFAULT_WORKERS

    @property
    def chunk_ms(self) -> int:
        return max(1000, int(self.chunk_seconds) * 1000)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "parallel_enabled": bool(self.enabled),
            "parallel_chunk_seconds": int(self.chunk_seconds),
            "parallel_max_workers": int(self.max_workers),
        }


@dataclass
class AudioPreprocessOptions:
    apply_enhancement: bool
    apply_super_resolution: bool
    enhancement_model_name: str
    audio_separator_enabled: bool
    audio_separator_model: str
    audio_separator_use_soundfile: bool
    parallel_config: ClearVoiceParallelConfig

    def to_clearvoice_settings(self) -> Dict[str, Any]:
        settings: Dict[str, Any] = {
            "enhancement": self.apply_enhancement,
            "enhancement_model": self.enhancement_model_name,
            "super_resolution": self.apply_super_resolution,
            "audio_separator_enabled": self.audio_separator_enabled,
            "audio_separator_model": self.audio_separator_model,
            "audio_separator_use_soundfile": self.audio_separator_use_soundfile,
        }
        if self.parallel_config.enabled:
            settings.update(self.parallel_config.to_metadata())
        return settings


@dataclass
class VolumeOptions:
    generated: float
    backing: float
    silence: float


@dataclass
class ClearVoiceParallelChunkJob:
    chunk_idx: int
    chunk_path: str
    apply_enhancement: bool
    apply_super_resolution: bool
    enhancement_model_name: Optional[str] = None


@dataclass
class ClearVoiceParallelChunkResult:
    chunk_idx: int
    final_path: str
    enhancement_path: Optional[str]
    generated_paths: List[str]


async def _run_blocking(func: Callable, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))


def _normalize_gemini_model_name(gemini_model_value: Optional[str]) -> str:
    sanitized = (gemini_model_value or "").strip()
    if sanitized and sanitized not in ALLOWED_GEMINI_MODELS:
        return _get_gemini_model_name()
    return sanitized or _get_gemini_model_name()


ALLOWED_TRANSLATION_LLM_MODELS: Tuple[str, ...] = (
    "tencent/Hy-MT2-1.8B",
    "lightning-ai/gemma-4-31B-it",
    "lightning-ai/gpt-oss-20b",
    "lightning-ai/gpt-oss-120b",
    "lightning-ai/minimax-m2.5",
)
DEFAULT_TRANSLATION_LLM_MODEL = "tencent/Hy-MT2-1.8B"


def _normalize_translation_llm_model(model_value: Optional[str]) -> str:
    sanitized = (model_value or "").strip()
    if sanitized in ALLOWED_TRANSLATION_LLM_MODELS:
        return sanitized
    return DEFAULT_TRANSLATION_LLM_MODEL


def _coerce_merge_backing_flag(
    requested: bool,
    apply_enhancement: bool,
    alternate_backing_available: bool = False,
    audio_separator_enabled: bool = False,
) -> bool:
    if not requested:
        return False
    if audio_separator_enabled or apply_enhancement or alternate_backing_available:
        return True
    print("⚠️ Merge-back requested without audio-separator, ClearVoice enhancement, or custom/reused backing; ignoring request.")
    return False


def _resolve_final_prompt(
    prompt_override: Optional[str],
    dest_language: str,
    translate_enabled: bool,
    ignore_non_speech_flag: bool,
) -> str:
    non_speech_instruction_text = f"{IGNORE_NON_SPEECH_PROMPT_SUFFIX} " if ignore_non_speech_flag else ""
    final_prompt = (prompt_override or "").strip()
    if not final_prompt:
        if translate_enabled:
            final_prompt = TRANSLATION_PROMPT_TEMPLATE.format(
                dest_language=dest_language,
                non_speech_instruction=non_speech_instruction_text,
            )
        else:
            final_prompt = TRANSCRIPTION_PROMPT_TEMPLATE.format(
                non_speech_instruction=non_speech_instruction_text,
            )
        return final_prompt.strip()

    placeholder_present = NON_SPEECH_PROMPT_PLACEHOLDER in final_prompt
    final_prompt = final_prompt.replace(
        NON_SPEECH_PROMPT_PLACEHOLDER,
        non_speech_instruction_text,
    ).strip()
    if ignore_non_speech_flag and not placeholder_present:
        final_prompt = f"{final_prompt.rstrip()} {IGNORE_NON_SPEECH_PROMPT_SUFFIX}".strip()
    return final_prompt


def _decode_audio_segment_sync(audio_bytes: bytes, audio_format: Optional[str]) -> AudioSegment:
    audio_buffer = BytesIO(audio_bytes)
    if audio_format:
        return AudioSegment.from_file(audio_buffer, format=audio_format)
    return AudioSegment.from_file(audio_buffer)


async def _decode_audio_segment(audio_bytes: bytes, audio_format: Optional[str]) -> AudioSegment:
    return await _run_blocking(_decode_audio_segment_sync, audio_bytes, audio_format)


def _export_audio_segment_bytes_sync(audio: AudioSegment, fmt: str = "mp3", bitrate: Optional[str] = None) -> bytes:
    """Export audio segment to bytes. Uses FFmpeg for long audio (>5 min) for better performance."""
    duration_sec = len(audio) / 1000.0
    
    # Use FFmpeg for long audio files (>5 minutes) - much faster
    if duration_sec > 300 and _ffmpeg_available():
        return _export_audio_via_ffmpeg_sync(audio, fmt, bitrate)
    
    with BytesIO() as buffer:
        export_kwargs: Dict[str, Any] = {"format": fmt}
        if bitrate and fmt in {"mp3", "ogg", "opus", "aac"}:
            export_kwargs["bitrate"] = bitrate
        audio.export(buffer, **export_kwargs)
        return buffer.getvalue()


def _export_audio_via_ffmpeg_sync(audio: AudioSegment, fmt: str = "mp3", bitrate: Optional[str] = None) -> bytes:
    """Export audio via FFmpeg for better performance on long files."""
    temp_input = None
    temp_output = None
    try:
        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            temp_input = tmp_in.name
        audio.export(temp_input, format="wav")
        
        # Create temp output file
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp_out:
            temp_output = tmp_out.name
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", temp_input]
        
        if fmt == "mp3":
            cmd.extend(["-codec:a", "libmp3lame"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "aac":
            cmd.extend(["-codec:a", "aac"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "opus":
            cmd.extend(["-codec:a", "libopus"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "ogg":
            cmd.extend(["-codec:a", "libvorbis"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "flac":
            cmd.extend(["-codec:a", "flac"])
        # WAV - no additional codec needed

        cmd.extend(_ffmpeg_thread_args())
        cmd.append(temp_output)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg export failed: {result.stderr[:200]}")
        
        with open(temp_output, "rb") as f:
            return f.read()
    finally:
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except:
                pass
        if temp_output and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass


async def _export_audio_segment_bytes(audio: AudioSegment, fmt: str = "mp3", bitrate: Optional[str] = None) -> bytes:
    return await _run_blocking(_export_audio_segment_bytes_sync, audio, fmt, bitrate)


def _export_audio_segment_to_path_sync(
    audio: AudioSegment,
    path: str,
    fmt: str = "mp3",
    bitrate: Optional[str] = None,
) -> str:
    """Export audio segment to path. Uses FFmpeg for long audio (>5 min) for better performance."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    duration_sec = len(audio) / 1000.0
    
    # Use FFmpeg for long audio files (>5 minutes) - much faster
    if duration_sec > 300 and _ffmpeg_available():
        return _export_audio_to_path_via_ffmpeg_sync(audio, path, fmt, bitrate)
    
    audio.export(path, format=fmt, bitrate=bitrate)
    return path


def _export_audio_to_path_via_ffmpeg_sync(
    audio: AudioSegment,
    path: str,
    fmt: str = "mp3",
    bitrate: Optional[str] = None,
) -> str:
    """Export audio to path via FFmpeg for better performance on long files."""
    temp_input = None
    try:
        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            temp_input = tmp_in.name
        audio.export(temp_input, format="wav")
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", temp_input]
        
        if fmt == "mp3":
            cmd.extend(["-codec:a", "libmp3lame"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "aac":
            cmd.extend(["-codec:a", "aac"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "opus":
            cmd.extend(["-codec:a", "libopus"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "ogg":
            cmd.extend(["-codec:a", "libvorbis"])
            if bitrate:
                cmd.extend(["-b:a", bitrate])
        elif fmt == "flac":
            cmd.extend(["-codec:a", "flac"])

        cmd.extend(_ffmpeg_thread_args())
        cmd.append(path)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg export failed: {result.stderr[:200]}")
        
        return path
    finally:
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except:
                pass


async def _export_audio_segment_to_path(
    audio: AudioSegment,
    path: str,
    fmt: str = "mp3",
    bitrate: Optional[str] = None,
) -> str:
    return await _run_blocking(_export_audio_segment_to_path_sync, audio, path, fmt, bitrate)


def _export_audio_segment_to_tempfile_sync(audio: AudioSegment, suffix: str = ".wav") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        audio.export(tmp_file.name, format="wav")
        return tmp_file.name


async def _export_audio_segment_to_tempfile(audio: AudioSegment, suffix: str = ".wav") -> str:
    return await _run_blocking(_export_audio_segment_to_tempfile_sync, audio, suffix)


def _load_audio_segment_from_path_sync(path: str) -> AudioSegment:
    """Load audio segment from file path, auto-detecting format from extension."""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if ext == "mp3":
        return AudioSegment.from_file(path, format="mp3")
    elif ext == "wav":
        return AudioSegment.from_file(path, format="wav")
    else:
        return AudioSegment.from_file(path)


async def _load_audio_segment_from_path(path: str) -> AudioSegment:
    return await _run_blocking(_load_audio_segment_from_path_sync, path)


# ==============================================================================
# Audio-Separator: Vocal/Instrumental separation using audio-separator library
# ==============================================================================

@dataclass
class AudioSeparatorResult:
    """Result of audio-separator separation operation."""
    vocals_path: str
    instrumental_path: str
    vocals_audio: Optional[AudioSegment] = None
    instrumental_audio: Optional[AudioSegment] = None
    cache_hash: str = ""
    from_cache: bool = False


def _get_audio_separator_model_path(model_key: str) -> str:
    """Get the model filename for a given model key."""
    return AUDIO_SEPARATOR_MODELS.get(model_key, AUDIO_SEPARATOR_MODELS[DEFAULT_AUDIO_SEPARATOR_MODEL])


def _get_audio_separator_sync(
    model_key: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
    use_soundfile: Optional[bool] = None,
) -> Any:
    """Initialize or return cached audio-separator instance (blocking)."""
    global _audio_separator, _audio_separator_model_name, _audio_separator_output_format, _audio_separator_use_soundfile
    
    if AudioSeparator is None:
        raise RuntimeError(
            "audio-separator is not installed. Install with: pip install audio-separator[gpu]"
        )
    
    model_filename = _get_audio_separator_model_path(model_key)
    resolved_use_soundfile = AUDIO_SEPARATOR_USE_SOUNDFILE if use_soundfile is None else bool(use_soundfile)
    
    # Return existing instance if same model is loaded
    if (
        _audio_separator is not None
        and _audio_separator_model_name == model_key
        and _audio_separator_output_format == AUDIO_SEPARATOR_RAW_OUTPUT_FORMAT
        and _audio_separator_use_soundfile == resolved_use_soundfile
    ):
        return _audio_separator
    
    # Always use the base AUDIO_SEPARATOR_CACHE_DIR as output
    # We'll move files to the per-hash cache directory after separation
    print(
        f"🎛️ Initializing audio-separator with model: {model_filename} "
        f"(format={AUDIO_SEPARATOR_RAW_OUTPUT_FORMAT}, use_soundfile={resolved_use_soundfile})"
    )
    _audio_separator = AudioSeparator(
        log_level=logging.INFO,
        model_file_dir=AUDIO_SEPARATOR_MODEL_DIR,
        output_dir=AUDIO_SEPARATOR_CACHE_DIR,  # Fixed output directory
        output_format=AUDIO_SEPARATOR_RAW_OUTPUT_FORMAT,
        use_soundfile=resolved_use_soundfile,
        use_autocast=True,  # Faster GPU inference
    )
    _audio_separator.load_model(model_filename=model_filename)
    _audio_separator_model_name = model_key
    _audio_separator_output_format = AUDIO_SEPARATOR_RAW_OUTPUT_FORMAT
    _audio_separator_use_soundfile = resolved_use_soundfile
    print(f"✅ Audio-separator model loaded: {model_filename}")
    return _audio_separator


async def _get_audio_separator(model_key: str = DEFAULT_AUDIO_SEPARATOR_MODEL) -> Any:
    """Async wrapper to get audio-separator instance."""
    return await _run_blocking(_get_audio_separator_sync, model_key)


def _audio_separator_cache_paths(cache_hash: str, model_key: str) -> Tuple[str, str, str]:
    """Get cache paths for audio-separator results."""
    cache_dir = os.path.join(AUDIO_SEPARATOR_CACHE_DIR, f"{cache_hash}_{model_key}")
    vocals_path = os.path.join(cache_dir, "vocals.mp3")
    instrumental_path = os.path.join(cache_dir, "instrumental.mp3")
    return cache_dir, vocals_path, instrumental_path


def _audio_separator_cached_mp3_is_usable(path: str) -> bool:
    if not os.path.exists(path):
        return False
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return path.lower().endswith(".mp3")
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=format_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False
        formats = {item.strip().lower() for item in result.stdout.split(",") if item.strip()}
        return "mp3" in formats
    except Exception:
        return False


def _cache_audio_separator_output_as_mp3(source_path: str, target_path: str, label: str) -> None:
    """Store an audio-separator stem as real MP3 instead of renaming the container."""
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"Audio-separator {label} output not found: {source_path}")

    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    if _audio_separator_cached_mp3_is_usable(source_path):
        if os.path.abspath(source_path) != os.path.abspath(target_path):
            os.replace(source_path, target_path)
        print(f"Audio-separator: cached {label} MP3: {target_path}")
        return

    fd, temp_target = tempfile.mkstemp(
        prefix=f"{Path(target_path).stem}_",
        suffix=".mp3",
        dir=os.path.dirname(target_path) or None,
    )
    os.close(fd)
    try:
        if not _transcode_file_with_ffmpeg(source_path, temp_target, "mp3"):
            audio = AudioSegment.from_file(source_path)
            audio.export(temp_target, format="mp3", bitrate="192k")
        os.replace(temp_target, target_path)
        if os.path.abspath(source_path) != os.path.abspath(target_path):
            _safe_remove_file(source_path)
        print(f"Audio-separator: cached {label} as MP3: {target_path}")
    except Exception:
        _safe_remove_file(temp_target)
        raise


def _run_audio_separator_sync(
    input_path: str,
    model_key: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
    cache_hash: Optional[str] = None,
    use_soundfile: Optional[bool] = None,
) -> AudioSeparatorResult:
    """
    Run audio-separator on input audio file to separate vocals and instrumentals.
    Uses caching based on input file MD5 hash.
    """
    # Compute cache hash if not provided
    if cache_hash is None:
        cache_hash = _compute_file_md5(input_path)
    
    cache_dir, cached_vocals_path, cached_instrumental_path = _audio_separator_cache_paths(cache_hash, model_key)
    
    # Check cache
    if (
        os.path.exists(cached_vocals_path)
        and os.path.exists(cached_instrumental_path)
        and _audio_separator_cached_mp3_is_usable(cached_vocals_path)
        and _audio_separator_cached_mp3_is_usable(cached_instrumental_path)
    ):
        print(f"♻️ Audio-separator: Using cached results for {cache_hash[:8]}...")
        return AudioSeparatorResult(
            vocals_path=cached_vocals_path,
            instrumental_path=cached_instrumental_path,
            cache_hash=cache_hash,
            from_cache=True,
        )
    if os.path.exists(cached_vocals_path) or os.path.exists(cached_instrumental_path):
        print(f"Audio-separator cache for {cache_hash[:8]} is not valid MP3; regenerating.")
        _safe_remove_file(cached_vocals_path)
        _safe_remove_file(cached_instrumental_path)
    
    # Create cache directory for this hash
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get separator instance (uses fixed AUDIO_SEPARATOR_CACHE_DIR for output)
    separator = _get_audio_separator_sync(model_key, use_soundfile=use_soundfile)
    
    print(f"🎵 Running audio-separator ({model_key}) on: {input_path}")
    
    # Use simple output names - separator will place them in AUDIO_SEPARATOR_CACHE_DIR
    output_names = {
        "Vocals": "vocals",
        "Instrumental": "instrumental",
    }
    
    try:
        output_files = separator.separate(input_path, output_names)
        print(f"✅ Audio-separator completed. Output files: {output_files}")
        
        # Files are saved to AUDIO_SEPARATOR_CACHE_DIR; find them and cache real MP3 copies.
        vocals_file = None
        instrumental_file = None
        
        for f in output_files:
            basename = os.path.basename(f)
            # Check in the fixed output directory
            full_path = os.path.join(AUDIO_SEPARATOR_CACHE_DIR, basename)
            if not os.path.exists(full_path):
                # Try the path as returned
                full_path = f if os.path.isabs(f) else os.path.join(AUDIO_SEPARATOR_CACHE_DIR, f)
            
            f_lower = basename.lower()
            if "vocal" in f_lower and os.path.exists(full_path):
                vocals_file = full_path
                print(f"📁 Found vocals at: {vocals_file}")
            elif ("instrumental" in f_lower or "instrum" in f_lower) and os.path.exists(full_path):
                instrumental_file = full_path
                print(f"📁 Found instrumental at: {instrumental_file}")
        
        if vocals_file is None or instrumental_file is None:
            # List files to help debug
            if os.path.exists(AUDIO_SEPARATOR_CACHE_DIR):
                files = os.listdir(AUDIO_SEPARATOR_CACHE_DIR)
                print(f"📂 Files in {AUDIO_SEPARATOR_CACHE_DIR}: {files}")
            raise RuntimeError(f"Could not find output files. Returned: {output_files}")
        
        # Transcode to per-hash MP3 cache; do not just rename M4A/MP4 stems.
        if vocals_file != cached_vocals_path:
            _cache_audio_separator_output_as_mp3(vocals_file, cached_vocals_path, "vocals")
            print(f"Audio-separator: cached vocals MP3: {cached_vocals_path}")
        if instrumental_file != cached_instrumental_path:
            _cache_audio_separator_output_as_mp3(instrumental_file, cached_instrumental_path, "instrumental")
            print(f"Audio-separator: cached instrumental MP3: {cached_instrumental_path}")
        
        return AudioSeparatorResult(
            vocals_path=cached_vocals_path,
            instrumental_path=cached_instrumental_path,
            cache_hash=cache_hash,
            from_cache=False,
        )
    except Exception as exc:
        print(f"❌ Audio-separator failed: {exc}")
        raise RuntimeError(f"Audio-separator separation failed: {exc}") from exc


async def _run_audio_separator(
    input_path: str,
    model_key: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
    cache_hash: Optional[str] = None,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    use_soundfile: Optional[bool] = None,
) -> AudioSeparatorResult:
    """
    Async wrapper for audio-separator separation.
    Separates audio into vocals and instrumentals with caching.
    """
    if AudioSeparator is None:
        raise TranslateWorkflowHttpError(
            500,
            {
                "status": "error",
                "message": "audio-separator is not installed. Install with: pip install audio-separator[gpu]",
            },
        )
    
    model_name = _get_audio_separator_model_path(model_key)
    if emit_status:
        await emit_status(
            stage="audio_separation",
            message=f"Separating vocals/instrumentals using {model_key} model ({model_name})...",
        )
    
    result = await _run_blocking(_run_audio_separator_sync, input_path, model_key, cache_hash, use_soundfile)
    
    if emit_status:
        cache_note = " (cached)" if result.from_cache else ""
        await emit_status(
            stage="audio_separation",
            message=f"Audio separation complete{cache_note}. Vocals and instrumentals separated.",
        )
    
    return result


@dataclass
class AudioSeparatorPipelineResult:
    """Result of the audio-separator pipeline with both AudioSegments and file paths."""
    vocals_audio: AudioSegment
    instrumental_audio: AudioSegment
    vocals_path: str
    instrumental_path: str
    cache_hash: str
    from_cache: bool = False


async def _run_audio_separator_pipeline(
    source_audio_path: str,
    *,
    model_key: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    cache_hash: Optional[str] = None,
    skip_audio_loading: bool = False,
    use_soundfile: Optional[bool] = None,
) -> AudioSeparatorPipelineResult:
    """
    Run the full audio-separator pipeline on a source audio file.
    
    Args:
        cache_hash: Optional pre-computed hash of the original audio (for consistent caching
                    even when the input file is normalized/converted)
        skip_audio_loading: If True, skip loading AudioSegments (for when only paths are needed)
    
    Returns:
        AudioSeparatorPipelineResult with audio segments, paths, and cache info
    """
    # Use provided cache_hash or compute from file
    if cache_hash is None:
        cache_hash = _compute_file_md5(source_audio_path)
    
    # Run separation
    result = await _run_audio_separator(
        source_audio_path,
        model_key=model_key,
        cache_hash=cache_hash,
        emit_status=emit_status,
        use_soundfile=use_soundfile,
    )
    
    if skip_audio_loading:
        # Return empty AudioSegments when loading is skipped
        # Caller should use the paths directly
        return AudioSeparatorPipelineResult(
            vocals_audio=AudioSegment.empty(),
            instrumental_audio=AudioSegment.empty(),
            vocals_path=result.vocals_path,
            instrumental_path=result.instrumental_path,
            cache_hash=result.cache_hash,
            from_cache=result.from_cache,
        )
    
    # Load the separated audio segments
    print(f"[audio-separator] Loading vocals from: {result.vocals_path}")
    load_start = time.perf_counter()
    vocals_audio = await _load_audio_segment_from_path(result.vocals_path)
    vocals_elapsed = (time.perf_counter() - load_start) * 1000
    print(f"[audio-separator] Vocals loaded ({len(vocals_audio) / 1000:.1f}s) in {vocals_elapsed:.1f}ms")
    
    print(f"[audio-separator] Loading instrumentals from: {result.instrumental_path}")
    load_start = time.perf_counter()
    instrumental_audio = await _load_audio_segment_from_path(result.instrumental_path)
    instrumental_elapsed = (time.perf_counter() - load_start) * 1000
    print(f"[audio-separator] Instrumentals loaded ({len(instrumental_audio) / 1000:.1f}s) in {instrumental_elapsed:.1f}ms")
    
    if emit_status:
        await emit_status(
            stage="audio_separation",
            message="Separated audio tracks loaded.",
        )
    
    return AudioSeparatorPipelineResult(
        vocals_audio=vocals_audio,
        instrumental_audio=instrumental_audio,
        vocals_path=result.vocals_path,
        instrumental_path=result.instrumental_path,
        cache_hash=result.cache_hash,
        from_cache=result.from_cache,
    )


# ==============================================================================


async def _run_clearvoice_pipeline(
    original_audio: AudioSegment,
    *,
    apply_enhancement: bool,
    apply_super_resolution: bool,
    pre_clearvoice_mix_audio: Optional[AudioSegment],
    emit_status: Optional[Callable[..., Awaitable[None]]],
    source_audio_path: Optional[str] = None,
    clearvoice_parallel_config: Optional[ClearVoiceParallelConfig] = None,
    enhancement_model_name: Optional[str] = None,
) -> Tuple[AudioSegment, Optional[AudioSegment]]:
    if not (apply_enhancement or apply_super_resolution):
        return original_audio, None
    if ClearVoice is None:
        raise TranslateWorkflowHttpError(
            500,
            {
                "status": "error",
                "message": "ClearVoice package is required for enhancement or super-resolution.",
            },
        )

    processed_paths: Set[str] = set()
    # Determine the enhancement model to use
    effective_model = enhancement_model_name if enhancement_model_name in AVAILABLE_ENHANCEMENT_MODELS else DEFAULT_ENHANCEMENT_MODEL
    try:
        if emit_status:
            action = f"Applying {effective_model} enhancement..."
            if apply_super_resolution and not apply_enhancement:
                action = "Applying MossFormer2_SR_48K super-resolution..."
            elif apply_super_resolution and apply_enhancement:
                action = f"Applying {effective_model} enhancement + SR_48K super-resolution..."
            await emit_status(stage="enhancement", message=action)

        wav_input_path: Optional[str] = None
        if source_audio_path:
            wav_input_path, created_temp = await _ensure_clearvoice_input_path(source_audio_path)
            if created_temp and wav_input_path:
                processed_paths.add(wav_input_path)
                temp_input_path = wav_input_path
            else:
                temp_input_path = wav_input_path or source_audio_path
        else:
            temp_input_path = await _export_audio_segment_to_tempfile(original_audio)
            processed_paths.add(temp_input_path)

        total_duration_ms = int(len(original_audio) if original_audio is not None else 0)
        parallel_config = (
            clearvoice_parallel_config if clearvoice_parallel_config and clearvoice_parallel_config.enabled else None
        )
        planned_chunks: List[Tuple[int, int]] = []
        if parallel_config and (apply_enhancement or apply_super_resolution):
            planned_chunks = _plan_clearvoice_parallel_chunks(total_duration_ms, parallel_config.chunk_ms)
            if len(planned_chunks) <= 1:
                parallel_config = None
        else:
            parallel_config = None

        cache_hash = _compute_file_md5(temp_input_path)
        cache_dir, cached_enhanced_path, cached_sr_path, cached_backing_path = _clearvoice_cache_paths(
            cache_hash, effective_model
        )
        os.makedirs(cache_dir, exist_ok=True)
        use_cached_enhancement = apply_enhancement and os.path.exists(cached_enhanced_path)
        use_cached_sr = apply_super_resolution and os.path.exists(cached_sr_path)
        clearvoice_paths: List[str] = []

        final_processed_path: Optional[str] = None
        enhancement_output_path: Optional[str] = None

        if apply_super_resolution:
            if use_cached_sr and (not apply_enhancement or use_cached_enhancement):
                final_processed_path = cached_sr_path
                enhancement_output_path = cached_enhanced_path if apply_enhancement else None
                print(f"♻️ ClearVoice: Reusing cached MossFormer2_SR_48K output for {cache_hash}.")
        elif apply_enhancement and use_cached_enhancement:
            final_processed_path = cached_enhanced_path
            enhancement_output_path = cached_enhanced_path
            print(f"♻️ ClearVoice: Reusing cached {effective_model} output for {cache_hash}.")

        if final_processed_path is None:
            if parallel_config:
                if emit_status:
                    await emit_status(
                        stage="enhancement",
                        message=f"Running ClearVoice in parallel across {len(planned_chunks)} chunk(s)...",
                    )
                else:
                    print(f"⚡ ClearVoice: running parallel ClearVoice across {len(planned_chunks)} chunk(s).")
                final_processed_local_path, clearvoice_paths, enhancement_output_local = await _run_blocking(
                    functools.partial(
                        _apply_clearvoice_parallel_sync,
                        temp_input_path,
                        total_duration_ms,
                        apply_enhancement,
                        apply_super_resolution,
                        parallel_config,
                        enhancement_model_name=effective_model,
                    ),
                )
            else:
                final_processed_local_path, clearvoice_paths, enhancement_output_local = await apply_clearvoice_processing(
                    temp_input_path,
                    apply_enhancement,
                    apply_super_resolution,
                    enhancement_model_name=effective_model,
                )
            processed_paths.update(clearvoice_paths)
            processed_paths.add(final_processed_local_path)
            final_processed_path = final_processed_local_path
            enhancement_output_path = enhancement_output_local

            if apply_enhancement and enhancement_output_local:
                enhancement_output_path = await _normalize_audio_to_cached(
                    enhancement_output_local,
                    cached_enhanced_path,
                    delete_source=True,
                )
                print(f"💾 ClearVoice: Cached {effective_model} output for {cache_hash}.")

            if apply_super_resolution:
                final_processed_path = await _normalize_audio_to_cached(
                    final_processed_local_path,
                    cached_sr_path,
                    delete_source=True,
                )
                print(f"💾 ClearVoice: Cached MossFormer2_SR_48K output for {cache_hash}.")
            elif apply_enhancement and enhancement_output_path is not None:
                final_processed_path = enhancement_output_path
            else:
                final_processed_path = await _normalize_audio_to_cached(
                    final_processed_local_path,
                    cached_enhanced_path,
                    delete_source=True,
                )
        else:
            # Ensure enhancement path is set when reusing cache
            if apply_enhancement and enhancement_output_path is None and os.path.exists(cached_enhanced_path):
                enhancement_output_path = cached_enhanced_path

        processed_audio = await _load_audio_segment_from_path(final_processed_path)

        backing_track_audio: Optional[AudioSegment] = None
        if os.path.exists(cached_backing_path):
            try:
                backing_track_audio = await _load_audio_segment_from_path(cached_backing_path)
                print(
                    "♻️ ClearVoice: Reusing cached backing track for %s (%.2fs)"
                    % (cache_hash, len(backing_track_audio) / 1000.0)
                )
            except Exception as backing_exc:
                print(f"⚠️ Failed to load cached backing track for {cache_hash}: {backing_exc}")
                backing_track_audio = None
        if apply_enhancement and pre_clearvoice_mix_audio is not None:
            enhancement_audio: Optional[AudioSegment] = None
            if backing_track_audio is None:
                if enhancement_output_path is None:
                    print("⚠️ Enhancement output path not available, cannot extract backing track.")
                else:
                    # Avoid duplicate load: if final_processed_path == enhancement_output_path, reuse processed_audio
                    if final_processed_path == enhancement_output_path:
                        enhancement_audio = processed_audio
                        print(f"[clearvoice] Reusing already-loaded processed audio for backing extraction")
                    else:
                        try:
                            enhancement_audio = await _load_audio_segment_from_path(enhancement_output_path)
                        except Exception as enhancement_load_error:
                            print(f"⚠️ Failed to load ClearVoice enhancement output: {enhancement_load_error}")
                if enhancement_audio is None:
                    print("⚠️ Cannot extract backing track: enhancement audio not available.")
                else:
                    backing_track_audio = _extract_backing_track_from_vocals(
                        pre_clearvoice_mix_audio,
                        enhancement_audio,
                    )
                    if backing_track_audio is not None:
                        backing_dbfs = backing_track_audio.dBFS
                        if math.isinf(backing_dbfs):
                            backing_dbfs = -120.0
                        print(
                            "🎼 Extracted backing track %.2fs @ %d Hz (%d ch, %.1f dBFS)"
                            % (
                                len(backing_track_audio) / 1000.0,
                                backing_track_audio.frame_rate,
                                backing_track_audio.channels,
                                backing_dbfs,
                            )
                        )
                        try:
                            backing_track_audio.export(cached_backing_path, format="mp3")
                            print(f"💾 ClearVoice: Cached backing track for {cache_hash}.")
                        except Exception as cache_exc:
                            print(f"⚠️ Failed to cache backing track for {cache_hash}: {cache_exc}")
                    if backing_track_audio is not None and emit_status:
                        sr_note = " (super-resolution applied after extraction)" if apply_super_resolution else ""
                        await emit_status(
                            stage="enhancement",
                            message=f"Extracted instrumental backing track via {effective_model}{sr_note}.",
                        )

        if backing_track_audio is not None:
            reference_audio = pre_clearvoice_mix_audio or processed_audio
            reference_ms = len(reference_audio)
            backing_ms = len(backing_track_audio)
            if abs(backing_ms - reference_ms) > 1000:
                print(
                    f"⚠️ Backing track duration mismatch (backing={backing_ms} ms vs reference={reference_ms} ms) for cache {cache_hash}."
                )
        if emit_status:
            await emit_status(stage="enhancement", message="ClearVoice enhancement complete.")
        return processed_audio, backing_track_audio
    except TranslateWorkflowHttpError:
        raise
    except Exception as exc:
        raise TranslateWorkflowHttpError(
            500,
            {"status": "error", "message": f"ClearVoice processing failed: {str(exc)}"},
        ) from exc
    finally:
        for path in processed_paths:
            try:
                os.remove(path)
            except Exception:
                pass


async def _prepare_audio_assets(
    *,
    reuse_source_session: Optional[TranslateSessionData],
    audio_file: Optional[UploadFile],
    audio_reference: Optional[str],
    preloaded_audio_bytes: Optional[bytes] = None,
    source_audio_filename: Optional[str] = None,
    audio_mime_type_value: Optional[str],
    apply_enhancement: bool,
    apply_super_resolution: bool,
    requested_merge_backing: bool,
    custom_backing_audio_file: Optional[UploadFile] = None,
    custom_backing_audio_reference: Optional[str] = None,
    custom_backing_mime_type_value: Optional[str] = None,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    clearvoice_parallel_config: Optional[ClearVoiceParallelConfig] = None,
    enhancement_model_name: Optional[str] = None,
    audio_separator_enabled: bool = False,
    audio_separator_model: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
    audio_separator_use_soundfile: bool = AUDIO_SEPARATOR_USE_SOUNDFILE,
) -> Tuple[AudioSegment, str, bytes, str, Optional[AudioSegment], bool, str, Optional[str], Optional[str]]:
    """Returns: (processed_audio, input_mime, processed_bytes, gemini_mime, backing_audio, merge_backing, backing_source, vocals_path, backing_path)"""
    source_audio_temp_path: Optional[str] = None
    custom_backing_audio: Optional[AudioSegment] = None
    backing_track_source = "none"
    effective_apply_enhancement = apply_enhancement or apply_super_resolution
    custom_backing_ref = (custom_backing_audio_reference or "").strip()
    custom_backing_requested = bool(custom_backing_audio_file) or bool(custom_backing_ref)
    if custom_backing_requested:
        if emit_status:
            await emit_status(stage="backing", message="Loading custom backing track...")
        custom_audio_io = await load_audio_bytes_from_request(custom_backing_audio_file, custom_backing_audio_reference)
        if custom_audio_io is None:
            raise TranslateWorkflowHttpError(
                400,
                {"status": "error", "message": "Custom backing track not provided."},
            )
        custom_audio_bytes = custom_audio_io.read()
        if not custom_audio_bytes:
            raise TranslateWorkflowHttpError(
                400,
                {"status": "error", "message": "Custom backing track data is empty."},
            )
        custom_mime_type = (
            custom_backing_mime_type_value
            or (custom_backing_audio_file.content_type if custom_backing_audio_file else None)
            or "audio/wav"
        )
        custom_audio_format = _guess_audio_format_from_mime(custom_mime_type)
        try:
            custom_backing_audio = await _decode_audio_segment(custom_audio_bytes, custom_audio_format)
        except Exception as exc:
            raise TranslateWorkflowHttpError(
                400,
                {"status": "error", "message": f"Failed to decode custom backing track: {str(exc)}"},
            ) from exc
        backing_track_source = "custom"
        if emit_status:
            await emit_status(stage="backing", message="Custom backing track loaded.")

    if reuse_source_session is not None:
        try:
            original_audio = _get_session_original_audio(reuse_source_session)
        except RuntimeError:
            raise TranslateWorkflowHttpError(
                404,
                {
                    "status": "error",
                    "message": "Reusable session is missing separated vocals. Please re-upload the audio.",
                },
            )
        input_mime_type = reuse_source_session.input_mime_type or audio_mime_type_value or "audio/wav"
        processed_audio_bytes: Optional[bytes] = None
        reuse_audio_path = getattr(reuse_source_session, "original_audio_path", None)
        if (
            reuse_audio_path
            and reuse_audio_path.lower().endswith(".mp3")
            and os.path.exists(reuse_audio_path)
        ):
            processed_audio_bytes = _read_file_bytes(reuse_audio_path)
            if processed_audio_bytes is None:
                print(
                    f"⚠️ Falling back to re-export for Gemini input; failed to read cached chunk '{reuse_audio_path}'."
                )
        if emit_status:
            await emit_status(
                stage="decode",
                message=f"Reusing audio from session {reuse_source_session.session_id}.",
            )
        if processed_audio_bytes is None:
            processed_audio_bytes = await _export_audio_segment_bytes(
                original_audio,
                fmt="mp3",
                bitrate=GEMINI_AUDIO_EXPORT_BITRATE,
            )
        backing_track_audio = custom_backing_audio or _get_session_backing_audio(reuse_source_session)
        if custom_backing_audio is not None:
            backing_track_source = "custom"
        elif backing_track_audio is not None:
            backing_track_source = reuse_source_session.backing_track_source or "reuse"
        else:
            backing_track_source = "none"
        merge_with_backing = requested_merge_backing and backing_track_audio is not None
        if requested_merge_backing and backing_track_audio is None:
            print("⚠️ Unable to merge with backing track because no instrumental was derived.")
        # Return None for cached paths when reusing session (paths already stored in session)
        return (
            original_audio,
            input_mime_type,
            processed_audio_bytes,
            "audio/mpeg",
            backing_track_audio,
            merge_with_backing,
            backing_track_source,
            getattr(reuse_source_session, 'original_audio_path', None),  # Reuse existing path
            getattr(reuse_source_session, 'backing_track_path', None),  # Reuse existing path
        )

    try:
        if preloaded_audio_bytes is not None:
            audio_bytes = preloaded_audio_bytes
        else:
            audio_io = await load_audio_bytes_from_request(audio_file, audio_reference)
            if audio_io is None:
                raise TranslateWorkflowHttpError(
                    400,
                    {"status": "error", "message": "No audio provided for translation."},
                )
            audio_bytes = audio_io.read()
        if not audio_bytes:
            raise TranslateWorkflowHttpError(
                400,
                {"status": "error", "message": "Provided audio data is empty."},
            )
        original_filename = source_audio_filename or (audio_file.filename if audio_file else None)
        input_mime_type = audio_mime_type_value or (audio_file.content_type if audio_file else None) or "audio/wav"
        normalized_audio_temp_path: Optional[str] = None
        upload_temp_path: Optional[str] = None
        audio_separator_input_path: Optional[str] = None  # Original file for audio-separator
        
        # Compute original audio hash BEFORE any normalization (for consistent caching)
        original_audio_hash = _compute_bytes_md5(audio_bytes)
        
        # Persist uploaded audio for processing
        if effective_apply_enhancement or audio_separator_enabled:
            upload_temp_path = _persist_audio_upload(audio_bytes, original_filename)
            source_audio_temp_path = upload_temp_path
        
        # Audio-separator uses the original file directly; FFmpeg handles containers
        # such as M4A when use_soundfile is disabled.
        if audio_separator_enabled and upload_temp_path:
            audio_separator_input_path = upload_temp_path
        
        # Normalize to WAV only for ClearVoice (when audio-separator not used)
        # When audio-separator is enabled, ClearVoice will process the separated vocals directly
        if effective_apply_enhancement and not audio_separator_enabled:
            normalized_audio_temp_path = await _normalize_uploaded_audio(upload_temp_path)
            source_audio_temp_path = normalized_audio_temp_path
        else:
            source_audio_temp_path = upload_temp_path

        audio_format = _guess_audio_format_from_mime(input_mime_type)

        if emit_status:
            await emit_status(stage="decode", message="Decoding audio input...")
        try:
            if normalized_audio_temp_path:
                original_audio = await _load_audio_segment_from_path(normalized_audio_temp_path)
            else:
                original_audio = await _decode_audio_segment(audio_bytes, audio_format)
        except Exception as exc:
            raise TranslateWorkflowHttpError(
                400,
                {"status": "error", "message": f"Failed to decode audio: {str(exc)}"},
            ) from exc
        if emit_status:
            await emit_status(
                stage="decode",
                message=f"Decoded audio ({len(original_audio) / 1000:.1f}s).",
            )

        # Audio-Separator: Separate vocals and instrumentals before ClearVoice
        audio_separator_vocals: Optional[AudioSegment] = None
        audio_separator_instrumental: Optional[AudioSegment] = None
        audio_separator_vocals_path: Optional[str] = None
        audio_separator_instrumental_path: Optional[str] = None
        backing_track_audio: Optional[AudioSegment] = None  # Will be set by audio-separator or clearvoice
        audio_separator_from_cache = False
        
        if audio_separator_enabled and AudioSeparator is not None:
            # Use original file for audio-separator (no WAV normalization needed)
            separator_input = audio_separator_input_path
            if not separator_input:
                # Fallback: export audio segment to temp file if no path available
                separator_input = await _export_audio_segment_to_tempfile(original_audio)
            try:
                # Use original audio hash for caching (consistent across sessions)
                # Skip loading AudioSegments if enhancement is disabled - we can use cached files directly
                skip_loading = not effective_apply_enhancement
                separator_result = await _run_audio_separator_pipeline(
                    separator_input,
                    model_key=audio_separator_model,
                    emit_status=emit_status,
                    cache_hash=original_audio_hash,
                    skip_audio_loading=skip_loading,
                    use_soundfile=audio_separator_use_soundfile,
                )
                audio_separator_vocals_path = separator_result.vocals_path
                audio_separator_instrumental_path = separator_result.instrumental_path
                audio_separator_from_cache = separator_result.from_cache
                
                if not skip_loading:
                    # Use separated vocals as input to ClearVoice
                    audio_separator_vocals = separator_result.vocals_audio
                    audio_separator_instrumental = separator_result.instrumental_audio
                    original_audio = audio_separator_vocals
                    backing_track_audio = audio_separator_instrumental
                    backing_track_source = "audio_separator"
                    if emit_status:
                        await emit_status(
                            stage="audio_separation",
                            message=f"Vocals ({len(audio_separator_vocals) / 1000:.1f}s) and instrumentals ({len(audio_separator_instrumental) / 1000:.1f}s) separated.",
                        )
                else:
                    # When skipping loading, we'll load later or use cached paths directly
                    backing_track_source = "audio_separator"
                    if emit_status:
                        await emit_status(
                            stage="audio_separation",
                            message=f"Audio separation complete (using cached files).",
                        )
            except Exception as exc:
                print(f"⚠️ Audio-separator failed, falling back to ClearVoice extraction: {exc}")
                if emit_status:
                    await emit_status(
                        stage="audio_separation",
                        message=f"⚠️ Audio separation failed: {exc}. Falling back to ClearVoice extraction.",
                    )
        elif audio_separator_enabled and AudioSeparator is None:
            if emit_status:
                await emit_status(
                    stage="audio_separation",
                    message="⚠️ audio-separator not installed. Skipping vocal/instrumental separation.",
                )

        # Optimization: If audio-separator produced cached MP3 and no enhancement needed,
        # read the cached MP3 directly for Gemini instead of re-encoding
        can_use_cached_mp3 = (
            audio_separator_vocals_path is not None
            and audio_separator_vocals_path.lower().endswith('.mp3')
            and not effective_apply_enhancement
            and os.path.exists(audio_separator_vocals_path)
        )
        
        processed_audio_bytes: Optional[bytes] = None
        processed_audio: Optional[AudioSegment] = None
        
        if can_use_cached_mp3:
            # Read cached MP3 directly - much faster than decode + re-encode!
            print(f"[audio] Using cached vocals MP3 directly for Gemini: {audio_separator_vocals_path}")
            read_start = time.perf_counter()
            cached_bytes = await _run_blocking(lambda: _read_file_bytes(audio_separator_vocals_path))
            read_elapsed = (time.perf_counter() - read_start) * 1000
            
            if cached_bytes is None:
                print(f"⚠️ Failed to read cached MP3, falling back to standard processing")
                can_use_cached_mp3 = False
            else:
                processed_audio_bytes = cached_bytes
                print(f"[audio] Cached MP3 read ({len(processed_audio_bytes) / 1024:.1f} KB) in {read_elapsed:.1f}ms")
                
                # OPTIMIZATION: Load audio in parallel, but also track paths for session storage
                # Session can copy files instead of re-encoding if paths match
                if emit_status:
                    await emit_status(
                        stage="decode",
                        message="Loading separated audio tracks in parallel...",
                    )
                print(f"[audio] Loading vocals and instrumentals AudioSegments in parallel...")
                load_start = time.perf_counter()
                
                # Load both in parallel using asyncio.gather
                load_tasks = [_load_audio_segment_from_path(audio_separator_vocals_path)]
                if audio_separator_instrumental_path and os.path.exists(audio_separator_instrumental_path):
                    load_tasks.append(_load_audio_segment_from_path(audio_separator_instrumental_path))
                
                load_results = await asyncio.gather(*load_tasks)
                processed_audio = load_results[0]
                if len(load_results) > 1:
                    backing_track_audio = load_results[1]
                
                load_elapsed = (time.perf_counter() - load_start) * 1000
                print(f"[audio] Parallel load complete: vocals ({len(processed_audio) / 1000:.1f}s), instrumentals ({len(backing_track_audio) / 1000:.1f}s if loaded) in {load_elapsed:.1f}ms")
        
        if not can_use_cached_mp3:
            # ClearVoice: Enhancement and super-resolution (applied to vocals from audio-separator or original)
            # Only extract backing track via ClearVoice if audio-separator was not used
            if audio_separator_vocals is not None:
                original_audio = audio_separator_vocals
            
            pre_clearvoice_mix_audio = original_audio if (effective_apply_enhancement and not audio_separator_enabled) else None
            clearvoice_input_audio = original_audio
            clearvoice_source_path = (
                audio_separator_vocals_path
                if (
                    effective_apply_enhancement
                    and audio_separator_vocals_path
                    and os.path.exists(audio_separator_vocals_path)
                )
                else None
            )
            processed_audio, clearvoice_backing = await _run_clearvoice_pipeline(
                clearvoice_input_audio,
                apply_enhancement=effective_apply_enhancement,
                apply_super_resolution=apply_super_resolution,
                pre_clearvoice_mix_audio=pre_clearvoice_mix_audio,
                emit_status=emit_status,
                source_audio_path=clearvoice_source_path,
                clearvoice_parallel_config=clearvoice_parallel_config,
                enhancement_model_name=enhancement_model_name,
            )
            # Use ClearVoice backing track only if audio-separator was not used
            if backing_track_audio is None and clearvoice_backing is not None:
                backing_track_audio = clearvoice_backing
                backing_track_source = "extracted"
            
            print(f"[audio] Exporting processed audio ({len(processed_audio) / 1000:.1f}s) to MP3 for Gemini...")
            export_start = time.perf_counter()
            processed_audio_bytes = await _export_audio_segment_bytes(
                processed_audio,
                fmt="mp3",
                bitrate=GEMINI_AUDIO_EXPORT_BITRATE,
            )
            export_elapsed = (time.perf_counter() - export_start) * 1000
            print(f"[audio] Export complete ({len(processed_audio_bytes) / 1024:.1f} KB) in {export_elapsed:.1f}ms")
        
        if custom_backing_audio is not None:
            backing_track_audio = custom_backing_audio
            backing_track_source = "custom"
        merge_with_backing = requested_merge_backing and backing_track_audio is not None
        if requested_merge_backing and backing_track_audio is None:
            print("⚠️ Unable to merge with backing track because no instrumental was derived.")

        if emit_status:
            effective_model = enhancement_model_name if enhancement_model_name in AVAILABLE_ENHANCEMENT_MODELS else DEFAULT_ENHANCEMENT_MODEL
            if apply_super_resolution:
                await emit_status(
                    stage="gemini_prep",
                    message="📤 Sending super-resolved audio (MossFormer2_SR_48K) to Gemini for transcription/translation.",
                )
            elif apply_enhancement:
                await emit_status(
                    stage="gemini_prep",
                    message=f"📤 Sending enhanced audio ({effective_model}) to Gemini for transcription/translation.",
                )
            else:
                await emit_status(
                    stage="gemini_prep",
                    message="📤 Sending original audio to Gemini for transcription/translation.",
                )

        # Return tuple now includes cached paths for fast session storage
        session_vocals_path = None if effective_apply_enhancement else audio_separator_vocals_path
        return (
            processed_audio,
            input_mime_type,
            processed_audio_bytes,
            "audio/mpeg",
            backing_track_audio,
            merge_with_backing,
            backing_track_source,
            session_vocals_path,  # Cached vocals path (for fast session storage)
            audio_separator_instrumental_path,  # Cached backing path (for fast session storage)
        )
    finally:
        if source_audio_temp_path:
            try:
                os.remove(source_audio_temp_path)
            except Exception:
                pass


def _clearvoice_input_manifest_dir() -> str:
    path = os.path.join(CLEARVOICE_CACHE_DIR, "inputs")
    os.makedirs(path, exist_ok=True)
    return path


def _clearvoice_input_manifest_path(source_hash: str) -> str:
    return os.path.join(_clearvoice_input_manifest_dir(), f"{source_hash}.json")


def _load_clearvoice_input_manifest(source_hash: Optional[str]) -> Optional[Dict[str, Any]]:
    if not source_hash:
        return None
    path = _clearvoice_input_manifest_path(source_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as infile:
            return json.load(infile)
    except Exception as exc:
        print(f"⚠️ Failed to read ClearVoice input manifest '{path}': {exc}")
        return None


def _write_clearvoice_input_manifest(
    source_hash: Optional[str],
    clearvoice_hash: str,
    *,
    processed_hash: Optional[str] = None,
) -> None:
    if not source_hash:
        return
    payload = {
        "source_hash": source_hash,
        "clearvoice_hash": clearvoice_hash,
        "updated_at": time.time(),
    }
    if processed_hash:
        payload["processed_hash"] = processed_hash
    path = _clearvoice_input_manifest_path(source_hash)
    try:
        with open(path, "w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, ensure_ascii=False)
    except Exception as exc:
        print(f"⚠️ Failed to write ClearVoice input manifest '{path}': {exc}")


def _clearvoice_cache_paths(cache_hash: str, enhancement_model: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Get cache paths for ClearVoice outputs.
    Enhancement model is included in the cache filename to prevent different models from hitting the same cache.
    """
    cache_dir = os.path.join(CLEARVOICE_CACHE_DIR, cache_hash)
    # Include enhancement model in cache filename to distinguish between different models
    model_suffix = enhancement_model.lower().replace("_", "-") if enhancement_model else "default"
    cached_enhanced_path = os.path.join(cache_dir, f"enhanced_{model_suffix}.mp3")
    cached_sr_path = os.path.join(cache_dir, "mossformer2_sr.mp3")  # SR is always MossFormer2_SR_48K
    cached_backing_path = os.path.join(cache_dir, f"backing_{model_suffix}.mp3")
    return cache_dir, cached_enhanced_path, cached_sr_path, cached_backing_path


async def _prepare_clearvoice_for_split(
    *,
    audio_file: Optional[UploadFile],
    audio_reference: Optional[str],
    preloaded_audio_bytes: Optional[bytes],
    source_audio_filename: Optional[str],
    apply_enhancement: bool,
    apply_super_resolution: bool,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    clearvoice_parallel_config: Optional[ClearVoiceParallelConfig] = None,
    enhancement_model_name: Optional[str] = None,
    audio_separator_enabled: bool = False,
    audio_separator_model: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
    audio_separator_use_soundfile: bool = AUDIO_SEPARATOR_USE_SOUNDFILE,
) -> SplitClearVoiceAssets:
    """
    Prepare audio for chunk splitting, optionally with Audio-Separator and/or ClearVoice enhancement.
    Audio-Separator separates vocals from instrumentals (recommended for mixed audio).
    ClearVoice enhancement is optional - recommended for noisy audio but not needed for clean vocal-only uploads.
    """
    cleanup_paths: List[str] = []
    upload_temp_path: Optional[str] = None
    normalized_audio_temp_path: Optional[str] = None
    wav_input_path: Optional[str] = None
    cache_hash: Optional[str] = None
    cache_dir: Optional[str] = None
    cached_enhanced_path: Optional[str] = None
    cached_sr_path: Optional[str] = None
    cached_backing_path: Optional[str] = None
    source_hash: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None
    processed_hash: Optional[str] = None

    def _update_cache_paths(hash_value: str) -> None:
        nonlocal cache_hash, cache_dir, cached_enhanced_path, cached_sr_path, cached_backing_path
        cache_dir, cached_enhanced_path, cached_sr_path, cached_backing_path = _clearvoice_cache_paths(
            hash_value, enhancement_model_name
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache_hash = hash_value

    async def _ensure_wav_input_path(audio_bytes_value: bytes) -> str:
        nonlocal upload_temp_path, normalized_audio_temp_path, wav_input_path
        if wav_input_path:
            return wav_input_path
        upload_temp_path = _persist_audio_upload(audio_bytes_value, source_audio_filename)
        cleanup_paths.append(upload_temp_path)
        normalized_audio_temp_path = await _normalize_uploaded_audio(upload_temp_path)
        cleanup_paths.append(normalized_audio_temp_path)
        wav_input_local, created_temp = await _ensure_clearvoice_input_path(normalized_audio_temp_path)
        if created_temp:
            cleanup_paths.append(wav_input_local)
        wav_input_path = wav_input_local
        return wav_input_path
    try:
        if preloaded_audio_bytes is not None:
            audio_bytes = preloaded_audio_bytes
        else:
            audio_io = await load_audio_bytes_from_request(audio_file, audio_reference)
            if audio_io is None:
                raise TranslateWorkflowHttpError(
                    400,
                    {"status": "error", "message": "No audio provided for translation."},
                )
            audio_bytes = audio_io.read()
        if not audio_bytes:
            raise TranslateWorkflowHttpError(
                400,
                {"status": "error", "message": "Provided audio data is empty."},
            )
        source_hash = _compute_bytes_md5(audio_bytes)
        
        # Audio-Separator: Run first to separate vocals from instrumentals (if enabled)
        audio_separator_vocals_path: Optional[str] = None
        audio_separator_instrumental_path: Optional[str] = None
        audio_for_processing: Optional[bytes] = audio_bytes
        
        if audio_separator_enabled and AudioSeparator is not None:
            if emit_status:
                await emit_status(
                    stage="audio_separation",
                    message="Separating vocals from instrumentals with Audio-Separator...",
            )
            upload_temp_path = _persist_audio_upload(audio_bytes, source_audio_filename)
            cleanup_paths.append(upload_temp_path)
            separator_input_path = upload_temp_path
            try:
                # Skip loading full AudioSegments when no enhancement/super-resolution is requested
                # to avoid re-decoding multi-hour files when cached stems are sufficient.
                skip_separator_loading = not (apply_enhancement or apply_super_resolution)
                separator_result = await _run_audio_separator_pipeline(
                    separator_input_path,
                    model_key=audio_separator_model,
                    emit_status=emit_status,
                    cache_hash=source_hash,
                    skip_audio_loading=skip_separator_loading,
                    use_soundfile=audio_separator_use_soundfile,
                )
                audio_separator_from_cache = separator_result.from_cache
                audio_separator_vocals_path = separator_result.vocals_path
                audio_separator_instrumental_path = separator_result.instrumental_path

                if not skip_separator_loading:
                    audio_separator_vocals = separator_result.vocals_audio
                    audio_separator_instrumental = separator_result.instrumental_audio
                    # Export vocals to temp file for further processing
                    vocals_temp_path = await _export_audio_segment_to_tempfile(audio_separator_vocals)
                    cleanup_paths.append(vocals_temp_path)
                    audio_separator_vocals_path = vocals_temp_path
                    # Use vocals for subsequent processing
                    audio_for_processing = await _run_blocking(lambda: audio_separator_vocals.export(format="wav").read())
                else:
                    # We'll rely on cached file paths instead of re-decoding large AudioSegments
                    audio_for_processing = None
                # Export instrumental to backing cache
                _update_cache_paths(source_hash)
                instrumental_cache_path = os.path.join(cache_dir, f"audio_sep_{audio_separator_model}_instrumental.mp3")
                if not os.path.exists(instrumental_cache_path):
                    # Fast path: copy the cached file instead of re-encoding when available
                    src_instrumental_path = audio_separator_instrumental_path
                    if src_instrumental_path and os.path.exists(src_instrumental_path):
                        shutil.copy2(src_instrumental_path, instrumental_cache_path)
                    elif audio_separator_instrumental is not None:
                        audio_separator_instrumental.export(instrumental_cache_path, format="mp3", bitrate="192k")
                    print(f"💾 Cached audio-separator instrumental for {source_hash}.")
                audio_separator_instrumental_path = instrumental_cache_path
                if emit_status:
                    await emit_status(
                        stage="audio_separation",
                        message=(
                            f"Vocals and instrumentals separated."
                            if skip_separator_loading
                            else f"Vocals ({len(audio_separator_vocals) / 1000:.1f}s) and instrumentals ({len(audio_separator_instrumental) / 1000:.1f}s) separated."
                        ),
                    )
            except Exception as exc:
                print(f"⚠️ Audio-separator failed: {exc}")
                if emit_status:
                    await emit_status(
                        stage="audio_separation",
                        message=f"⚠️ Audio separation failed: {exc}. Processing original audio.",
                    )
        elif audio_separator_enabled and AudioSeparator is None:
            if emit_status:
                await emit_status(
                    stage="audio_separation",
                    message="⚠️ audio-separator not installed. Processing original audio.",
                )
        
        # Use separated vocals hash if available, otherwise use source hash
        if audio_separator_vocals_path:
            processing_source_hash = _compute_file_md5(audio_separator_vocals_path)
        elif audio_for_processing is not None:
            processing_source_hash = _compute_bytes_md5(audio_for_processing)
        else:
            processing_source_hash = source_hash
        
        manifest = _load_clearvoice_input_manifest(processing_source_hash)
        if manifest:
            manifest_hash = manifest.get("clearvoice_hash")
            processed_hash = manifest.get("processed_hash")
            if manifest_hash:
                _update_cache_paths(manifest_hash)
                if os.path.exists(cached_enhanced_path) or os.path.exists(cached_sr_path):
                    print(
                        f"♻️ ClearVoice: Reusing cached assets for upload hash {processing_source_hash[:8]} → {cache_hash}."
                    )
                else:
                    cache_hash = None
                    manifest = None
        if cache_hash is None:
            if audio_separator_vocals_path:
                _update_cache_paths(_compute_file_md5(audio_separator_vocals_path))
            else:
                wav_input_path = await _ensure_wav_input_path(audio_for_processing)
                _update_cache_paths(_compute_file_md5(wav_input_path))

        # Determine the final target based on enhancement/super-resolution settings
        needs_clearvoice = apply_enhancement or apply_super_resolution
        if needs_clearvoice:
            final_processed_target = cached_sr_path if apply_super_resolution else cached_enhanced_path
            need_processing = not os.path.exists(final_processed_target)
            if apply_super_resolution and not os.path.exists(cached_enhanced_path):
                need_processing = True
            # Only extract backing via ClearVoice if audio-separator was not used
            backing_needed = not os.path.exists(cached_backing_path) and not audio_separator_instrumental_path

            if need_processing or backing_needed:
                if emit_status:
                    await emit_status(
                        stage="enhancement",
                        message="Generating ClearVoice cache for splitting...",
                    )
                # Use separated vocals if available, otherwise use original audio
                source_path = audio_separator_vocals_path or normalized_audio_temp_path
                if source_path is None:
                    await _ensure_wav_input_path(audio_for_processing)
                    source_path = normalized_audio_temp_path or wav_input_path
                if source_path is None:
                    raise TranslateWorkflowHttpError(
                        500,
                        {"status": "error", "message": "Unable to prepare source audio for ClearVoice."},
                    )
                original_audio = await _load_audio_segment_from_path(source_path)
                # Only extract backing track via ClearVoice if audio-separator was not used
                pre_clearvoice_mix = original_audio if not audio_separator_instrumental_path else None
                processed_audio, backing_track_audio = await _run_clearvoice_pipeline(
                    original_audio,
                    apply_enhancement=apply_enhancement,
                    apply_super_resolution=apply_super_resolution,
                    pre_clearvoice_mix_audio=pre_clearvoice_mix,
                    emit_status=emit_status,
                    source_audio_path=source_path,
                    clearvoice_parallel_config=clearvoice_parallel_config,
                    enhancement_model_name=enhancement_model_name,
                )
                processed_audio = None
                backing_track_audio = None

            if not os.path.exists(final_processed_target):
                raise TranslateWorkflowHttpError(
                    500,
                    {
                        "status": "error",
                        "message": "ClearVoice processing did not produce an enhanced track.",
                    },
                )
        else:
            # No ClearVoice enhancement requested
            # Use separated vocals if available, otherwise use normalized source audio
            if emit_status:
                status_msg = "Preparing audio for splitting"
                if audio_separator_vocals_path:
                    status_msg += " (using separated vocals, no ClearVoice enhancement)"
                else:
                    status_msg += " (no preprocessing)"
                await emit_status(stage="preparation", message=status_msg + "...")
            source_path = audio_separator_vocals_path or normalized_audio_temp_path
            if source_path is None:
                await _ensure_wav_input_path(audio_for_processing)
                source_path = normalized_audio_temp_path or wav_input_path
            if source_path is None:
                raise TranslateWorkflowHttpError(
                    500,
                    {"status": "error", "message": "Unable to prepare source audio."},
                )
            # Cache the source as the final target for consistency
            final_processed_target = cached_enhanced_path
            if not os.path.exists(final_processed_target):
                await _normalize_audio_to_cached(source_path, final_processed_target, delete_source=False)
                print(f"💾 Cached audio for splitting {cache_hash}.")

        try:
            processed_info = _probe_audio_metadata_from_path(
                final_processed_target,
                compute_hash=processed_hash is None,
            )
        except Exception as exc:
            raise TranslateWorkflowHttpError(
                500,
                {"status": "error", "message": f"Failed to inspect ClearVoice output: {str(exc)}"},
            ) from exc

        backing_info: Optional[AudioAssetInfo] = None
        backing_source = "none"
        actual_backing_path: Optional[str] = None
        # Prefer audio-separator instrumental over ClearVoice extracted backing
        if audio_separator_instrumental_path and os.path.exists(audio_separator_instrumental_path):
            actual_backing_path = audio_separator_instrumental_path
            backing_source = "audio_separator"
        elif os.path.exists(cached_backing_path):
            actual_backing_path = cached_backing_path
            backing_source = "cache"
        if actual_backing_path:
            try:
                backing_info = _probe_audio_metadata_from_path(actual_backing_path, compute_hash=False)
            except Exception as exc:
                print(f"⚠️ Failed to probe backing track: {exc}")

        if processed_hash and not processed_info.hash_md5:
            processed_info = AudioAssetInfo(
                duration_ms=processed_info.duration_ms,
                sample_rate=processed_info.sample_rate,
                channels=processed_info.channels,
                sample_width=processed_info.sample_width,
                frame_count=processed_info.frame_count,
                hash_md5=processed_hash,
            )
        clearvoice_hash = processed_info.hash_md5 or _compute_file_md5(final_processed_target)
        _write_clearvoice_input_manifest(
            source_hash,
            cache_hash or "",
            processed_hash=clearvoice_hash,
        )
        return SplitClearVoiceAssets(
            processed_audio_path=final_processed_target,
            processed_audio_info=processed_info,
            clearvoice_hash=clearvoice_hash,
            processed_mime_type="audio/mpeg",
            backing_track_path=actual_backing_path,
            backing_audio_info=backing_info,
            backing_track_source=backing_source,
            applied_super_resolution=apply_super_resolution,
        )
    finally:
        for path in cleanup_paths:
            _safe_remove_file(path)


@dataclass
class SegmentBuildResult:
    session: TranslateSessionData
    segments: List[Dict[str, Any]]
    ui_segments: List[Dict[str, Any]]
    gemini_chunks: List[Dict[str, Any]]
    speaker_profiles: List[Dict[str, Any]]
    gemini_raw_text: Optional[str]
    metadata: Dict[str, Any]
    manual_segments_used: bool


async def _build_translation_segments(
    *,
    original_audio: AudioSegment,
    processed_audio_bytes: bytes,
    gemini_mime_type: str,
    dest_language: str,
    final_prompt: str,
    translate_enabled: bool,
    response_format_value: str,
    bitrate_value: str,
    input_mime_type: str,
    apply_enhancement: bool,
    apply_super_resolution: bool,
    ignore_non_speech_flag: bool,
    preserve_silence_audio_flag: bool,
    generated_volume_percent_value: float,
    backing_volume_percent_value: float,
    silence_volume_percent_value: float,
    backing_track_audio: Optional[AudioSegment],
    backing_track_source: str,
    merge_with_backing: bool,
    segments_override_value: Optional[str],
    min_speech_duration: int,
    max_merge_interval: int,
    resolved_gemini_model: str,
    gemini_api_key_value: Optional[str],
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    source_chunk_session: Optional[TranslateSessionData] = None,
    source_audio_filename: Optional[str] = None,
    source_base_name: Optional[str] = None,
    source_video_path: Optional[str] = None,
    source_video_filename: Optional[str] = None,
    force_gemini_regenerate: bool = False,
    initial_speaker_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    default_speaker_preset: Optional[str] = None,
    default_emotion_weight: Optional[float] = None,
    clearvoice_settings: Optional[Dict[str, Any]] = None,
    # Cached source paths for fast session storage (avoids re-encoding)
    original_audio_source_path: Optional[str] = None,
    backing_track_source_path: Optional[str] = None,
    transcription_pipeline: str = "qwen_omnivad",
    whisperx_proxy_refiner: bool = False,
    translation_llm_model: Optional[str] = None,
    qwen_omnivad_enable_diarization: bool = True,
    qwen_omnivad_diarization_backend: str = "auto",
    qwen_omnivad_diarization_min_seconds: float = 0.0,
    qwen_omnivad_enable_forced_aligner: bool = True,
    qwen_omnivad_merge_gap_seconds: float = DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS,
) -> SegmentBuildResult:
    manual_chunk_data = None
    manual_speaker_profiles: List[Dict[str, Any]] = []
    manual_segments_used = False
    raw_gemini_response_text: Optional[str] = None
    manual_raw_input = (segments_override_value or "").strip() if segments_override_value else None
    if segments_override_value:
        try:
            parsed_override = _parse_manual_segments_input(segments_override_value)
            if parsed_override is not None:
                manual_chunk_data, manual_speaker_profiles = parsed_override
                manual_segments_used = True
                raw_gemini_response_text = manual_raw_input
        except ValueError as exc:
            raise TranslateWorkflowHttpError(
                400,
                {"status": "error", "message": str(exc)},
            ) from exc

    transcription_pipeline = _normalize_transcription_pipeline(transcription_pipeline)
    whisperx_proxy_refiner = (
        _coerce_to_bool(whisperx_proxy_refiner)
        and transcription_pipeline == "whisperx"
    )
    resolved_translation_llm_model = _normalize_translation_llm_model(
        translation_llm_model
    )

    # Resolve pipeline label
    use_whisperx = transcription_pipeline == "whisperx"
    use_qwen_omnivad = transcription_pipeline == "qwen_omnivad"
    use_parakeet = transcription_pipeline == "parakeet"
    (
        qwen_omnivad_enable_diarization,
        qwen_omnivad_diarization_min_seconds,
    ) = _resolve_qwen_omnivad_diarization_options(
        transcription_pipeline,
        qwen_omnivad_enable_diarization,
        qwen_omnivad_diarization_min_seconds,
    )
    qwen_omnivad_diarization_backend = _resolve_qwen_omnivad_diarization_backend_option(
        transcription_pipeline,
        qwen_omnivad_diarization_backend,
    )
    qwen_omnivad_enable_forced_aligner = _resolve_qwen_omnivad_forced_aligner_option(
        transcription_pipeline,
        qwen_omnivad_enable_forced_aligner,
    )
    qwen_omnivad_merge_gap_seconds = _resolve_qwen_omnivad_merge_gap_seconds_option(
        transcription_pipeline,
        qwen_omnivad_merge_gap_seconds,
    )
    if use_whisperx:
        pipeline_label = "WhisperX"
    elif use_qwen_omnivad:
        pipeline_label = "Qwen3-ASR + OmniVAD"
    elif use_parakeet:
        pipeline_label = "NVIDIA Parakeet"
    else:
        pipeline_label = f"Gemini model '{resolved_gemini_model}'"

    if emit_status:
        await emit_status(
            stage="transcription",
            message=f"Analyzing audio with {pipeline_label}...",
        )

    speaker_profiles: List[Dict[str, Any]] = []
    gemini_cache_info: Dict[str, Any] = {}
    if manual_chunk_data is not None:
        gemini_chunks = manual_chunk_data
        speaker_profiles = manual_speaker_profiles or []
        print(f"[translate] Using {len(gemini_chunks)} manual segments (SRT/JSON), skipping inference")
        if emit_status:
            await emit_status(
                stage="manual_segments",
                message=f"Using {len(gemini_chunks)} manual segments from SRT/JSON input...",
            )
    elif use_whisperx:
        # --- WhisperX local pipeline ---
        if _run_whisperx_pipeline_sync is None:
            raise TranslateWorkflowHttpError(
                500,
                {"status": "error", "message": "WhisperX pipeline is not installed. Install whisperx, pyannote.audio, and litai."},
            )
        print(f"[translate] Using WhisperX local pipeline for transcription/translation")
        loop = asyncio.get_event_loop()
        (
            gemini_chunks,
            speaker_profiles,
            raw_gemini_response_text,
            gemini_cache_info,
        ) = await loop.run_in_executor(
            executor,
            functools.partial(
                _run_whisperx_pipeline_sync,
                processed_audio_bytes,
                dest_language=dest_language,
                enable_translation=translate_enabled,
                translation_llm_model=resolved_translation_llm_model,
                force_refresh=force_gemini_regenerate,
                enable_proxy_refiner=whisperx_proxy_refiner,
            ),
        )
    elif use_qwen_omnivad:
        # --- Qwen3-ASR + OmniVAD pipeline ---
        if _run_qwen_omnivad_pipeline_sync is None:
            raise TranslateWorkflowHttpError(
                500,
                {"status": "error", "message": "Qwen3-ASR/OmniVAD pipeline is not installed. Install qwen-asr, omnivad, and litai."},
            )
        print(f"[translate] Using Qwen3-ASR + OmniVAD pipeline for transcription/translation")
        loop = asyncio.get_event_loop()
        (
            gemini_chunks,
            speaker_profiles,
            raw_gemini_response_text,
            gemini_cache_info,
        ) = await loop.run_in_executor(
            executor,
            functools.partial(
                _run_qwen_omnivad_pipeline_sync,
                processed_audio_bytes,
                input_mime_type=gemini_mime_type,
                dest_language=dest_language,
                enable_translation=translate_enabled,
                translation_llm_model=resolved_translation_llm_model,
                force_refresh=force_gemini_regenerate,
                enable_diarization=qwen_omnivad_enable_diarization,
                diarization_backend=qwen_omnivad_diarization_backend,
                diarization_min_seconds=qwen_omnivad_diarization_min_seconds,
                enable_forced_aligner=qwen_omnivad_enable_forced_aligner,
                merge_gap_seconds=qwen_omnivad_merge_gap_seconds,
            ),
        )
    elif use_parakeet:
        # --- NVIDIA Parakeet NeMo pipeline ---
        if _run_parakeet_pipeline_sync is None:
            raise TranslateWorkflowHttpError(
                500,
                {"status": "error", "message": "NVIDIA Parakeet pipeline is not installed. Install nemo_toolkit[asr]."},
            )
        print(f"[translate] Using NVIDIA Parakeet pipeline for transcription/translation")
        loop = asyncio.get_event_loop()
        (
            gemini_chunks,
            speaker_profiles,
            raw_gemini_response_text,
            gemini_cache_info,
        ) = await loop.run_in_executor(
            executor,
            functools.partial(
                _run_parakeet_pipeline_sync,
                processed_audio_bytes,
                input_mime_type=gemini_mime_type,
                dest_language=dest_language,
                enable_translation=translate_enabled,
                translation_llm_model=resolved_translation_llm_model,
                force_refresh=force_gemini_regenerate,
            ),
        )
    else:
        # --- Gemini cloud pipeline (default) ---
        (
            gemini_chunks,
            speaker_profiles,
            raw_gemini_response_text,
            gemini_cache_info,
        ) = await _gemini_transcribe_translate(
            processed_audio_bytes,
            gemini_mime_type,
            dest_language,
            final_prompt,
            model_name=resolved_gemini_model,
            api_key_override=gemini_api_key_value,
            force_refresh=force_gemini_regenerate,
        )
    if raw_gemini_response_text is None and manual_raw_input:
        raw_gemini_response_text = manual_raw_input

    if emit_status:
        await emit_status(
            stage="segmentation",
            message=f"Processing {len(gemini_chunks)} segments; preparing timeline...",
        )

    original_duration_ms = len(original_audio)
    print(f"[translate] Preparing {len(gemini_chunks)} segments for timeline processing...")
    segment_prep_start = time.perf_counter()
    segments = _prepare_translation_segments(
        original_duration_ms,
        gemini_chunks,
        dest_language,
        speaker_profiles=speaker_profiles,
        min_speech_duration_ms=min_speech_duration,
        max_merge_interval_ms=max_merge_interval,
    )
    segment_prep_elapsed = (time.perf_counter() - segment_prep_start) * 1000
    print(f"[translate] Segment preparation complete: {len(segments)} segments in {segment_prep_elapsed:.1f}ms")

    detected_speaker_ids: Set[str] = set()
    for segment in segments:
        speaker_label = str(segment.get("speaker") or "").strip().lower()
        if speaker_label:
            detected_speaker_ids.add(speaker_label)

    normalized_overrides = _normalize_speaker_overrides(
        initial_speaker_overrides,
        speaker_profiles,
    )
    normalized_default_speaker = (default_speaker_preset or "").strip()
    normalized_default_emotion = _coerce_emotion_weight(
        default_emotion_weight,
        DEFAULT_EMOTION_WEIGHT,
    )
    if normalized_default_speaker:
        if not detected_speaker_ids and speaker_profiles:
            for profile in speaker_profiles:
                profile_id = str(profile.get("id") or "").strip().lower()
                if profile_id:
                    detected_speaker_ids.add(profile_id)
        for speaker_id in detected_speaker_ids:
            if not speaker_id or speaker_id in normalized_overrides:
                continue
            normalized_overrides[speaker_id] = {
                "preset_name": normalized_default_speaker,
                "use_emotion_prompt": True,
                "emotion_weight": normalized_default_emotion,
            }

    if not translate_enabled:
        for segment in segments:
            if segment.get("type") != "speech":
                continue
            translated_text = (segment.get("translated_text") or "").strip()
            source_text = (segment.get("source_text") or "").strip()
            if not translated_text and source_text:
                segment["translated_text"] = source_text

    if emit_status:
        await emit_status(
            stage="session",
            message=f"Creating translation session with {len(segments)} segments...",
        )
    print(f"[translate] Creating session with {len(segments)} segments...")
    session_create_start = time.perf_counter()
    session = await _create_translate_session(
        original_audio,
        dest_language,
        final_prompt,
        translate_enabled,
        response_format_value,
        bitrate_value,
        input_mime_type,
        clearvoice_settings
        or {
            "enhancement": apply_enhancement,
            "super_resolution": apply_super_resolution,
        },
        segments,
        gemini_chunks,
        resolved_gemini_model,
        gemini_api_key_value,
        translation_llm_model=resolved_translation_llm_model,
        transcription_pipeline=transcription_pipeline,
        whisperx_proxy_refiner=whisperx_proxy_refiner,
        qwen_omnivad_enable_diarization=qwen_omnivad_enable_diarization,
        qwen_omnivad_diarization_backend=qwen_omnivad_diarization_backend,
        qwen_omnivad_diarization_min_seconds=qwen_omnivad_diarization_min_seconds,
        qwen_omnivad_enable_forced_aligner=qwen_omnivad_enable_forced_aligner,
        qwen_omnivad_merge_gap_seconds=qwen_omnivad_merge_gap_seconds,
        backing_track_audio=backing_track_audio,
        merge_with_backing=merge_with_backing,
        ignore_non_speech=ignore_non_speech_flag,
        preserve_silence_audio=preserve_silence_audio_flag,
        generated_volume_percent=generated_volume_percent_value,
        backing_volume_percent=backing_volume_percent_value,
        silence_volume_percent=silence_volume_percent_value,
        speaker_profiles=speaker_profiles,
        speaker_overrides=normalized_overrides,
        gemini_raw_text=raw_gemini_response_text,
        backing_track_source=backing_track_source,
        source_audio_filename=source_audio_filename,
        source_base_name=source_base_name,
        source_video_path=source_video_path,
        source_video_filename=source_video_filename,
        default_speaker_preset=normalized_default_speaker,
        default_emotion_weight=normalized_default_emotion,
        # Pass cached paths for fast session storage
        original_audio_path=original_audio_source_path,
        backing_track_path=backing_track_source_path,
    )
    session_create_elapsed = (time.perf_counter() - session_create_start) * 1000
    print(f"[translate] Session created: {session.session_id} in {session_create_elapsed:.1f}ms")
    
    if source_chunk_session and source_chunk_session.chunk_parent_id:
        session.chunk_parent_id = source_chunk_session.chunk_parent_id
        session.chunk_index = source_chunk_session.chunk_index
        session.chunk_start_ms = source_chunk_session.chunk_start_ms
        session.chunk_end_ms = source_chunk_session.chunk_end_ms
        session.chunk_cut_reason = source_chunk_session.chunk_cut_reason
        session.chunk_silence_midpoint_ms = source_chunk_session.chunk_silence_midpoint_ms
        session.chunk_source_session_id = source_chunk_session.session_id
    
    if emit_status:
        await emit_status(
            stage="serialization",
            message=f"Serializing {len(segments)} segments for UI...",
        )
    print(f"[translate] Serializing {len(segments)} segments for UI...")
    serialize_start = time.perf_counter()
    ui_segments = _serialize_segments_for_ui(segments, session.session_id)
    serialize_elapsed = (time.perf_counter() - serialize_start) * 1000
    print(f"[translate] Serialization complete in {serialize_elapsed:.1f}ms")

    metadata = {
        "dest_language": dest_language,
        "segment_count": len(segments),
        "speech_segment_count": sum(1 for seg in segments if seg.get("type") == "speech"),
        "silence_segment_count": sum(1 for seg in segments if seg.get("type") == "silence"),
        "audio_duration_ms": original_duration_ms,
        "translate_enabled": translate_enabled,
        "response_format": response_format_value,
        "bitrate": bitrate_value,
        "prompt": final_prompt,
        "gemini_model": resolved_gemini_model,
        "translation_llm_model": resolved_translation_llm_model,
        "transcription_pipeline": transcription_pipeline,
        "transcription_pipeline_label": pipeline_label if transcription_pipeline != "gemini" else "Gemini",
        "whisperx_proxy_refiner": whisperx_proxy_refiner,
        "qwen_omnivad_enable_diarization": qwen_omnivad_enable_diarization,
        "qwen_omnivad_diarization_backend": qwen_omnivad_diarization_backend,
        "qwen_omnivad_diarization_min_seconds": qwen_omnivad_diarization_min_seconds,
        "qwen_omnivad_enable_forced_aligner": qwen_omnivad_enable_forced_aligner,
        "qwen_omnivad_merge_gap_seconds": qwen_omnivad_merge_gap_seconds,
        "ignore_non_speech": ignore_non_speech_flag,
        "preserve_silence_audio": preserve_silence_audio_flag,
        "generated_volume_percent": generated_volume_percent_value,
        "backing_volume_percent": backing_volume_percent_value,
        "silence_volume_percent": silence_volume_percent_value,
        "gemini_raw_segments": gemini_chunks,
        "speaker_profiles": copy.deepcopy(speaker_profiles),
        "speaker_overrides": copy.deepcopy(getattr(session, "speaker_overrides", {})),
        "gemini_raw_text": raw_gemini_response_text,
        "clearvoice": clearvoice_settings
        or {
            "enhancement": apply_enhancement,
            "super_resolution": apply_super_resolution,
        },
        "default_speaker_preset": normalized_default_speaker,
        "default_emotion_weight": normalized_default_emotion,
        "backing_track": {
            "available": backing_track_audio is not None,
            "requested": merge_with_backing or False,
            "merge_with_backing": merge_with_backing,
            "preview_url": f"/api/translate_backing_track/{session.session_id}"
            if backing_track_audio is not None
            else None,
            "volume_percent": backing_volume_percent_value,
            "source": backing_track_source,
        },
        "segment_rules": {
            "min_speech_ms": min_speech_duration,
            "max_merge_ms": max_merge_interval,
        },
        "manual_segments": manual_segments_used,
        "force_gemini_regenerate": force_gemini_regenerate,
    }
    _apply_source_video_metadata(metadata, session)
    if gemini_cache_info:
        metadata["gemini_cache"] = gemini_cache_info
    chunk_reference = source_chunk_session if source_chunk_session else (session if session.chunk_parent_id else None)
    if chunk_reference:
        metadata["chunk"] = {
            "session_id": chunk_reference.session_id,
            "chunk_index": chunk_reference.chunk_index,
            "start_ms": chunk_reference.chunk_start_ms,
            "end_ms": chunk_reference.chunk_end_ms,
            "cut_reason": chunk_reference.chunk_cut_reason,
            "batch_id": chunk_reference.chunk_parent_id,
            "silence_midpoint_ms": chunk_reference.chunk_silence_midpoint_ms,
            "start_label": _format_ms_to_timestamp(chunk_reference.chunk_start_ms or 0),
            "end_label": _format_ms_to_timestamp(chunk_reference.chunk_end_ms or 0),
            "duration_label": _format_ms_to_timestamp(
                max(0, (chunk_reference.chunk_end_ms or 0) - (chunk_reference.chunk_start_ms or 0))
            ),
            "backing_available": _session_has_backing_audio(chunk_reference),
            "backing_source": chunk_reference.backing_track_source or "none",
            "audio_url": _chunk_output_url(chunk_reference),
            "output_format": chunk_reference.chunk_output_format,
            "output_filename": chunk_reference.chunk_output_filename,
            "vocals_url": f"/api/translate_vocals/{chunk_reference.session_id}",
            "backing_url": (
                f"/api/translate_backing_track/{chunk_reference.session_id}"
                if _session_has_backing_audio(chunk_reference)
                else None
            ),
        }
    metadata["session_id"] = session.session_id
    metadata["reuse_session_id"] = session.session_id
    separation_info = {
        "vocals_available": True,
        "vocals_url": f"/api/translate_vocals/{session.session_id}",
        "backing_available": backing_track_audio is not None,
        "backing_url": metadata["backing_track"]["preview_url"] if backing_track_audio is not None else None,
        "backing_source": backing_track_source,
        "session_id": session.session_id,
    }
    metadata["separation"] = separation_info

    return SegmentBuildResult(
        session=session,
        segments=segments,
        ui_segments=ui_segments,
        gemini_chunks=gemini_chunks,
        speaker_profiles=speaker_profiles,
        gemini_raw_text=raw_gemini_response_text,
        metadata=metadata,
        manual_segments_used=manual_segments_used,
    )


@dataclass
class TranslateSessionData:
    session_id: str
    dest_language: str
    prompt: str
    translate_enabled: bool
    response_format: str
    bitrate: str
    input_mime_type: Optional[str]
    clearvoice_settings: Dict[str, Any]
    base_segments: List[Dict[str, Any]]
    gemini_chunks: List[Dict[str, Any]]
    gemini_model: str
    gemini_api_key: Optional[str]
    translation_llm_model: str = DEFAULT_TRANSLATION_LLM_MODEL
    transcription_pipeline: str = "qwen_omnivad"
    whisperx_proxy_refiner: bool = False
    qwen_omnivad_enable_diarization: bool = True
    qwen_omnivad_diarization_backend: str = "auto"
    qwen_omnivad_diarization_min_seconds: float = 0.0
    qwen_omnivad_enable_forced_aligner: bool = True
    qwen_omnivad_merge_gap_seconds: float = DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS
    source_audio_filename: Optional[str] = None
    source_base_name: Optional[str] = None
    source_video_path: Optional[str] = None
    source_video_filename: Optional[str] = None
    original_audio_path: Optional[str] = None
    original_audio_info: Optional[AudioAssetInfo] = None
    backing_track_path: Optional[str] = None
    backing_audio_info: Optional[AudioAssetInfo] = None
    backing_track_source: str = "none"
    merge_with_backing: bool = False
    ignore_non_speech: bool = False
    preserve_silence_audio: bool = False
    generated_volume_percent: float = DEFAULT_GENERATED_VOLUME_PERCENT
    backing_volume_percent: float = DEFAULT_GENERATED_VOLUME_PERCENT
    silence_volume_percent: float = DEFAULT_SILENCE_VOLUME_PERCENT
    speaker_profiles: List[Dict[str, Any]] = field(default_factory=list)
    speaker_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_speaker_preset: Optional[str] = None
    default_emotion_weight: float = DEFAULT_EMOTION_WEIGHT
    gemini_raw_text: Optional[str] = None
    created_at: float = field(default_factory=lambda: time.time())
    chunk_parent_id: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_start_ms: Optional[int] = None
    chunk_end_ms: Optional[int] = None
    chunk_generated: bool = False
    chunk_output_path: Optional[str] = None
    chunk_output_filename: Optional[str] = None
    chunk_output_format: Optional[str] = None
    chunk_generated_at: Optional[float] = None
    chunk_cut_reason: Optional[str] = None
    chunk_silence_midpoint_ms: Optional[int] = None
    chunk_source_session_id: Optional[str] = None
    original_audio: Optional[AudioSegment] = field(default=None, repr=False)
    backing_track_audio: Optional[AudioSegment] = field(default=None, repr=False)


ADVANCED_TRANSLATE_SESSIONS: Dict[str, TranslateSessionData] = {}
ADVANCED_TRANSLATE_SESSION_LOCK = asyncio.Lock()
ADVANCED_TRANSLATE_SESSION_TTL_SECONDS = 60 * 60  # 1 hour


def _session_media_path(session_id: str, kind: str, fmt: str = "mp3") -> str:
    safe_kind = "".join(c for c in kind if c.isalnum() or c in {"_", "-"}).strip() or "media"
    return os.path.join(TRANSLATE_SESSION_MEDIA_DIR, f"{session_id}_{safe_kind}.{fmt}")


def _chunk_batch_media_path(batch_id: str, kind: str, fmt: str = "mp3") -> str:
    safe_kind = "".join(c for c in kind if c.isalnum() or c in {"_", "-"}).strip() or kind
    return os.path.join(TRANSLATE_SESSION_MEDIA_DIR, f"{batch_id}_{safe_kind}.{fmt}")


def _compute_file_md5(path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as infile:
        for chunk in iter(lambda: infile.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _compute_bytes_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _read_file_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as infile:
            return infile.read()
    except Exception as exc:
        print(f"⚠️ Failed to read binary file '{path}': {exc}")
        return None


def _gemini_cache_key(
    audio_bytes: bytes,
    *,
    dest_language: str,
    prompt_text: str,
    model_name: str,
) -> Tuple[str, str]:
    audio_hash = _compute_bytes_md5(audio_bytes)
    normalized_prompt = (prompt_text or "").strip()
    meta_blob = "|".join(
        [
            model_name.strip().lower(),
            dest_language.strip().lower(),
            normalized_prompt,
        ]
    )
    meta_hash = hashlib.md5(meta_blob.encode("utf-8")).hexdigest()
    cache_key = f"{audio_hash}_{meta_hash}"
    return audio_hash, cache_key


def _gemini_cache_path(cache_key: str) -> str:
    safe_key = "".join(c for c in cache_key if c.isalnum() or c in {"_", "-"}).strip()
    return os.path.join(GEMINI_CACHE_DIR, f"{safe_key}.json")


def _load_gemini_cache_entry(cache_key: str) -> Optional[Dict[str, Any]]:
    path = _gemini_cache_path(cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        if data.get("version") != GEMINI_CACHE_VERSION:
            return None
        return data
    except Exception as exc:
        print(f"⚠️ Failed to read Gemini cache '{path}': {exc}")
        return None


def _write_gemini_cache_entry(cache_key: str, record: Dict[str, Any]) -> Optional[str]:
    os.makedirs(GEMINI_CACHE_DIR, exist_ok=True)
    path = _gemini_cache_path(cache_key)
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="gemini_cache_", suffix=".json", dir=GEMINI_CACHE_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(record, tmp_file, ensure_ascii=False)
        os.replace(tmp_path, path)
        return path
    except Exception as exc:
        print(f"⚠️ Failed to write Gemini cache '{path}': {exc}")
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return None


def _split_audio_cache_key(
    clearvoice_hash: str,
    *,
    min_chunk_minutes: float,
    max_chunk_minutes: float,
    min_silence_ms: int,
    silence_threshold_db: float,
) -> str:
    """
    Compute a cache key for split audio operations based on clearvoice hash + split settings.
    Same audio (after clearvoice) + same settings = same split result.
    """
    settings_str = f"min={min_chunk_minutes}|max={max_chunk_minutes}|silence_ms={min_silence_ms}|threshold={silence_threshold_db}"
    settings_hash = hashlib.md5(settings_str.encode("utf-8")).hexdigest()[:12]
    return f"{clearvoice_hash}_{settings_hash}"


def _split_audio_cache_path(cache_key: str, chunk_idx: int) -> str:
    """Get path to cached split MP3 file for a specific chunk."""
    safe_key = "".join(c for c in cache_key if c.isalnum() or c in {"_", "-"}).strip()
    return os.path.join(SPLIT_AUDIO_CACHE_DIR, f"{safe_key}_chunk{chunk_idx:03d}.mp3")


def _split_audio_metadata_path(cache_key: str) -> str:
    """Get path to split audio metadata JSON file."""
    safe_key = "".join(c for c in cache_key if c.isalnum() or c in {"_", "-"}).strip()
    return os.path.join(SPLIT_AUDIO_CACHE_DIR, f"{safe_key}_metadata.json")


def _load_split_audio_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Load cached split audio metadata including chunk ranges and file paths.
    Returns None if cache doesn't exist or is invalid.
    """
    metadata_path = _split_audio_metadata_path(cache_key)
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as infile:
            metadata = json.load(infile)
        
        # Verify version
        if metadata.get("version") != 1:
            print(f"⚠️ Split audio cache version mismatch: {metadata_path}")
            return None
        
        # Verify all chunk files exist
        chunk_count = metadata.get("chunk_count", 0)
        chunk_files = metadata.get("chunk_files", [])
        for chunk_idx in range(1, chunk_count + 1):
            chunk_path = None
            if chunk_files and chunk_idx - 1 < len(chunk_files):
                chunk_path = chunk_files[chunk_idx - 1].get("file_path")
            if not chunk_path:
                chunk_path = _split_audio_cache_path(cache_key, chunk_idx)
            if not os.path.exists(chunk_path):
                print(f"⚠️ Split audio cache incomplete, missing chunk {chunk_idx}: {chunk_path}")
                return None
        
        return metadata
    except Exception as exc:
        print(f"⚠️ Failed to load split audio cache '{metadata_path}': {exc}")
        return None


def _iter_split_metadata_files() -> List[str]:
    if not os.path.exists(SPLIT_AUDIO_CACHE_DIR):
        return []
    paths: List[str] = []
    for entry in os.scandir(SPLIT_AUDIO_CACHE_DIR):
        if entry.is_file() and entry.name.endswith("_metadata.json"):
            paths.append(entry.path)
    return paths


def _load_split_cache_metadata_for_batch(batch_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not batch_id:
        return None
    for path in _iter_split_metadata_files():
        try:
            with open(path, "r", encoding="utf-8") as infile:
                data = json.load(infile)
        except Exception:
            continue
        if data.get("chunk_batch_id") == batch_id:
            return path, data
    return None


def _write_split_cache_metadata(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="split_audio_",
        suffix=".json",
        dir=os.path.dirname(path) or SPLIT_AUDIO_CACHE_DIR,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _refresh_split_cache_for_session(session: TranslateSessionData) -> None:
    batch_id = session.chunk_parent_id
    if not batch_id:
        return
    lookup = _load_split_cache_metadata_for_batch(batch_id)
    if not lookup:
        return
    metadata_path, metadata = lookup
    manifests: List[Dict[str, Any]] = metadata.get("chunk_sessions", [])
    updated_manifest = _session_to_manifest(session)
    updated = False
    for idx, manifest in enumerate(manifests):
        if manifest.get("session_id") == session.session_id or manifest.get("chunk_index") == session.chunk_index:
            manifests[idx] = updated_manifest
            updated = True
            break
    if not updated:
        manifests.append(updated_manifest)
    metadata["chunk_sessions"] = manifests
    _write_split_cache_metadata(metadata_path, metadata)


def _save_split_audio_cache(
    cache_key: str,
    chunk_ranges: List[Dict[str, Any]],
    chunk_audio_segments: Optional[List[AudioSegment]],
    silence_stats: Dict[str, Any],
    *,
    chunk_audio_paths: Optional[List[Optional[str]]] = None,
    chunk_session_manifests: Optional[List[Dict[str, Any]]] = None,
    chunk_batch_id: Optional[str] = None,
    base_output_name: Optional[str] = None,
) -> None:
    """
    Save split audio chunks as MP3 files and metadata to cache.
    
    Args:
        cache_key: Cache key identifying this split configuration
        chunk_ranges: List of chunk range dictionaries with start_ms, end_ms, etc.
        chunk_audio_segments: List of AudioSegment objects for each chunk
        silence_stats: Statistics about silence detection
    """
    os.makedirs(SPLIT_AUDIO_CACHE_DIR, exist_ok=True)
    
    try:
        chunk_files: List[Dict[str, Any]] = []
        if chunk_audio_segments is not None:
            for chunk_idx, (chunk_range, audio_segment) in enumerate(zip(chunk_ranges, chunk_audio_segments), start=1):
                desired_path = None
                if chunk_audio_paths and chunk_idx - 1 < len(chunk_audio_paths):
                    desired_path = chunk_audio_paths[chunk_idx - 1]
                chunk_path = desired_path or _split_audio_cache_path(cache_key, chunk_idx)
                os.makedirs(os.path.dirname(chunk_path) or ".", exist_ok=True)
                audio_segment.export(chunk_path, format="mp3", bitrate="128k")
                chunk_files.append(
                    {
                        "chunk_index": chunk_idx,
                        "file_path": chunk_path,
                        "start_ms": chunk_range.get("start_ms"),
                        "end_ms": chunk_range.get("end_ms"),
                        "duration_ms": len(audio_segment),
                        "cut_reason": chunk_range.get("cut_reason"),
                        "silence_midpoint_ms": chunk_range.get("silence_midpoint_ms"),
                    }
                )
        else:
            for chunk_idx, chunk_range in enumerate(chunk_ranges, start=1):
                range_duration = max(0, int(chunk_range.get("end_ms", 0)) - int(chunk_range.get("start_ms", 0)))
                desired_path = None
                if chunk_audio_paths and chunk_idx - 1 < len(chunk_audio_paths):
                    desired_path = chunk_audio_paths[chunk_idx - 1]
                chunk_path = desired_path or _split_audio_cache_path(cache_key, chunk_idx)
                if not os.path.exists(chunk_path):
                    raise FileNotFoundError(f"Chunk audio missing for cache at {chunk_path}")
                chunk_files.append(
                    {
                        "chunk_index": chunk_idx,
                        "file_path": chunk_path,
                        "start_ms": chunk_range.get("start_ms"),
                        "end_ms": chunk_range.get("end_ms"),
                        "duration_ms": range_duration,
                        "cut_reason": chunk_range.get("cut_reason"),
                        "silence_midpoint_ms": chunk_range.get("silence_midpoint_ms"),
                    }
                )
        
        # Save metadata
        metadata = {
            "version": 1,
            "created_at": time.time(),
            "cache_key": cache_key,
            "chunk_count": len(chunk_ranges),
            "chunk_ranges": chunk_ranges,
            "chunk_files": chunk_files,
            "silence_stats": silence_stats,
            "chunk_sessions": chunk_session_manifests or [],
            "chunk_batch_id": chunk_batch_id,
            "base_output_name": base_output_name,
        }
        
        metadata_path = _split_audio_metadata_path(cache_key)
        fd, tmp_path = tempfile.mkstemp(
            prefix="split_audio_",
            suffix=".json",
            dir=SPLIT_AUDIO_CACHE_DIR
        )
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(metadata, tmp_file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, metadata_path)
        
        print(f"💾 Split audio cache stored: {len(chunk_ranges)} chunks → {os.path.basename(metadata_path)}")
    except Exception as exc:
        print(f"⚠️ Failed to save split audio cache: {exc}")
        # Clean up partial cache
        try:
            metadata_path = _split_audio_metadata_path(cache_key)
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
            for chunk_idx in range(1, len(chunk_ranges) + 1):
                chunk_path = _split_audio_cache_path(cache_key, chunk_idx)
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
        except Exception:
            pass


async def _materialize_split_chunks_ffmpeg(
    source_audio_path: str,
    chunk_specs: List[Dict[str, Any]],
    *,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    bitrate: str = TRANSLATE_DEFAULT_BITRATE,
) -> None:
    if not chunk_specs:
        return
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg split requested but ffmpeg is unavailable.")
    semaphore = asyncio.Semaphore(max(1, CHUNK_SPLIT_FFMPEG_CONCURRENCY))
    progress_lock = asyncio.Lock()
    total = len(chunk_specs)
    completed = 0

    async def _extract(spec: Dict[str, Any]) -> None:
        nonlocal completed
        chunk_path = spec["chunk_mp3_path"]
        start_ms = int(spec["start_ms"])
        end_ms = int(spec["end_ms"])
        os.makedirs(os.path.dirname(chunk_path) or ".", exist_ok=True)
        async with semaphore:
            await _run_blocking(
                _ffmpeg_extract_segment,
                source_audio_path,
                chunk_path,
                start_ms,
                end_ms,
                bitrate=bitrate,
            )
        async with progress_lock:
            completed += 1
            if emit_status:
                await emit_status(
                    stage="chunking",
                    message=f"Extracted chunk {completed}/{total} via ffmpeg...",
                )

    await asyncio.gather(*(asyncio.create_task(_extract(spec)) for spec in chunk_specs))


async def _materialize_split_chunks_pydub(
    source_audio: AudioSegment,
    chunk_specs: List[Dict[str, Any]],
    *,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    bitrate: str = TRANSLATE_DEFAULT_BITRATE,
) -> None:
    total = len(chunk_specs)
    for idx, spec in enumerate(chunk_specs, start=1):
        start_ms = int(spec["start_ms"])
        end_ms = int(spec["end_ms"])
        chunk_audio = source_audio[start_ms:end_ms]
        await _export_audio_segment_to_path(
            chunk_audio,
            spec["chunk_mp3_path"],
            fmt="mp3",
            bitrate=bitrate,
        )
        if emit_status:
            await emit_status(stage="chunking", message=f"Exported chunk {idx}/{total}...")


def _transcode_file_with_ffmpeg(
    source_path: str,
    target_path: str,
    target_fmt: str = "wav",
) -> bool:
    """Transcode audio file directly using FFmpeg (no memory loading!)."""
    if not _ffmpeg_available():
        return False
    try:
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        cmd = ["ffmpeg", "-y", "-i", source_path]
        
        if target_fmt == "wav":
            cmd.extend(["-codec:a", "pcm_s16le"])
        elif target_fmt == "mp3":
            cmd.extend(["-codec:a", "libmp3lame", "-b:a", "192k"])
        elif target_fmt == "aac":
            cmd.extend(["-codec:a", "aac", "-b:a", "192k"])

        cmd.extend(_ffmpeg_thread_args())
        cmd.append(target_path)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and os.path.exists(target_path)
    except Exception as exc:
        print(f"⚠️ FFmpeg transcode failed: {exc}")
        return False


def _copy_or_transcode_file(
    source_path: str,
    target_path: str,
    target_fmt: str,
) -> bool:
    """Copy file if same format, or transcode if different. Returns success."""
    source_ext = os.path.splitext(source_path)[1].lstrip(".").lower()
    
    # Same format - just copy!
    if source_ext == target_fmt.lower():
        try:
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
            shutil.copy2(source_path, target_path)
            print(f"[session] Copied {os.path.basename(source_path)} → {os.path.basename(target_path)}")
            return True
        except Exception as exc:
            print(f"⚠️ File copy failed: {exc}")
            return False
    
    # Different format - transcode with FFmpeg
    return _transcode_file_with_ffmpeg(source_path, target_path, target_fmt)


def _persist_session_audio_segment_sync(
    session_id: str,
    audio: AudioSegment,
    kind: str,
    fmt: str = "mp3",
) -> Tuple[str, AudioAssetInfo]:
    """Synchronous version of session audio persistence."""
    path = _session_media_path(session_id, kind, fmt)
    duration_sec = len(audio) / 1000.0
    
    # Use FFmpeg for long audio (>5 min) - much faster
    use_ffmpeg = duration_sec > 300 and _ffmpeg_available()
    method = "FFmpeg" if use_ffmpeg else "pydub"
    print(f"[session] Persisting {kind} audio ({duration_sec:.1f}s) to {path} using {method}...")
    persist_start = time.perf_counter()
    
    if use_ffmpeg:
        _export_audio_to_path_via_ffmpeg_sync(audio, path, fmt)
    else:
        audio.export(path, format=fmt)
    
    persist_elapsed = (time.perf_counter() - persist_start) * 1000
    print(f"[session] Audio persist complete in {persist_elapsed:.1f}ms")
    return path, _audio_segment_metadata(audio)


def _persist_session_audio_from_path_sync(
    session_id: str,
    source_path: str,
    kind: str,
    fmt: str = "mp3",
) -> Tuple[str, AudioAssetInfo]:
    """Persist session audio by copying/transcoding from source file (no memory loading!)."""
    target_path = _session_media_path(session_id, kind, fmt)
    
    print(f"[session] Persisting {kind} audio from {os.path.basename(source_path)} to {os.path.basename(target_path)}...")
    persist_start = time.perf_counter()
    
    if _copy_or_transcode_file(source_path, target_path, fmt):
        persist_elapsed = (time.perf_counter() - persist_start) * 1000
        print(f"[session] Audio persist (from path) complete in {persist_elapsed:.1f}ms")
        # Get metadata without loading full file
        info = _probe_audio_metadata_from_path(target_path, compute_hash=False)
        return target_path, info
    else:
        # Fallback: load and export
        print(f"[session] Fallback to load+export for {kind}")
        audio = _load_audio_segment_from_path_sync(source_path)
        return _persist_session_audio_segment_sync(session_id, audio, kind, fmt)


async def _persist_session_audio_from_path(
    session_id: str,
    source_path: str,
    kind: str,
    fmt: str = "mp3",
) -> Tuple[str, AudioAssetInfo]:
    """Async version - persist from source file path."""
    return await _run_blocking(_persist_session_audio_from_path_sync, session_id, source_path, kind, fmt)


async def _persist_session_audio_segment(
    session_id: str,
    audio: AudioSegment,
    kind: str,
    fmt: str = "mp3",
) -> Tuple[str, AudioAssetInfo]:
    """Async version - runs persistence in thread pool to avoid blocking event loop."""
    return await _run_blocking(_persist_session_audio_segment_sync, session_id, audio, kind, fmt)


def _get_session_original_audio(session: TranslateSessionData) -> AudioSegment:
    if session.original_audio is not None:
        return session.original_audio
    if session.original_audio_path and os.path.exists(session.original_audio_path):
        session.original_audio = _load_audio_segment_from_path_sync(session.original_audio_path)
        return session.original_audio
    raise RuntimeError(f"Session {session.session_id} does not have a stored vocal track.")


def _get_session_backing_audio(session: TranslateSessionData) -> Optional[AudioSegment]:
    if session.backing_track_audio is not None:
        return session.backing_track_audio
    path = getattr(session, "backing_track_path", None)
    if path and os.path.exists(path):
        session.backing_track_audio = _load_audio_segment_from_path_sync(path)
        return session.backing_track_audio
    return None


def _session_has_backing_audio(session: TranslateSessionData) -> bool:
    if session.backing_track_audio is not None:
        return True
    path = getattr(session, "backing_track_path", None)
    return bool(path and os.path.exists(path))


def _persist_chunk_batch_media(
    batch_id: str,
    audio: Union[AudioSegment, str],
    kind: str,
    fmt: str = "mp3",
) -> str:
    path = _chunk_batch_media_path(batch_id, kind, fmt)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(audio, AudioSegment):
        audio.export(path, format=fmt)
    else:
        _transcode_or_copy_audio_file(audio, path, fmt)
    return path


def _transcode_or_copy_audio_file(source_path: str, target_path: str, fmt: str) -> None:
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Audio source not found: {source_path}")
    source_ext = os.path.splitext(source_path)[1].lstrip(".").lower()
    target_ext = (fmt or source_ext or "mp3").lower()
    if source_ext == target_ext:
        shutil.copy2(source_path, target_path)
        return
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is required to transcode audio for chunk batch media.")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source_path,
    ]
    codec_args = _ffmpeg_codec_args_for_format(target_ext, TRANSLATE_DEFAULT_BITRATE)
    if target_ext == "wav":
        codec_args = ["-c:a", "pcm_s16le"]
    cmd += codec_args
    cmd.append(target_path)
    _run_ffmpeg_command(cmd)


def _session_audio_duration_ms(session: TranslateSessionData) -> int:
    info = getattr(session, "original_audio_info", None)
    if info and info.duration_ms:
        return int(info.duration_ms)
    audio = _get_session_original_audio(session)
    return len(audio)


def _resolve_chunk_duration_ms(session: TranslateSessionData) -> int:
    if session.chunk_start_ms is not None and session.chunk_end_ms is not None:
        duration = int(session.chunk_end_ms) - int(session.chunk_start_ms)
        if duration > 0:
            return duration
    try:
        return _session_audio_duration_ms(session)
    except RuntimeError:
        return 0


class TranslateWorkflowHttpError(Exception):
    def __init__(self, status_code: int, content: Dict[str, Any]):
        super().__init__(content.get("message") or "Translate workflow error")
        self.status_code = status_code
        self.content = content


@functools.lru_cache(maxsize=4)
def _create_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError(
            "The google-genai package is required for translation. Install it with `pip install google-genai`."
        )
    return genai.Client(api_key=api_key)


def _get_gemini_client(api_key: str):
    return _create_gemini_client(api_key)


def estimate_speech_duration(text: str, language: str = "auto") -> int:
    """
    Estimate speech duration in milliseconds based on text length.
    
    Args:
        text: Input text
        language: Language hint ("zh", "en", or "auto")
    
    Returns:
        Estimated duration in milliseconds
    """
    if not text or not text.strip():
        return 0
    
    # Clean text
    text = text.strip()
    
    # Detect language if auto
    if language == "auto":
        # Simple heuristic: if more than 30% Chinese characters, treat as Chinese
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars / max(len(text), 1) > 0.3:
            language = "zh"
        else:
            language = "en"
    
    # Speech rate estimates (characters per second)
    # These are conservative estimates based on typical TTS output
    if language == "zh":
        # Chinese: ~4-5 characters per second
        chars_per_second = 4.5
        char_count = len([c for c in text if '\u4e00' <= c <= '\u9fff' or c.isalnum()])
    else:
        # English: ~12-15 characters per second (including spaces)
        # Or ~150-180 words per minute = 2.5-3 words per second
        chars_per_second = 13.0
        char_count = len(text)
    
    # Calculate base duration
    duration_seconds = char_count / chars_per_second
    
    # Add padding for punctuation pauses (10% extra time)
    punctuation_count = sum(1 for c in text if c in ',.!?;:。，！？；：')
    pause_time = punctuation_count * 0.3  # 300ms per punctuation
    
    total_duration_seconds = duration_seconds + pause_time
    
    # Add 10% buffer for natural speech variations
    total_duration_seconds *= 1.1
    
    # Convert to milliseconds and round up to nearest 100ms
    duration_ms = int(total_duration_seconds * 1000)
    duration_ms = ((duration_ms + 99) // 100) * 100  # Round up to nearest 100ms
    
    return duration_ms

parser = argparse.ArgumentParser(description="IndexTTS vLLM v2 FastAPI WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=8000, help="Port to run the web API on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web API on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
parser.add_argument("--use_torch_compile", action="store_true", default=False, help="use torch compile")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.25, help="GPU memory utilization")

# Parse args if run as script, otherwise use defaults
try:
    cmd_args = parser.parse_args()
except SystemExit:
    # If running in interactive mode, use defaults
    cmd_args = argparse.Namespace(
        verbose=False,
        port=8000,
        host="0.0.0.0",
        model_dir="checkpoints",
        is_fp16=False,
        use_torch_compile=False,
        gpu_memory_utilization=0.25
    )

SETTINGS = AppSettings.from_namespace(cmd_args)

# Create directories
APP_DIR = Path(__file__).resolve().parent
_OUTPUT_SUBDIRS = (
    "translate_results",
    "clearvoice_cache",
    "translate_session_media",
    "split_audio_cache",
    "downloaded_videos",
    "downloaded_video_audio",
)


def _resolve_outputs_root(app_dir: Path) -> Path:
    candidates = [
        app_dir / "outputs",
        app_dir.parent / "outputs",
    ]
    best_path = candidates[0]
    best_score = -1
    for candidate in candidates:
        score = sum((candidate / sub).exists() for sub in _OUTPUT_SUBDIRS)
        if score > best_score:
            best_score = score
            best_path = candidate
    best_path.mkdir(parents=True, exist_ok=True)
    return best_path.resolve()


ROOT_OUTPUT_DIR = _resolve_outputs_root(APP_DIR)
os.makedirs(ROOT_OUTPUT_DIR / "tasks", exist_ok=True)
os.makedirs((APP_DIR / "prompts").resolve(), exist_ok=True)
os.makedirs((APP_DIR / "speaker_presets").resolve(), exist_ok=True)
TRANSLATE_OUTPUT_DIR = str((ROOT_OUTPUT_DIR / "translate_results").resolve())
os.makedirs(TRANSLATE_OUTPUT_DIR, exist_ok=True)
TRANSLATE_SESSION_MEDIA_DIR = str((ROOT_OUTPUT_DIR / "translate_session_media").resolve())
os.makedirs(TRANSLATE_SESSION_MEDIA_DIR, exist_ok=True)
VIDEO_DOWNLOAD_DIR = str((ROOT_OUTPUT_DIR / "downloaded_videos").resolve())
os.makedirs(VIDEO_DOWNLOAD_DIR, exist_ok=True)
VIDEO_AUDIO_CACHE_DIR = str((ROOT_OUTPUT_DIR / "downloaded_video_audio").resolve())
os.makedirs(VIDEO_AUDIO_CACHE_DIR, exist_ok=True)
VIDEO_SNAPSHOT_CACHE_DIR = str((ROOT_OUTPUT_DIR / "video_snapshots").resolve())
os.makedirs(VIDEO_SNAPSHOT_CACHE_DIR, exist_ok=True)
COOKIES_DIR = str((ROOT_OUTPUT_DIR / "cookies").resolve())
os.makedirs(COOKIES_DIR, exist_ok=True)
CLEARVOICE_CACHE_DIR = str((ROOT_OUTPUT_DIR / "clearvoice_cache").resolve())
os.makedirs(CLEARVOICE_CACHE_DIR, exist_ok=True)
AUDIO_SEPARATOR_CACHE_DIR = str((ROOT_OUTPUT_DIR / "audio_separator_cache").resolve())
os.makedirs(AUDIO_SEPARATOR_CACHE_DIR, exist_ok=True)
AUDIO_SEPARATOR_MODEL_DIR = str((Path(current_dir) / "checkpoints").resolve())
os.makedirs(AUDIO_SEPARATOR_MODEL_DIR, exist_ok=True)
_APP_MODEL_DIR = Path(SETTINGS.model_dir)
if not _APP_MODEL_DIR.is_absolute():
    _APP_MODEL_DIR = APP_DIR / _APP_MODEL_DIR
STABLE_AUDIO3_CHECKPOINT_DIR = str((_APP_MODEL_DIR / "stable-audio-3").resolve())
stable_audio3_manager = StableAudio3Manager(STABLE_AUDIO3_CHECKPOINT_DIR)
GEMINI_CACHE_DIR = str((ROOT_OUTPUT_DIR / "gemini_cache").resolve())
os.makedirs(GEMINI_CACHE_DIR, exist_ok=True)
SPLIT_AUDIO_CACHE_DIR = str((ROOT_OUTPUT_DIR / "split_audio_cache").resolve())
os.makedirs(SPLIT_AUDIO_CACHE_DIR, exist_ok=True)


def _guess_media_type_from_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    mapping = {
        **AUDIO_MEDIA_TYPES,
        "srt": "application/x-subrip",
        "mp4": "video/mp4",
        "m4v": "video/mp4",
        "mov": "video/quicktime",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
    }
    return mapping.get(ext, "application/octet-stream")


def _clean_ytdl_error(exc: Exception) -> str:
    msg = str(exc)
    msg = re.sub(r'\x1b\[[0-9;]*m', '', msg)  # Remove ANSI codes
    msg = msg.replace('ERROR:', '').strip()
    return msg


def _is_ytdl_format_unavailable_error(exc: Exception) -> bool:
    return "requested format is not available" in _clean_ytdl_error(exc).lower()


def _status_error(message: str, status_code: int = 400, **extra: Any) -> JSONResponse:
    payload = {"status": "error", "message": message}
    payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def _success_error(message: str, status_code: int = 500, **extra: Any) -> JSONResponse:
    payload = {"success": False, "error": message}
    payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def _request_has_json_body(request: Request) -> bool:
    return "application/json" in request.headers.get("content-type", "").lower()


JSON_STREAM_MIN_CHUNK_BYTES = 4096


def _json_event_bytes(event: Dict[str, Any]) -> bytes:
    event_payload = event
    encoded = json.dumps(event_payload, ensure_ascii=False)
    if len(encoded.encode("utf-8")) < JSON_STREAM_MIN_CHUNK_BYTES:
        event_payload = dict(event)
        event_payload["_flush_padding"] = "0" * (
            JSON_STREAM_MIN_CHUNK_BYTES - len(encoded.encode("utf-8"))
        )
        encoded = json.dumps(event_payload, ensure_ascii=False)
    return (f"data: {encoded}\n\n").encode("utf-8")


JSON_STREAM_RESPONSE_HEADERS = {
    "Cache-Control": "no-store, no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "X-Content-Type-Options": "nosniff",
}


async def _json_stream_with_prelude(iterator: Any):
    yield _json_event_bytes(
        {
            "event": "stream_open",
            "timestamp": time.time(),
        }
    )
    async for chunk in iterator:
        yield chunk


def _json_stream_response(iterator: Any) -> StreamingResponse:
    return StreamingResponse(
        _json_stream_with_prelude(iterator),
        media_type="text/event-stream",
        headers=JSON_STREAM_RESPONSE_HEADERS,
    )


async def _read_optional_json_payload(
    request: Request,
    should_read: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[JSONResponse]]:
    if not should_read:
        return None, None
    try:
        payload = await request.json()
    except Exception as exc:
        return None, _status_error(f"Invalid JSON payload: {str(exc)}")
    if not isinstance(payload, dict):
        return None, _status_error("Invalid JSON payload: expected a JSON object.")
    return payload, None

# Async wrapper functions for blocking operations
async def async_write_file(file_path: str, data: bytes) -> None:
    """Async wrapper for writing file data"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _write_file_sync, file_path, data)

def _write_file_sync(file_path: str, data: bytes) -> None:
    """Synchronous file write operation"""
    with open(file_path, 'wb') as f:
        f.write(data)

async def async_read_file(file_path: str) -> bytes:
    """Async wrapper for reading file data"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _read_file_sync, file_path)

def _read_file_sync(file_path: str) -> bytes:
    """Synchronous file read operation"""
    with open(file_path, 'rb') as f:
        return f.read()


def _safe_remove_file(path: Optional[str]) -> None:
    """Best effort removal for temporary files."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


async def async_remove_file(file_path: str) -> None:
    """Async wrapper for removing files"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _remove_file_sync, file_path)

def _remove_file_sync(file_path: str) -> None:
    """Synchronous file removal operation"""
    try:
        os.unlink(file_path)
    except Exception:
        pass  # Ignore errors when removing temporary files


def _sanitize_preview_basename(speaker_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", speaker_name.strip())
    sanitized = sanitized.strip("_") or "speaker"
    digest = hashlib.md5(speaker_name.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}_{digest}"


def _get_speaker_preview_path(speaker_name: str) -> Path:
    return SPEAKER_PREVIEW_DIR / f"{_sanitize_preview_basename(speaker_name)}.mp3"


def _speaker_preview_url(speaker_name: str) -> str:
    return f"/api/speaker_preview/{quote(speaker_name, safe='')}"


def _speaker_preview_exists(speaker_name: str) -> bool:
    return _get_speaker_preview_path(speaker_name).exists()


async def _remove_speaker_preview(speaker_name: str) -> None:
    preview_path = _get_speaker_preview_path(speaker_name)
    if preview_path.exists():
        await async_remove_file(str(preview_path))


async def _create_speaker_preview_mp3(
    source_audio_path: str,
    speaker_name: str,
    *,
    bitrate: str = SPEAKER_PREVIEW_BITRATE,
) -> Optional[str]:
    """Create/overwrite a speaker preview MP3 derived from the processed reference audio."""
    loop = asyncio.get_event_loop()
    preview_path = _get_speaker_preview_path(speaker_name)

    def _export_preview() -> str:
        SPEAKER_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        audio = AudioSegment.from_file(source_audio_path)
        audio.export(preview_path, format="mp3", bitrate=bitrate)
        return str(preview_path)

    try:
        return await loop.run_in_executor(executor, _export_preview)
    except Exception as exc:
        print(f"⚠️ Failed to create preview MP3 for speaker '{speaker_name}': {exc}")
        return None


async def async_audio_read(file_path: str):
    """Async wrapper for soundfile.read()"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, sf.read, file_path)

async def async_cut_audio_to_duration(input_path: str, max_duration: float = 10.0):
    """Async wrapper for smart audio cutting at silence intervals"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _smart_cut_audio_at_silence, input_path, max_duration)

def _detect_silence_intervals(audio_data, sample_rate, min_silence_duration=0.3, silence_threshold=-40):
    """
    Detect silence intervals in audio data.
    
    Args:
        audio_data: Audio samples (mono or stereo)
        sample_rate: Sample rate in Hz
        min_silence_duration: Minimum duration of silence in seconds (default: 0.3s)
        silence_threshold: Silence threshold in dB (default: -40dB)
    
    Returns:
        List of tuples (start_sample, end_sample) for each silence interval
    """
    import numpy as np

    if sample_rate <= 0:
        return []

    samples = np.asarray(audio_data)
    if samples.size == 0:
        return []

    # Convert to mono if stereo
    if samples.ndim > 1:
        audio_mono = samples.mean(axis=1)
    else:
        audio_mono = samples

    frame_length = max(1, int(0.02 * sample_rate))  # 20ms frames
    hop_length = max(1, int(0.01 * sample_rate))  # 10ms hop
    total_samples = audio_mono.shape[0]
    if total_samples < frame_length:
        return []

    squared = np.square(audio_mono, dtype=np.float32)
    cumsum = np.cumsum(squared, dtype=np.float64)
    cumsum = np.concatenate(([0.0], cumsum))

    total_frames = 1 + (total_samples - frame_length) // hop_length
    if total_frames <= 0:
        return []

    starts = np.arange(total_frames, dtype=np.int64) * hop_length
    ends = starts + frame_length
    window_energy = cumsum[ends] - cumsum[starts]
    rms = np.sqrt(np.maximum(window_energy / frame_length, 1e-12))
    energy_db = 20.0 * np.log10(rms)

    if not np.any(energy_db < silence_threshold):
        return []

    is_silence = energy_db < silence_threshold
    min_silence_frames = max(1, int(math.ceil(min_silence_duration * sample_rate / hop_length)))

    padded = np.pad(is_silence.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    start_frames = np.where(changes == 1)[0]
    end_frames = np.where(changes == -1)[0]

    silence_intervals: List[Tuple[int, int]] = []
    for start_frame, end_frame in zip(start_frames, end_frames):
        if end_frame - start_frame >= min_silence_frames:
            start_sample = start_frame * hop_length
            end_sample = min(total_samples, end_frame * hop_length + frame_length)
            silence_intervals.append((int(start_sample), int(end_sample)))

    return silence_intervals

def _smart_cut_audio_at_silence(input_path: str, max_duration: float = 10.0):
    """
    Smart audio cutting that uses silence intervals as natural cut points.
    Tries to find a segment between 3s and 15s that ends at a silence.
    
    Args:
        input_path: Path to input audio file
        max_duration: Maximum duration (unused, kept for compatibility)
    
    Returns:
        Path to cut audio file
    """
    try:
        # Load audio data
        audio_data, sample_rate = sf.read(input_path)
        
        total_duration = len(audio_data) / sample_rate
        
        # If audio is shorter than 3 seconds, return original
        if total_duration < 3.0:
            print(f"📏 Audio duration ({total_duration:.1f}s) is too short, keeping original")
            return input_path
        
        # If audio is between 3s and 15s, return original
        if total_duration <= 15.0:
            print(f"📏 Audio duration ({total_duration:.1f}s) is within ideal range (3-15s)")
            return input_path
        
        print(f"🔍 Analyzing audio ({total_duration:.1f}s) for silence intervals...")
        
        # Detect silence intervals
        silence_intervals = _detect_silence_intervals(audio_data, sample_rate)
        
        if not silence_intervals:
            print(f"⚠️ No silence intervals found, cutting at 10 seconds")
            cut_sample = int(10.0 * sample_rate)
            cut_audio = audio_data[:cut_sample]
        else:
            print(f"✓ Found {len(silence_intervals)} silence intervals")
            
            # Find the best cut point
            best_cut_sample = None
            
            for start_silence, end_silence in silence_intervals:
                # Use the middle of the silence interval as cut point
                cut_sample = (start_silence + end_silence) // 2
                cut_duration = cut_sample / sample_rate
                
                # Check if this cut point gives us a good duration (3s to 15s)
                if 3.0 <= cut_duration <= 15.0:
                    best_cut_sample = cut_sample
                    print(f"✓ Found ideal cut point at {cut_duration:.1f}s (at silence interval)")
                    break
            
            # If no ideal cut point found, try to get closest to target
            if best_cut_sample is None:
                # Find the silence interval closest to 10 seconds
                target_sample = int(10.0 * sample_rate)
                closest_silence = min(silence_intervals, 
                                    key=lambda x: abs((x[0] + x[1]) // 2 - target_sample))
                best_cut_sample = (closest_silence[0] + closest_silence[1]) // 2
                cut_duration = best_cut_sample / sample_rate
                print(f"✓ Using closest silence interval at {cut_duration:.1f}s")
            
            cut_audio = audio_data[:best_cut_sample]
        
        # Create output path with _cut suffix
        input_name = os.path.splitext(input_path)[0]
        output_path = f"{input_name}_cut.wav"
        
        # Save cut audio
        sf.write(output_path, cut_audio, sample_rate)
        
        cut_duration = len(cut_audio) / sample_rate
        
        print(f"✂️ Smart cut: {total_duration:.1f}s → {cut_duration:.1f}s (saved to {os.path.basename(output_path)})")
        
        # Remove original file and return cut file path
        try:
            os.remove(input_path)
        except Exception as cleanup_error:
            print(f"⚠️ Could not remove original audio file: {cleanup_error}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error cutting audio: {e}")
        # Return original path if cutting fails
        return input_path



_AUDIO_SUBTYPE_SAMPLE_WIDTHS = {
    "PCM_16": 2,
    "PCM_U8": 1,
    "PCM_S8": 1,
    "PCM_24": 3,
    "PCM_32": 4,
    "FLOAT": 4,
    "DOUBLE": 8,
    "ULAW": 2,
    "ALAW": 2,
}


def _audio_segment_metadata(audio: AudioSegment, *, hash_md5: Optional[str] = None) -> AudioAssetInfo:
    if audio is None:
        raise ValueError("AudioSegment is required for metadata extraction.")
    frame_rate = int(audio.frame_rate or 16000)
    channels = int(audio.channels or 1)
    sample_width = int(audio.sample_width or 2)
    duration_ms = max(0, len(audio))
    frame_count = int(frame_rate * (duration_ms / 1000.0))
    return AudioAssetInfo(
        duration_ms=duration_ms,
        sample_rate=frame_rate,
        channels=channels,
        sample_width=sample_width,
        frame_count=frame_count,
        hash_md5=hash_md5,
    )


def _probe_audio_metadata_from_path(path: str, *, compute_hash: bool = False) -> AudioAssetInfo:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio path not found: {path}")
    info = sf.info(path)
    sample_rate = int(getattr(info, "samplerate", 0) or 0)
    channels = int(getattr(info, "channels", 0) or 1)
    frame_count = int(getattr(info, "frames", 0) or 0)
    duration_ms = 0
    if getattr(info, "duration", None):
        duration_ms = int(float(info.duration) * 1000)
    elif sample_rate > 0:
        duration_ms = int((frame_count / sample_rate) * 1000)
    subtype = (getattr(info, "subtype", None) or "").upper()
    sample_width = _AUDIO_SUBTYPE_SAMPLE_WIDTHS.get(subtype, 2)
    hash_md5 = _compute_file_md5(path) if compute_hash else None
    return AudioAssetInfo(
        duration_ms=duration_ms,
        sample_rate=sample_rate or 0,
        channels=channels or 1,
        sample_width=sample_width,
        frame_count=frame_count,
        hash_md5=hash_md5,
    )


def _read_analysis_samples_from_path(
    path: str,
    *,
    target_sample_rate: int = 16_000,
) -> Tuple[np.ndarray, int, int]:
    desired_rate = target_sample_rate if target_sample_rate and target_sample_rate > 0 else None
    try:
        samples, sample_rate = librosa.load(
            path,
            sr=desired_rate,
            mono=True,
        )
        analysis_rate = int(desired_rate or sample_rate)
        downsample_factor = 1
        return np.asarray(samples, dtype=np.float32), analysis_rate, downsample_factor
    except Exception as exc:
        print(f"⚠️ librosa.load failed for {path}: {exc}")
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    analysis_rate = int(sample_rate or desired_rate or 16000)
    downsample_factor = 1
    if (
        target_sample_rate
        and target_sample_rate > 0
        and sample_rate
        and sample_rate > target_sample_rate
    ):
        try:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=target_sample_rate)
            downsample_factor = max(1, int(round(sample_rate / target_sample_rate)))
            analysis_rate = target_sample_rate
        except Exception as resample_exc:
            print(f"⚠️ librosa.resample failed for {path}: {resample_exc}")
    return np.asarray(samples, dtype=np.float32), analysis_rate, downsample_factor


def _load_processed_audio(path: str, *, target_sample_rate: int = 16_000) -> AudioBuffer:
    info = _probe_audio_metadata_from_path(path, compute_hash=False)
    samples, analysis_rate, downsample_factor = _read_analysis_samples_from_path(
        path,
        target_sample_rate=target_sample_rate,
    )
    return AudioBuffer(
        path=path,
        sample_rate=info.sample_rate,
        channels=info.channels,
        frame_count=info.frame_count,
        duration_ms=info.duration_ms,
        analysis_samples=samples,
        analysis_sample_rate=analysis_rate,
        analysis_downsample_factor=downsample_factor,
    )


def _audio_source_duration_ms(audio: Union[AudioSegment, AudioBuffer]) -> int:
    if isinstance(audio, AudioBuffer):
        return int(audio.duration_ms)
    return len(audio)


def _audio_segment_to_ndarray(audio: AudioSegment) -> Tuple[np.ndarray, int]:
    """Convert a pydub AudioSegment into a normalized numpy array (float32)."""
    if audio is None:
        return np.zeros((0,), dtype=np.float32), 16000
    frame_rate = int(audio.frame_rate or 16000)
    sample_width = max(1, int(audio.sample_width or 2))
    raw_samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        try:
            raw_samples = raw_samples.reshape((-1, audio.channels))
        except ValueError:
            raw_samples = raw_samples.reshape((-1, audio.channels), order="F")
    max_val = float(1 << (8 * sample_width - 1))
    if max_val > 0:
        normalized = raw_samples.astype(np.float32) / max_val
    else:
        normalized = raw_samples.astype(np.float32)
    return normalized, frame_rate


def _prepare_analysis_samples(
    audio: Union[AudioSegment, AudioBuffer],
    target_sample_rate: int = 16_000,
) -> Tuple[np.ndarray, int, int]:
    """
    Prepare a numpy array for analysis-heavy operations.
    Returns (samples, effective_sample_rate, downsample_factor).
    """
    if isinstance(audio, AudioBuffer):
        return audio.ensure_analysis_samples(target_sample_rate)
    samples, sample_rate = _audio_segment_to_ndarray(audio)
    if sample_rate <= 0:
        return samples, 1, 1

    downsample_factor = 1
    if target_sample_rate and sample_rate > target_sample_rate:
        downsample_factor = max(1, sample_rate // target_sample_rate)
        samples = samples[::downsample_factor].copy()
        sample_rate = max(1, sample_rate // downsample_factor)

    return samples, sample_rate, downsample_factor


def _detect_silence_midpoints_in_window(
    audio: Optional[Union[AudioSegment, AudioBuffer]] = None,
    *,
    min_silence_ms: int,
    silence_threshold_db: float,
    offset_ms: int = 0,
    samples: Optional[np.ndarray] = None,
    sample_rate: Optional[int] = None,
) -> List[Dict[str, int]]:
    """
    Detect silence intervals within a short audio window and return rich interval metadata.
    """
    if samples is None or sample_rate is None:
        if audio is None:
            return []
        samples, sample_rate = _audio_segment_to_ndarray(audio)

    silence_seconds = max(min_silence_ms / 1000.0, 0.1)
    silence_intervals = _detect_silence_intervals(
        samples,
        sample_rate,
        min_silence_duration=silence_seconds,
        silence_threshold=silence_threshold_db,
    )
    intervals: List[Dict[str, int]] = []
    for start, end in silence_intervals:
        if end <= start:
            continue
        start_ms = offset_ms + int(start * 1000 / sample_rate)
        end_ms = offset_ms + int(end * 1000 / sample_rate)
        midpoint_ms = offset_ms + int(((start + end) / 2) * 1000 / sample_rate)
        intervals.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "midpoint_ms": midpoint_ms,
                "duration_ms": max(0, end_ms - start_ms),
            }
        )
    return intervals


def _detect_silence_midpoints_parallel(
    audio: Union[AudioSegment, AudioBuffer],
    *,
    min_silence_ms: int,
    silence_threshold_db: float,
    window_ms: int = 240_000,
    analysis_samples: Optional[np.ndarray] = None,
    analysis_sample_rate: Optional[int] = None,
    analysis_downsample_factor: int = 1,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Split the audio into overlapping windows and detect silence midpoints in parallel.
    Returns (sorted_midpoints, stats). Full interval metadata is included in stats["silence_intervals_ms"].
    """
    total_duration_ms = _audio_source_duration_ms(audio)
    if total_duration_ms <= 0:
        return [], {"windows": 0, "strategy": "empty"}

    detection_start = time.perf_counter()
    if analysis_samples is None or analysis_sample_rate is None:
        analysis_samples, analysis_sample_rate, analysis_downsample_factor = _prepare_analysis_samples(audio)

    analysis_samples = np.asarray(analysis_samples)
    analysis_sample_rate = max(1, int(analysis_sample_rate or 1))
    samples_per_ms = analysis_sample_rate / 1000.0
    total_samples = analysis_samples.shape[0]
    if total_samples <= 0:
        return [], {
            "windows": 0,
            "strategy": "empty",
            "analysis_sample_rate": analysis_sample_rate,
            "analysis_downsample_factor": analysis_downsample_factor,
        }

    effective_window = max(window_ms, min_silence_ms * 4, 60_000)
    overlap_ms = max(min_silence_ms * 2, 2_000)
    step_ms = max(10_000, effective_window - overlap_ms)

    tasks: List[Tuple[int, np.ndarray]] = []
    start_ms = 0
    while start_ms < total_duration_ms:
        end_ms = min(total_duration_ms, start_ms + effective_window)
        start_sample = int(start_ms * samples_per_ms)
        end_sample = int(math.ceil(end_ms * samples_per_ms))
        if end_sample <= start_sample:
            end_sample = min(total_samples, start_sample + max(1, int(samples_per_ms * min_silence_ms)))
        end_sample = min(total_samples, end_sample)
        window_samples = analysis_samples[start_sample:end_sample]
        if window_samples.size == 0:
            break
        tasks.append((start_ms, window_samples))
        if end_ms >= total_duration_ms:
            break
        start_ms += step_ms

    intervals: List[Dict[str, int]] = []
    stats: Dict[str, Any] = {
        "windows": len(tasks),
        "strategy": "single" if len(tasks) <= 1 else "parallel",
        "analysis_sample_rate": analysis_sample_rate,
        "analysis_downsample_factor": analysis_downsample_factor,
        "window_ms": effective_window,
        "overlap_ms": overlap_ms,
        "step_ms": step_ms,
    }

    if not tasks:
        stats["elapsed_ms"] = (time.perf_counter() - detection_start) * 1000
        return [], stats

    if len(tasks) == 1:
        offset_ms, window_samples = tasks[0]
        intervals.extend(
            _detect_silence_midpoints_in_window(
                min_silence_ms=min_silence_ms,
                silence_threshold_db=silence_threshold_db,
                offset_ms=offset_ms,
                samples=window_samples,
                sample_rate=analysis_sample_rate,
            )
        )
    else:
        futures: List[Any] = []
        for offset_ms, window_samples in tasks:
            futures.append(
                executor.submit(
                    _detect_silence_midpoints_in_window,
                    min_silence_ms=min_silence_ms,
                    silence_threshold_db=silence_threshold_db,
                    offset_ms=offset_ms,
                    samples=window_samples,
                    sample_rate=analysis_sample_rate,
                )
            )

        for future in futures:
            try:
                intervals.extend(future.result())
            except Exception as exc:
                print(f"⚠️ Silence detection window failed: {exc}")

    dedup: Dict[int, Dict[str, int]] = {}
    for entry in intervals:
        midpoint_ms = int(entry.get("midpoint_ms", 0) if isinstance(entry, dict) else int(entry))
        if midpoint_ms <= 0:
            continue
        duration_ms = int(entry.get("duration_ms", 0)) if isinstance(entry, dict) else 0
        existing = dedup.get(midpoint_ms)
        if existing is None or duration_ms > existing.get("duration_ms", 0):
            if isinstance(entry, dict):
                dedup[midpoint_ms] = entry
            else:
                dedup[midpoint_ms] = {
                    "midpoint_ms": midpoint_ms,
                    "duration_ms": duration_ms,
                    "start_ms": midpoint_ms,
                    "end_ms": midpoint_ms,
                }

    interval_list = [dedup[key] for key in sorted(dedup.keys())]
    midpoints_only = [entry.get("midpoint_ms", 0) for entry in interval_list]

    stats["silence_intervals_ms"] = interval_list
    stats["silence_count"] = len(midpoints_only)
    stats["max_silence_ms"] = max((entry.get("duration_ms", 0) for entry in interval_list), default=0)
    stats["elapsed_ms"] = (time.perf_counter() - detection_start) * 1000
    return midpoints_only, stats


def _plan_chunk_ranges(
    audio: Union[AudioSegment, AudioBuffer],
    *,
    min_chunk_ms: int,
    max_chunk_ms: int,
    min_silence_ms: int = CHUNK_SPLIT_MIN_SILENCE_MS,
    silence_threshold_db: float = CHUNK_SPLIT_SILENCE_THRESHOLD_DB,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Determine chunk boundaries for long-form audio by opportunistically snapping to nearby silence.
    Returns a list of chunk metadata dictionaries plus diagnostics.
    """
    total_duration_ms = _audio_source_duration_ms(audio)
    if total_duration_ms <= 0:
        return [(0, 0)], {
            "silence_probes": 0,
            "silence_points": 0,
            "silence_count": 0,
            "timing_ms": {},
        }
    min_chunk_ms = max(60_000, int(min_chunk_ms))
    max_chunk_ms = max(min_chunk_ms, int(max_chunk_ms))
    if total_duration_ms <= max_chunk_ms:
        return [(0, total_duration_ms)], {
            "silence_probes": 0,
            "silence_points": 0,
            "silence_count": 0,
            "timing_ms": {},
        }

    timing_ms: Dict[str, float] = {}
    analysis_start = time.perf_counter()
    analysis_samples, analysis_sample_rate, analysis_downsample_factor = _prepare_analysis_samples(audio)
    timing_ms["analysis_samples"] = (time.perf_counter() - analysis_start) * 1000

    silence_detection_start = time.perf_counter()
    silence_midpoints_ms, silence_profile = _detect_silence_midpoints_parallel(
        audio,
        min_silence_ms=min_silence_ms,
        silence_threshold_db=silence_threshold_db,
        window_ms=max(max_chunk_ms * 2, 240_000),
        analysis_samples=analysis_samples,
        analysis_sample_rate=analysis_sample_rate,
        analysis_downsample_factor=analysis_downsample_factor,
    )
    timing_ms["silence_detection"] = (time.perf_counter() - silence_detection_start) * 1000

    print(
        f"🧮 chunk planner: analysis prep {timing_ms['analysis_samples']:.1f}ms "
        f"(downsample x{analysis_downsample_factor}, rate={analysis_sample_rate}Hz)"
    )
    print(
        f"🧮 chunk planner: silence detection {timing_ms['silence_detection']:.1f}ms "
        f"(windows={silence_profile.get('windows', 0)}, points={len(silence_midpoints_ms)})"
    )

    chunk_ranges: List[Dict[str, Any]] = []
    current_start = 0
    pointer = 0
    silence_entries = silence_profile.get("silence_intervals_ms") or [
        {
            "midpoint_ms": point,
            "duration_ms": 0,
            "start_ms": point,
            "end_ms": point,
        }
        for point in silence_midpoints_ms
    ]
    silence_entries = sorted(silence_entries, key=lambda entry: entry.get("midpoint_ms", 0))
    silence_midpoints_ms = [entry.get("midpoint_ms", 0) for entry in silence_entries if entry.get("midpoint_ms") is not None]
    silence_profile["silence_count"] = len(silence_midpoints_ms)
    grace_window = min(CHUNK_SPLIT_SILENCE_GRACE_MS, max_chunk_ms // 2 or 15_000)

    range_selection_start = time.perf_counter()
    while current_start < total_duration_ms and len(chunk_ranges) < CHUNK_SPLIT_MAX_CHUNKS:
        desired_min = current_start + min_chunk_ms
        desired_max = min(current_start + max_chunk_ms, total_duration_ms)
        candidate_cut: Optional[int] = None
        cut_from_silence = False
        candidate_silence_duration: Optional[int] = None
        candidate_idx: Optional[int] = None

        while pointer < len(silence_entries) and silence_entries[pointer].get("midpoint_ms", 0) <= current_start:
            pointer += 1

        best_within: Optional[Dict[str, Any]] = None
        best_within_idx: Optional[int] = None
        best_after: Optional[Dict[str, Any]] = None
        best_after_idx: Optional[int] = None

        lookahead = pointer
        while lookahead < len(silence_entries):
            entry = silence_entries[lookahead]
            point = entry.get("midpoint_ms")
            if point is None:
                lookahead += 1
                continue
            if point < desired_min:
                lookahead += 1
                continue
            if point <= desired_max:
                if best_within is None or entry.get("duration_ms", 0) > best_within.get("duration_ms", 0):
                    best_within = entry
                    best_within_idx = lookahead
                lookahead += 1
                continue
            if point <= desired_max + grace_window:
                if best_after is None or entry.get("duration_ms", 0) > best_after.get("duration_ms", 0):
                    best_after = entry
                    best_after_idx = lookahead
                lookahead += 1
                continue
            break

        if best_within is not None:
            candidate_cut = min(int(best_within.get("midpoint_ms", desired_max)), total_duration_ms)
            candidate_silence_duration = int(best_within.get("duration_ms", 0))
            cut_from_silence = True
            candidate_idx = best_within_idx
        elif best_after is not None:
            candidate_cut = min(int(best_after.get("midpoint_ms", desired_max)), total_duration_ms)
            candidate_silence_duration = int(best_after.get("duration_ms", 0))
            cut_from_silence = True
            candidate_idx = best_after_idx
        else:
            candidate_cut = desired_max
            cut_from_silence = False

        if candidate_cut <= current_start:
            candidate_cut = min(total_duration_ms, max(current_start + min_chunk_ms, current_start + 1000))
            cut_from_silence = False
            candidate_silence_duration = None

        chunk_ranges.append(
            {
                "start_ms": current_start,
                "end_ms": candidate_cut,
                "cut_reason": "silence_center" if cut_from_silence else "hard_limit",
                "silence_midpoint_ms": candidate_cut if cut_from_silence else None,
                "silence_duration_ms": candidate_silence_duration if cut_from_silence else None,
            }
        )
        current_start = candidate_cut
        if candidate_idx is not None:
            pointer = max(pointer, candidate_idx + 1)
        else:
            pointer = lookahead

    timing_ms["range_selection"] = (time.perf_counter() - range_selection_start) * 1000

    if chunk_ranges:
        last_range = chunk_ranges[-1]
        if last_range["end_ms"] < total_duration_ms:
            last_range["end_ms"] = total_duration_ms
            if last_range["cut_reason"] != "silence_center":
                last_range["cut_reason"] = "hard_limit"

    merge_start = time.perf_counter()
    merged_ranges: List[Dict[str, Any]] = []
    for entry in chunk_ranges:
        start_ms = entry["start_ms"]
        end_ms = entry["end_ms"]
        duration = end_ms - start_ms
        if merged_ranges and duration < max(10_000, int(0.25 * min_chunk_ms)):
            merged_ranges[-1]["end_ms"] = end_ms
            merged_ranges[-1]["cut_reason"] = entry["cut_reason"]
            merged_ranges[-1]["silence_midpoint_ms"] = entry.get("silence_midpoint_ms")
            merged_ranges[-1]["silence_duration_ms"] = entry.get("silence_duration_ms")
        else:
            merged_ranges.append(entry)

    timing_ms["range_merge"] = (time.perf_counter() - merge_start) * 1000

    print(
        f"🧮 chunk planner: range selection {timing_ms['range_selection']:.1f}ms, "
        f"merge {timing_ms['range_merge']:.1f}ms -> {len(merged_ranges)} chunk(s)"
    )

    diagnostics = {
        "silence_probes": silence_profile.get("windows", 0),
        "silence_points": len(silence_midpoints_ms),
        "silence_count": len(silence_midpoints_ms),
        "analysis_sample_rate": analysis_sample_rate,
        "analysis_downsample_factor": analysis_downsample_factor,
        "timing_ms": timing_ms,
        "silence_profile": silence_profile,
    }

    return merged_ranges, diagnostics


def _append_suffix_to_path(file_path: str, suffix: str) -> str:
    """Create a new path with the given suffix before the extension."""
    path_obj = Path(file_path)
    return str(path_obj.with_name(f"{path_obj.stem}{suffix}{path_obj.suffix}"))


def _persist_audio_upload(bytes_data: bytes, original_filename: Optional[str] = None) -> str:
    """Persist uploaded audio bytes to a temporary file and return its path."""
    suffix = ".tmp"
    if original_filename:
        _, ext = os.path.splitext(original_filename)
        if ext:
            suffix = ext.lower()
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(bytes_data)
    except Exception:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise
    return temp_path


def _transcode_audio_to_wav(source_path: str) -> str:
    """Transcode an arbitrary audio file to WAV (48 kHz PCM16)."""
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        temp_path,
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        stderr_msg = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"Failed to transcode audio for ClearVoice: {stderr_msg.strip()}") from exc
    return temp_path


async def _normalize_audio_to_cached(
    source_path: str,
    target_path: str,
    *,
    delete_source: bool = False,
) -> str:
    """Convert ClearVoice output to MP3 and save it to cache."""
    # Load the audio from source (WAV from ClearVoice)
    audio = await _load_audio_segment_from_path(source_path)
    # Export as MP3 to cache
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    audio.export(target_path, format="mp3", bitrate="128k")
    if delete_source and os.path.exists(source_path) and source_path != target_path:
        try:
            os.remove(source_path)
        except Exception as exc:
            print(f"⚠️ Failed to delete temp ClearVoice output {source_path}: {exc}")
    return target_path


async def _normalize_uploaded_audio(source_path: str) -> str:
    """
    Normalize user-uploaded audio to canonical WAV (48 kHz PCM16) before
    ClearVoice processing and backing extraction.
    """
    normalized_path = await _run_blocking(_transcode_audio_to_wav, source_path)
    if os.path.exists(source_path) and source_path != normalized_path:
        try:
            os.remove(source_path)
        except Exception as exc:
            print(f"⚠️ Failed to remove original upload {source_path}: {exc}")
    return normalized_path


MAX_SAFE_BASE_FILENAME_LENGTH = 180
LANGUAGE_CODE_OVERRIDES = {
    "chinese": "chn",
    "zh": "chn",
    "zh-cn": "chn",
    "zh_cn": "chn",
    "zh-tw": "cnt",
    "english": "en",
    "en": "en",
    "spanish": "es",
    "es": "es",
    "japanese": "jp",
    "ja": "jp",
    "korean": "kr",
    "ko": "kr",
    "german": "de",
    "de": "de",
    "french": "fr",
    "fr": "fr",
}
DEFAULT_BASE_FILENAME = "translated_audio"


def _sanitize_base_filename(raw_name: Optional[str]) -> Optional[str]:
    if not raw_name:
        return None
    base_name = os.path.splitext(os.path.basename(str(raw_name)))[0].strip().strip(".")
    if not base_name:
        return None
    readable_replacements = str.maketrans(
        {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2015": "-",
            "\u2212": "-",
            "\uff0d": "-",
            "\u3000": " ",
            "&": " and ",
            "\uff06": " and ",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
        }
    )
    normalized = unicodedata.normalize("NFKD", base_name.translate(readable_replacements))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", ascii_name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip(" ._-")
    if len(sanitized) > MAX_SAFE_BASE_FILENAME_LENGTH:
        tail_length = 40
        head_length = MAX_SAFE_BASE_FILENAME_LENGTH - tail_length - 1
        sanitized = f"{sanitized[:head_length].rstrip('._-')}_{sanitized[-tail_length:].lstrip('._-')}"
        sanitized = sanitized.strip(" ._-")
    return sanitized or None


def _normalize_base_filename(
    candidate: Optional[str],
    *,
    fallback: Optional[str] = None,
) -> str:
    sanitized = _sanitize_base_filename(candidate)
    if sanitized:
        return sanitized
    fallback_sanitized = _sanitize_base_filename(fallback)
    if fallback_sanitized:
        return fallback_sanitized
    return f"{DEFAULT_BASE_FILENAME}_{uuid.uuid4().hex[:6]}"


def _language_code_from_label(label: Optional[str]) -> str:
    if not label:
        return "translated"
    normalized = label.strip().lower()
    if not normalized:
        return "translated"
    override = LANGUAGE_CODE_OVERRIDES.get(normalized)
    if override:
        return override
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if not compact:
        return "translated"
    if len(compact) <= 3:
        return compact
    return compact[:3]


def _compose_output_stem(
    base_name: str,
    *,
    chunk_index: Optional[int] = None,
    extra: Optional[str] = None,
) -> str:
    parts: List[str] = [base_name]
    if chunk_index is not None:
        parts.append(f"chunk{int(chunk_index):02d}")
    if extra:
        parts.append(extra)
    return "_".join(part for part in parts if part)


def _determine_output_base_name(
    *,
    user_base: Optional[str],
    upload_filename: Optional[str],
    reuse_session: Optional[TranslateSessionData],
) -> str:
    if user_base:
        normalized = _sanitize_base_filename(user_base)
        if normalized:
            return normalized
    if upload_filename:
        normalized = _sanitize_base_filename(upload_filename)
        if normalized:
            return normalized
    if reuse_session and reuse_session.source_base_name:
        normalized = _sanitize_base_filename(reuse_session.source_base_name)
        if normalized:
            return normalized
    return _normalize_base_filename(None)


async def _ensure_clearvoice_input_path(source_path: str) -> Tuple[str, bool]:
    """
    Ensure the ClearVoice input is a WAV file on disk.
    Returns (path, created_temp_flag).
    """
    if source_path.lower().endswith(".wav"):
        return source_path, False
    temp_wav_path = await _run_blocking(_transcode_audio_to_wav, source_path)
    return temp_wav_path, True


def _chunk_output_url(session: TranslateSessionData) -> Optional[str]:
    if session.chunk_output_filename:
        return f"/api/translate_outputs/{session.chunk_output_filename}"
    return None


def _serialize_chunk_session(session: TranslateSessionData) -> Dict[str, Any]:
    start_ms = max(0, session.chunk_start_ms or 0)
    end_ms = max(start_ms, session.chunk_end_ms or _session_audio_duration_ms(session))
    duration_ms = max(0, end_ms - start_ms)
    audio_url = _chunk_output_url(session)
    backing_available = _session_has_backing_audio(session)
    return {
        "chunk_index": session.chunk_index,
        "session_id": session.session_id,
        "reuse_session_id": session.session_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": duration_ms,
        "start_label": _format_ms_to_timestamp(start_ms),
        "end_label": _format_ms_to_timestamp(end_ms),
        "duration_label": _format_ms_to_timestamp(duration_ms),
        "generated": bool(session.chunk_generated),
        "generated_at": session.chunk_generated_at,
        "audio_url": audio_url,
        "output_format": session.chunk_output_format,
        "output_filename": session.chunk_output_filename,
        "backing_available": backing_available,
        "backing_source": session.backing_track_source or "none",
        "vocals_url": f"/api/translate_vocals/{session.session_id}",
        "backing_url": f"/api/translate_backing_track/{session.session_id}" if backing_available else None,
        "batch_id": session.chunk_parent_id,
        "cut_reason": session.chunk_cut_reason or "unknown",
        "silence_midpoint_ms": session.chunk_silence_midpoint_ms,
    }


def _session_to_manifest(session: TranslateSessionData) -> Dict[str, Any]:
    return {
        "session_id": session.session_id,
        "dest_language": session.dest_language,
        "prompt": session.prompt,
        "translate_enabled": session.translate_enabled,
        "response_format": session.response_format,
        "bitrate": session.bitrate,
        "input_mime_type": session.input_mime_type,
        "clearvoice_settings": copy.deepcopy(session.clearvoice_settings),
        "base_segments": copy.deepcopy(session.base_segments),
        "gemini_chunks": copy.deepcopy(session.gemini_chunks),
        "gemini_model": session.gemini_model,
        "gemini_api_key": session.gemini_api_key,
        "translation_llm_model": session.translation_llm_model,
        "transcription_pipeline": session.transcription_pipeline,
        "whisperx_proxy_refiner": session.whisperx_proxy_refiner,
        "qwen_omnivad_enable_diarization": session.qwen_omnivad_enable_diarization,
        "qwen_omnivad_diarization_backend": session.qwen_omnivad_diarization_backend,
        "qwen_omnivad_diarization_min_seconds": session.qwen_omnivad_diarization_min_seconds,
        "qwen_omnivad_enable_forced_aligner": session.qwen_omnivad_enable_forced_aligner,
        "qwen_omnivad_merge_gap_seconds": session.qwen_omnivad_merge_gap_seconds,
        "source_audio_filename": session.source_audio_filename,
        "source_base_name": session.source_base_name,
        "source_video_path": session.source_video_path,
        "source_video_filename": session.source_video_filename,
        "original_audio_path": session.original_audio_path,
        "original_audio_info": session.original_audio_info.to_dict() if session.original_audio_info else None,
        "backing_track_path": session.backing_track_path,
        "backing_audio_info": session.backing_audio_info.to_dict() if session.backing_audio_info else None,
        "backing_track_source": session.backing_track_source,
        "merge_with_backing": session.merge_with_backing,
        "ignore_non_speech": session.ignore_non_speech,
        "preserve_silence_audio": session.preserve_silence_audio,
        "generated_volume_percent": session.generated_volume_percent,
        "backing_volume_percent": session.backing_volume_percent,
        "silence_volume_percent": session.silence_volume_percent,
        "speaker_profiles": copy.deepcopy(session.speaker_profiles),
        "speaker_overrides": copy.deepcopy(session.speaker_overrides),
        "default_speaker_preset": session.default_speaker_preset,
        "default_emotion_weight": session.default_emotion_weight,
        "gemini_raw_text": session.gemini_raw_text,
        "chunk_parent_id": session.chunk_parent_id,
        "chunk_index": session.chunk_index,
        "chunk_start_ms": session.chunk_start_ms,
        "chunk_end_ms": session.chunk_end_ms,
        "chunk_cut_reason": session.chunk_cut_reason,
        "chunk_silence_midpoint_ms": session.chunk_silence_midpoint_ms,
        "chunk_output_path": session.chunk_output_path,
        "chunk_output_filename": session.chunk_output_filename,
        "chunk_output_format": session.chunk_output_format,
        "chunk_generated": session.chunk_generated,
        "chunk_generated_at": session.chunk_generated_at,
        "chunk_source_session_id": session.chunk_source_session_id,
        "created_at": session.created_at,
    }


async def _rehydrate_session_from_manifest(manifest: Dict[str, Any]) -> TranslateSessionData:
    audio_info = AudioAssetInfo.from_dict(manifest.get("original_audio_info"))
    backing_info = AudioAssetInfo.from_dict(manifest.get("backing_audio_info"))
    session = await _create_translate_session(
        None,
        manifest.get("dest_language", "unspecified"),
        manifest.get("prompt", ""),
        manifest.get("translate_enabled", True),
        _normalize_translate_output_format(manifest.get("response_format")),
        manifest.get("bitrate", TRANSLATE_DEFAULT_BITRATE),
        manifest.get("input_mime_type"),
        manifest.get("clearvoice_settings") or {},
        manifest.get("base_segments") or [],
        manifest.get("gemini_chunks") or [],
        manifest.get("gemini_model") or _get_gemini_model_name(),
        manifest.get("gemini_api_key"),
        translation_llm_model=manifest.get("translation_llm_model"),
        transcription_pipeline=manifest.get("transcription_pipeline", "qwen_omnivad"),
        whisperx_proxy_refiner=manifest.get("whisperx_proxy_refiner", False),
        qwen_omnivad_enable_diarization=manifest.get("qwen_omnivad_enable_diarization", True),
        qwen_omnivad_diarization_backend=manifest.get("qwen_omnivad_diarization_backend", "auto"),
        qwen_omnivad_diarization_min_seconds=manifest.get("qwen_omnivad_diarization_min_seconds", 0.0),
        qwen_omnivad_enable_forced_aligner=manifest.get("qwen_omnivad_enable_forced_aligner", True),
        qwen_omnivad_merge_gap_seconds=manifest.get(
            "qwen_omnivad_merge_gap_seconds",
            DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS,
        ),
        backing_track_audio=None,
        backing_track_path=manifest.get("backing_track_path"),
        backing_audio_info=backing_info,
        backing_track_source=manifest.get("backing_track_source", "none"),
        merge_with_backing=manifest.get("merge_with_backing", False),
        ignore_non_speech=manifest.get("ignore_non_speech", False),
        preserve_silence_audio=manifest.get("preserve_silence_audio", False),
        generated_volume_percent=manifest.get("generated_volume_percent", DEFAULT_GENERATED_VOLUME_PERCENT),
        backing_volume_percent=manifest.get("backing_volume_percent", DEFAULT_GENERATED_VOLUME_PERCENT),
        silence_volume_percent=manifest.get("silence_volume_percent", DEFAULT_SILENCE_VOLUME_PERCENT),
        speaker_profiles=manifest.get("speaker_profiles"),
        speaker_overrides=manifest.get("speaker_overrides"),
        gemini_raw_text=manifest.get("gemini_raw_text"),
        chunk_parent_id=manifest.get("chunk_parent_id"),
        chunk_index=manifest.get("chunk_index"),
        chunk_start_ms=manifest.get("chunk_start_ms"),
        chunk_end_ms=manifest.get("chunk_end_ms"),
        chunk_cut_reason=manifest.get("chunk_cut_reason"),
        chunk_silence_midpoint_ms=manifest.get("chunk_silence_midpoint_ms"),
        source_audio_filename=manifest.get("source_audio_filename"),
        source_base_name=manifest.get("source_base_name"),
        source_video_path=manifest.get("source_video_path"),
        source_video_filename=manifest.get("source_video_filename"),
        default_speaker_preset=manifest.get("default_speaker_preset"),
        default_emotion_weight=manifest.get("default_emotion_weight"),
        original_audio_path=manifest.get("original_audio_path"),
        original_audio_info=audio_info,
        session_id=manifest.get("session_id"),
        persist_media=False,
    )
    session.chunk_output_path = manifest.get("chunk_output_path")
    session.chunk_output_filename = manifest.get("chunk_output_filename")
    session.chunk_output_format = manifest.get("chunk_output_format")
    session.chunk_generated = manifest.get("chunk_generated", False)
    session.chunk_generated_at = manifest.get("chunk_generated_at")
    session.chunk_source_session_id = manifest.get("chunk_source_session_id")
    return session


def _discover_existing_chunk_artifacts(base_name: str) -> Dict[int, Dict[str, Any]]:
    safe_base = _sanitize_base_filename(base_name) or DEFAULT_BASE_FILENAME
    artifacts: Dict[int, Dict[str, Any]] = {}
    scan_dir = os.path.abspath(TRANSLATE_OUTPUT_DIR)
    if not os.path.exists(scan_dir):
        return artifacts
    audio_pattern = re.compile(
        rf"^{re.escape(safe_base)}_chunk(?P<idx>\d+)_(?P<label>[a-z0-9]+)\.(?P<ext>[a-z0-9]+)$",
        re.IGNORECASE,
    )
    srt_pattern = re.compile(
        rf"^{re.escape(safe_base)}_chunk(?P<idx>\d+)_(?P<label>[a-z0-9]+)\.srt$",
        re.IGNORECASE,
    )
    for entry in os.scandir(scan_dir):
        if not entry.is_file():
            continue
        match = audio_pattern.match(entry.name)
        if match:
            idx = int(match.group("idx"))
            ext = match.group("ext").lower()
            artifacts.setdefault(idx, {}).setdefault("audio", []).append(
                {
                    "path": os.path.abspath(entry.path),
                    "url": f"/api/translate_outputs/{entry.name}",
                    "suffix": match.group("label"),
                    "filename": entry.name,
                    "format": ext,
                }
            )
            continue
        srt_match = srt_pattern.match(entry.name)
        if srt_match:
            idx = int(srt_match.group("idx"))
            artifacts.setdefault(idx, {}).setdefault("subtitles", []).append(
                {
                    "path": os.path.abspath(entry.path),
                    "url": f"/api/translate_outputs/{entry.name}",
                    "suffix": srt_match.group("label"),
                    "filename": entry.name,
                }
            )
    return artifacts


def _parse_srt_entries(path: str, limit: int = 10) -> List[str]:
    entries: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as infile:
            buffer: List[str] = []
            for line in infile:
                stripped = line.strip()
                if not stripped:
                    if buffer:
                        entries.append(" ".join(buffer).strip())
                        buffer = []
                        if len(entries) >= limit:
                            break
                    continue
                if stripped.isdigit():
                    buffer = []
                    continue
                if "-->" in stripped:
                    continue
                buffer.append(stripped)
            if buffer and len(entries) < limit:
                entries.append(" ".join(buffer).strip())
    except Exception as exc:
        print(f"⚠️ Failed to parse SRT at {path}: {exc}")
    return entries


def _srt_matches_manifest(manifest: Dict[str, Any], srt_path: str, limit: int = 10) -> bool:
    segments = manifest.get("base_segments") or []
    manifest_texts: List[str] = []
    for segment in segments:
        if segment.get("type") == "silence":
            continue
        text = (segment.get("translated_text") or segment.get("source_text") or "").strip()
        if text:
            manifest_texts.append(text.lower())
        if len(manifest_texts) >= limit:
            break
    if not manifest_texts:
        return False
    srt_entries = [entry.lower() for entry in _parse_srt_entries(srt_path, limit=limit) if entry]
    if not srt_entries:
        return False
    compare_len = min(len(manifest_texts), len(srt_entries))
    return all(manifest_texts[i] == srt_entries[i] for i in range(compare_len))


def _mark_chunk_generated(
    session: TranslateSessionData,
    output_path: str,
    output_filename: str,
    response_format: str,
) -> None:
    response_format = _normalize_translate_output_format(response_format)
    session.chunk_generated = True
    session.chunk_output_path = output_path
    session.chunk_output_filename = output_filename
    session.chunk_output_format = response_format
    session.chunk_generated_at = time.time()
    _refresh_split_cache_for_session(session)


async def _generate_chunk_audio_from_session(
    chunk_session: TranslateSessionData,
    *,
    dest_language: str,
    response_format: str,
    bitrate: str,
    gemini_model: str,
    gemini_api_key: Optional[str],
    translation_llm_model: Optional[str],
    ignore_non_speech: bool,
    preserve_silence_audio: bool,
    generated_volume_percent: float,
    backing_volume_percent: float,
    merge_backing_track: bool,
    silence_volume_percent: float,
    force_gemini_regenerate: bool = False,
    transcription_pipeline: Optional[str] = None,
    whisperx_proxy_refiner: Any = None,
    default_speaker_preset: Optional[str] = None,
    default_emotion_weight: Optional[float] = None,
    qwen_omnivad_enable_diarization: Any = None,
    qwen_omnivad_diarization_backend: Any = None,
    qwen_omnivad_diarization_min_seconds: Any = None,
    qwen_omnivad_enable_forced_aligner: Any = None,
    qwen_omnivad_merge_gap_seconds: Any = None,
) -> Tuple[str, str, Dict[str, Any]]:
    response_format = _normalize_translate_output_format(response_format)
    transcription_pipeline = (
        transcription_pipeline
        or getattr(chunk_session, "transcription_pipeline", None)
        or "qwen_omnivad"
    ).strip().lower()
    transcription_pipeline = _normalize_transcription_pipeline(transcription_pipeline)
    whisperx_proxy_refiner = (
        _coerce_to_bool(
            whisperx_proxy_refiner
            if whisperx_proxy_refiner is not None
            else getattr(chunk_session, "whisperx_proxy_refiner", False)
        )
        and transcription_pipeline == "whisperx"
    )
    (
        qwen_omnivad_enable_diarization,
        qwen_omnivad_diarization_min_seconds,
    ) = _resolve_qwen_omnivad_diarization_options(
        transcription_pipeline,
        (
            qwen_omnivad_enable_diarization
            if qwen_omnivad_enable_diarization is not None
            else getattr(chunk_session, "qwen_omnivad_enable_diarization", True)
        ),
        (
            qwen_omnivad_diarization_min_seconds
            if qwen_omnivad_diarization_min_seconds is not None
            else getattr(chunk_session, "qwen_omnivad_diarization_min_seconds", 0.0)
        ),
    )
    qwen_omnivad_diarization_backend = _resolve_qwen_omnivad_diarization_backend_option(
        transcription_pipeline,
        (
            qwen_omnivad_diarization_backend
            if qwen_omnivad_diarization_backend is not None
            else getattr(chunk_session, "qwen_omnivad_diarization_backend", "auto")
        ),
    )
    qwen_omnivad_enable_forced_aligner = _resolve_qwen_omnivad_forced_aligner_option(
        transcription_pipeline,
        (
            qwen_omnivad_enable_forced_aligner
            if qwen_omnivad_enable_forced_aligner is not None
            else getattr(chunk_session, "qwen_omnivad_enable_forced_aligner", True)
        ),
    )
    qwen_omnivad_merge_gap_seconds = _resolve_qwen_omnivad_merge_gap_seconds_option(
        transcription_pipeline,
        (
            qwen_omnivad_merge_gap_seconds
            if qwen_omnivad_merge_gap_seconds is not None
            else getattr(
                chunk_session,
                "qwen_omnivad_merge_gap_seconds",
                DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS,
            )
        ),
    )

    clearvoice_settings = chunk_session.clearvoice_settings or {}
    apply_enhancement = bool(clearvoice_settings.get("enhancement"))
    apply_super_resolution = bool(clearvoice_settings.get("super_resolution"))
    enhancement_model_from_session = clearvoice_settings.get("enhancement_model")
    if apply_super_resolution and not apply_enhancement:
        apply_enhancement = True
        clearvoice_settings["enhancement"] = True
    parallel_config = _parallel_config_from_settings(clearvoice_settings)
    audio_separator_use_soundfile = (
        AUDIO_SEPARATOR_USE_SOUNDFILE
        if clearvoice_settings.get("audio_separator_use_soundfile") is None
        else _coerce_to_bool(clearvoice_settings.get("audio_separator_use_soundfile"))
    )
    resolved_base_name = _normalize_base_filename(
        chunk_session.source_base_name,
        fallback=chunk_session.source_audio_filename,
    )

    (
        original_audio,
        input_mime_type_value,
        processed_audio_bytes,
        gemini_mime_type,
        backing_track_audio,
        merge_with_backing,
        backing_track_source,
        cached_vocals_path,
        cached_backing_path,
    ) = await _prepare_audio_assets(
        reuse_source_session=chunk_session,
        audio_file=None,
        audio_reference=None,
        preloaded_audio_bytes=None,
        audio_mime_type_value=chunk_session.input_mime_type,
        apply_enhancement=apply_enhancement,
        apply_super_resolution=apply_super_resolution,
        requested_merge_backing=merge_backing_track,
        custom_backing_audio_file=None,
        custom_backing_audio_reference=None,
        custom_backing_mime_type_value=None,
        emit_status=None,
        clearvoice_parallel_config=parallel_config,
        enhancement_model_name=enhancement_model_from_session,
        audio_separator_use_soundfile=audio_separator_use_soundfile,
    )

    final_prompt = _resolve_final_prompt(
        chunk_session.prompt,
        dest_language,
        True,
        ignore_non_speech,
    )

    segment_result = await _build_translation_segments(
        original_audio=original_audio,
        processed_audio_bytes=processed_audio_bytes,
        gemini_mime_type=gemini_mime_type,
        dest_language=dest_language,
        final_prompt=final_prompt,
        translate_enabled=True,
        response_format_value=response_format,
        bitrate_value=bitrate,
        input_mime_type=input_mime_type_value,
        apply_enhancement=apply_enhancement,
        apply_super_resolution=apply_super_resolution,
        ignore_non_speech_flag=ignore_non_speech,
        preserve_silence_audio_flag=preserve_silence_audio,
        generated_volume_percent_value=generated_volume_percent,
        backing_volume_percent_value=backing_volume_percent,
        silence_volume_percent_value=silence_volume_percent,
        backing_track_audio=backing_track_audio,
        backing_track_source=backing_track_source,
        merge_with_backing=merge_with_backing,
        segments_override_value=None,
        min_speech_duration=MIN_SPEECH_DURATION_MS,
        max_merge_interval=MAX_MERGE_INTERVAL_MS,
        resolved_gemini_model=gemini_model,
        gemini_api_key_value=(gemini_api_key or "").strip() or None,
        translation_llm_model=translation_llm_model,
        emit_status=None,
        source_chunk_session=chunk_session,
        source_audio_filename=chunk_session.source_audio_filename,
        source_base_name=chunk_session.source_base_name,
        force_gemini_regenerate=force_gemini_regenerate,
        initial_speaker_overrides=chunk_session.speaker_overrides,
        default_speaker_preset=default_speaker_preset,
        default_emotion_weight=default_emotion_weight,
        clearvoice_settings=clearvoice_settings,
        transcription_pipeline=transcription_pipeline,
        whisperx_proxy_refiner=whisperx_proxy_refiner,
        qwen_omnivad_enable_diarization=qwen_omnivad_enable_diarization,
        qwen_omnivad_diarization_backend=qwen_omnivad_diarization_backend,
        qwen_omnivad_diarization_min_seconds=qwen_omnivad_diarization_min_seconds,
        qwen_omnivad_enable_forced_aligner=qwen_omnivad_enable_forced_aligner,
        qwen_omnivad_merge_gap_seconds=qwen_omnivad_merge_gap_seconds,
    )

    audio_payload, _media_type, synthesis_metadata = await _synthesize_translated_audio(
        original_audio,
        segment_result.segments,
        dest_language,
        response_format=response_format,
        bitrate=bitrate,
        input_mime_type=input_mime_type_value,
        clearvoice_settings=clearvoice_settings,
        backing_track_audio=backing_track_audio,
        backing_track_source=backing_track_source,
        merge_with_backing=merge_with_backing,
        preserve_silence_audio=preserve_silence_audio,
        generated_volume_percent=generated_volume_percent,
        silence_volume_percent=silence_volume_percent,
        speaker_overrides=chunk_session.speaker_overrides,
        backing_volume_percent=backing_volume_percent,
        default_speaker_preset=default_speaker_preset,
        default_emotion_weight=default_emotion_weight,
        return_unmixed_audio=merge_with_backing and backing_track_audio is not None,
    )

    metadata = dict(segment_result.metadata)
    metadata.update(synthesis_metadata or {})
    metadata["ignore_non_speech"] = ignore_non_speech
    metadata["preserve_silence_audio"] = preserve_silence_audio
    metadata["generated_volume_percent"] = generated_volume_percent
    metadata["backing_volume_percent"] = backing_volume_percent
    metadata["silence_volume_percent"] = silence_volume_percent
    metadata["gemini_model"] = gemini_model
    metadata["translation_llm_model"] = _normalize_translation_llm_model(
        translation_llm_model
    )
    metadata["transcription_pipeline"] = transcription_pipeline
    metadata["whisperx_proxy_refiner"] = whisperx_proxy_refiner
    metadata["qwen_omnivad_diarization_backend"] = qwen_omnivad_diarization_backend
    metadata["qwen_omnivad_enable_forced_aligner"] = qwen_omnivad_enable_forced_aligner
    metadata["qwen_omnivad_merge_gap_seconds"] = qwen_omnivad_merge_gap_seconds
    metadata["output_base_name"] = resolved_base_name
    metadata["default_speaker_preset"] = default_speaker_preset
    metadata["default_emotion_weight"] = default_emotion_weight

    language_code = _language_code_from_label(dest_language)
    base_stem = _compose_output_stem(resolved_base_name, chunk_index=chunk_session.chunk_index)
    audio_stem = _compose_output_stem(
        resolved_base_name,
        chunk_index=chunk_session.chunk_index,
        extra=language_code,
    )
    output_filename = f"{audio_stem}.{response_format}"
    output_path = os.path.join(TRANSLATE_OUTPUT_DIR, output_filename)
    with open(output_path, "wb") as outfile:
        outfile.write(audio_payload)
    audio_url = f"/api/translate_outputs/{output_filename}"
    unmixed_audio_bytes = metadata.pop("_unmixed_audio_bytes", None)
    if unmixed_audio_bytes:
        unmixed_filename = f"{audio_stem}_vocals.{response_format}"
        unmixed_path = os.path.join(TRANSLATE_OUTPUT_DIR, unmixed_filename)
        with open(unmixed_path, "wb") as outfile:
            outfile.write(unmixed_audio_bytes)
        metadata["translated_vocals_url"] = f"/api/translate_outputs/{unmixed_filename}"
        metadata["translated_vocals_file_name"] = unmixed_filename

    subtitle_translated = _export_srt_from_segments(
        segment_result.segments,
        base_name=base_stem,
        suffix=language_code,
        text_kind="translated",
        empty_note="No speech segments were available for subtitle export.",
    )
    subtitle_original = _export_srt_from_segments(
        segment_result.segments,
        base_name=base_stem,
        suffix="original",
        text_kind="source",
        empty_note="No speech segments were available for subtitle export.",
    )
    metadata["subtitle"] = subtitle_translated
    metadata["subtitle_translated"] = subtitle_translated
    metadata["subtitle_original"] = subtitle_original
    metadata["language_code"] = language_code

    _mark_chunk_generated(chunk_session, output_path, output_filename, response_format)
    metadata["chunk"] = _serialize_chunk_session(chunk_session)

    chunk_session.dest_language = dest_language
    chunk_session.ignore_non_speech = ignore_non_speech
    chunk_session.preserve_silence_audio = preserve_silence_audio
    chunk_session.generated_volume_percent = generated_volume_percent
    chunk_session.backing_volume_percent = backing_volume_percent
    chunk_session.silence_volume_percent = silence_volume_percent
    chunk_session.gemini_model = gemini_model

    await _update_translate_session_metadata(
        chunk_session.session_id,
        response_format=response_format,
        bitrate=bitrate,
        gemini_model=gemini_model,
        generated_volume_percent=generated_volume_percent,
        backing_volume_percent=backing_volume_percent,
    )
    await _update_translate_session_segments(
        chunk_session.session_id,
        segment_result.ui_segments,
    )

    return audio_url, output_filename, metadata


def _load_chunk_audio_for_merge(session: TranslateSessionData) -> Optional[AudioSegment]:
    if (
        session.chunk_generated
        and session.chunk_output_path
        and os.path.exists(session.chunk_output_path)
    ):
        ext = os.path.splitext(session.chunk_output_path)[1].lstrip(".") or None
        try:
            return AudioSegment.from_file(session.chunk_output_path, format=ext)
        except Exception as exc:
            print(f"⚠️ Failed to load generated chunk audio ({session.session_id}): {exc}")
    try:
        return _get_session_original_audio(session)
    except RuntimeError as exc:
        print(f"⚠️ Chunk session '{session.session_id}' is missing original audio: {exc}")
        return None


def _resolve_session_chunk_audio_path(session: TranslateSessionData) -> Optional[str]:
    candidates = [
        session.chunk_output_path,
        getattr(session, "original_audio_path", None),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _extract_backing_track_from_vocals(
    original_mix: AudioSegment,
    enhanced_vocals: AudioSegment,
) -> Optional[AudioSegment]:
    """
    Approximate an instrumental backing track by subtracting the ClearVoice
    MossFormer2 vocals estimate from the original mixture.
    """
    if original_mix is None or enhanced_vocals is None:
        return None

    try:
        target_rate = int(original_mix.frame_rate or enhanced_vocals.frame_rate or 44100)
        target_channels = int(original_mix.channels or enhanced_vocals.channels or 1)
        target_sample_width = 2  # work in int16 space for predictable clipping

        mix_aligned = (
            original_mix.set_frame_rate(target_rate)
            .set_channels(target_channels)
            .set_sample_width(target_sample_width)
        )
        vocals_aligned = (
            enhanced_vocals.set_frame_rate(target_rate)
            .set_channels(target_channels)
            .set_sample_width(target_sample_width)
        )

        mix_samples = np.array(mix_aligned.get_array_of_samples(), dtype=np.float32)
        vocal_samples = np.array(vocals_aligned.get_array_of_samples(), dtype=np.float32)

        if target_channels > 1:
            mix_samples = mix_samples.reshape((-1, target_channels))
            vocal_samples = vocal_samples.reshape((-1, target_channels))

        min_len = min(len(mix_samples), len(vocal_samples))
        if min_len == 0:
            return None

        residual = np.copy(mix_samples)
        residual[:min_len] = mix_samples[:min_len] - vocal_samples[:min_len]
        if len(mix_samples) > min_len:
            residual[min_len:] = mix_samples[min_len:]

        clip_min = np.iinfo(np.int16).min
        clip_max = np.iinfo(np.int16).max
        residual = np.clip(residual, clip_min, clip_max).astype(np.int16)

        if target_channels > 1:
            residual = residual.reshape(-1)

        return AudioSegment(
            residual.tobytes(),
            frame_rate=target_rate,
            sample_width=target_sample_width,
            channels=target_channels,
        )
    except Exception as exc:
        print(f"⚠️ Failed to extract backing track from ClearVoice output: {exc}")
        return None


def _plan_clearvoice_parallel_chunks(total_ms: int, chunk_ms: int) -> List[Tuple[int, int]]:
    chunk_ms = max(1_000, int(chunk_ms))
    total_ms = max(0, int(total_ms))
    if total_ms <= 0:
        return []
    ranges: List[Tuple[int, int]] = []
    start = 0
    max_chunks = max(1, CLEARVOICE_PARALLEL_MAX_CHUNKS)
    while start < total_ms and len(ranges) < max_chunks:
        # If we're at the cap, consume the remainder to avoid truncation.
        if len(ranges) == max_chunks - 1:
            end = total_ms
        else:
            end = min(total_ms, start + chunk_ms)
        if end <= start:
            break
        ranges.append((start, end))
        start = end
    return ranges


def _run_clearvoice_chunk_job(job: ClearVoiceParallelChunkJob) -> ClearVoiceParallelChunkResult:
    final_path, generated_paths, enhancement_path = _apply_clearvoice_processing_sync(
        job.chunk_path,
        job.apply_enhancement,
        job.apply_super_resolution,
        enhancement_model_name=job.enhancement_model_name,
    )
    cleanup_targets = set(generated_paths or [])
    cleanup_targets.add(final_path)
    if enhancement_path:
        cleanup_targets.add(enhancement_path)
    return ClearVoiceParallelChunkResult(
        chunk_idx=job.chunk_idx,
        final_path=final_path,
        enhancement_path=enhancement_path,
        generated_paths=list(cleanup_targets),
    )


def _apply_clearvoice_parallel_sync(
    input_path: str,
    total_duration_ms: int,
    apply_enhancement: bool,
    apply_super_resolution: bool,
    config: ClearVoiceParallelConfig,
    enhancement_model_name: Optional[str] = None,
) -> Tuple[str, List[str], Optional[str]]:
    if not config or not config.enabled:
        return _apply_clearvoice_processing_sync(input_path, apply_enhancement, apply_super_resolution, enhancement_model_name=enhancement_model_name)
    if not _ffmpeg_available():
        print("⚠️ ClearVoice parallel requested but ffmpeg is unavailable. Falling back to sequential run.")
        return _apply_clearvoice_processing_sync(input_path, apply_enhancement, apply_super_resolution, enhancement_model_name=enhancement_model_name)

    chunk_ranges = _plan_clearvoice_parallel_chunks(total_duration_ms, config.chunk_ms)
    if len(chunk_ranges) <= 1:
        return _apply_clearvoice_processing_sync(input_path, apply_enhancement, apply_super_resolution, enhancement_model_name=enhancement_model_name)

    cleanup_paths: Set[str] = set()

    def _track(path: Optional[str]) -> None:
        if path:
            cleanup_paths.add(path)

    chunk_jobs: List[ClearVoiceParallelChunkJob] = []
    try:
        for idx, (start_ms, end_ms) in enumerate(chunk_ranges):
            fd, chunk_path = tempfile.mkstemp(prefix="cv_parallel_chunk_", suffix=".wav")
            os.close(fd)
            _ffmpeg_extract_segment(input_path, chunk_path, start_ms, end_ms, reencode=True)
            _track(chunk_path)
            chunk_jobs.append(
                ClearVoiceParallelChunkJob(
                    chunk_idx=idx,
                    chunk_path=chunk_path,
                    apply_enhancement=apply_enhancement,
                    apply_super_resolution=apply_super_resolution,
                    enhancement_model_name=enhancement_model_name,
                )
            )
    except Exception:
        for path in cleanup_paths:
            _safe_remove_file(path)
        raise

    ctx = multiprocessing.get_context("spawn")
    worker_count = max(1, min(config.max_workers, len(chunk_jobs)))
    results: List[ClearVoiceParallelChunkResult] = []
    start_time = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            futures = [pool.submit(_run_clearvoice_chunk_job, job) for job in chunk_jobs]
            for future in as_completed(futures):
                results.append(future.result())
    except Exception:
        for path in cleanup_paths:
            _safe_remove_file(path)
        raise

    results.sort(key=lambda item: item.chunk_idx)
    for res in results:
        for generated in res.generated_paths:
            _track(generated)

    final_chunk_paths = [res.final_path for res in results]
    enhancement_chunk_paths = [res.enhancement_path for res in results if res.enhancement_path]

    fd, final_concat_path = tempfile.mkstemp(prefix="cv_parallel_final_", suffix=".wav")
    os.close(fd)
    _ffmpeg_concat_files(final_chunk_paths, final_concat_path, copy_codec=True, target_format="wav")
    _track(final_concat_path)

    enhancement_concat_path: Optional[str] = None
    if apply_enhancement and enhancement_chunk_paths:
        fd, enhancement_concat_path = tempfile.mkstemp(prefix="cv_parallel_enh_", suffix=".wav")
        os.close(fd)
        _ffmpeg_concat_files(
            enhancement_chunk_paths,
            enhancement_concat_path,
            copy_codec=True,
            target_format="wav",
        )
        _track(enhancement_concat_path)

    elapsed = time.perf_counter() - start_time
    print(
        f"⚡ ClearVoice parallel processed {len(final_chunk_paths)} chunk(s) "
        f"in {elapsed:.2f}s using {worker_count} worker(s)."
    )
    return final_concat_path, list(cleanup_paths), enhancement_concat_path


def _apply_clearvoice_processing_sync(
    input_path: str,
    apply_enhancement: bool,
    apply_super_resolution: bool,
    enhancement_model_name: Optional[str] = None,
) -> Tuple[str, List[str], Optional[str]]:
    """Run ClearVoice enhancement/super-resolution synchronously."""
    global _enhancement_model, _super_res_model, _current_enhancement_model_name
    
    if ClearVoice is None:
        raise RuntimeError("ClearVoice package is not available in the environment.")
    
    # Determine which enhancement model to use
    model_name = enhancement_model_name if enhancement_model_name in AVAILABLE_ENHANCEMENT_MODELS else DEFAULT_ENHANCEMENT_MODEL
    
    generated_paths: List[str] = []
    enhancement_output_path: Optional[str] = None
    current_input = input_path
    final_path = input_path
    
    try:
        if apply_enhancement:
            print(f"✨ ClearVoice: Applying {model_name} enhancement...")
            # Initialize enhancement model if not already created or if model changed
            if _enhancement_model is None or _current_enhancement_model_name != model_name:
                if _enhancement_model is not None:
                    print(f"🔄 Switching enhancement model from {_current_enhancement_model_name} to {model_name}...")
                else:
                    print(f"🔧 Initializing enhancement model ({model_name}) (first use)...")
                _enhancement_model = ClearVoice(task="speech_enhancement", model_names=[model_name])
                _current_enhancement_model_name = model_name
            enhancement_output = _enhancement_model(input_path=current_input, online_write=False)
            enhanced_path = _append_suffix_to_path(current_input, "_se")
            _enhancement_model.write(enhancement_output, output_path=enhanced_path)
            generated_paths.append(enhanced_path)
            final_path = enhanced_path
            current_input = enhanced_path
            enhancement_output_path = enhanced_path
            print(f"✅ ClearVoice: Enhancement saved to {os.path.basename(enhanced_path)}")
        
        if apply_super_resolution:
            print("🎛️ ClearVoice: Applying MossFormer2_SR_48K super-resolution...")
            # Initialize super-resolution model if not already created
            if _super_res_model is None:
                print("🔧 Initializing super-resolution model (first use)...")
                _super_res_model = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
            super_res_output = _super_res_model(input_path=current_input, online_write=False)
            super_res_path = _append_suffix_to_path(current_input, "_sr")
            _super_res_model.write(super_res_output, output_path=super_res_path)
            generated_paths.append(super_res_path)
            final_path = super_res_path
            print(f"✅ ClearVoice: Super-resolution saved to {os.path.basename(super_res_path)}")
        
        return final_path, generated_paths, enhancement_output_path
    except Exception:
        for created_path in generated_paths:
            try:
                os.remove(created_path)
            except Exception:
                pass
        raise


async def apply_clearvoice_processing(
    input_path: str,
    apply_enhancement: bool,
    apply_super_resolution: bool,
    enhancement_model_name: Optional[str] = None,
) -> Tuple[str, List[str], Optional[str]]:
    """Async wrapper for ClearVoice processing."""
    if not (apply_enhancement or apply_super_resolution):
        return input_path, []
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        functools.partial(
            _apply_clearvoice_processing_sync,
            input_path,
            apply_enhancement,
            apply_super_resolution,
            enhancement_model_name=enhancement_model_name,
        ),
    )

async def convert_audio_to_format(wav_data, sample_rate, output_format="mp3", bitrate="128k"):
    """Convert audio data to specified format (MP3 or WAV)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _convert_audio_to_format_sync, wav_data, sample_rate, output_format, bitrate)


async def _read_generated_audio_bytes(
    result_path: str,
    response_format: str,
    bitrate: str = "128k",
) -> bytes:
    """Read a generated WAV file and encode it for an API audio response."""
    if response_format != "wav":
        audio_data, sample_rate = await async_audio_read(result_path)
        audio_bytes, _, _ = await convert_audio_to_format(
            audio_data,
            sample_rate,
            response_format,
            bitrate,
        )
        return audio_bytes
    return await async_read_file(result_path)


async def _generated_audio_attachment_response(
    result_path: str,
    response_format: str,
    filename_stem: str = "speech",
) -> Response:
    try:
        audio_bytes = await _read_generated_audio_bytes(result_path, response_format)
    finally:
        await async_remove_file(result_path)

    return Response(
        content=audio_bytes,
        media_type=_audio_media_type(response_format),
        headers={
            "Content-Disposition": f"attachment; filename={filename_stem}.{response_format}",
            "Cache-Control": "no-cache",
        },
    )


def _encode_streaming_audio_chunk(wav_cpu: Any, response_format: str) -> bytes:
    wav_data = wav_cpu.numpy().astype(np.int16)
    with BytesIO() as wav_buffer:
        sf.write(wav_buffer, wav_data.T, 22050, format="WAV")
        wav_bytes = wav_buffer.getvalue()

    if response_format == "wav":
        return wav_bytes

    audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
    with BytesIO() as audio_buffer:
        audio_segment.export(
            audio_buffer,
            format=response_format,
            bitrate="128k" if response_format == "mp3" else None,
        )
        return audio_buffer.getvalue()


def _streaming_audio_frame(chunk_idx: int, audio_bytes: bytes, is_last: bool) -> bytes:
    state = "LAST" if is_last else "MORE"
    header = f"CHUNK:{chunk_idx}:{len(audio_bytes)}:{state}\n".encode("utf-8")
    return header + audio_bytes


STREAMING_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

def _convert_audio_to_format_sync(wav_data, sample_rate, output_format="mp3", bitrate="128k"):
    """Synchronous audio format conversion"""
    try:
        # Convert numpy array to AudioSegment
        # Ensure wav_data is in the right format
        if wav_data.dtype != 'int16':
            # Convert float to int16
            wav_data = (wav_data * 32767).astype('int16')
        
        # Create AudioSegment from raw audio data
        audio_segment = AudioSegment(
            wav_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=wav_data.dtype.itemsize,
            channels=1 if len(wav_data.shape) == 1 else wav_data.shape[1]
        )
        
        # Export to desired format
        with BytesIO() as output_buffer:
            if output_format.lower() == "mp3":
                audio_segment.export(output_buffer, format="mp3", bitrate=bitrate)
                media_type = "audio/mpeg"
                file_extension = "mp3"
            else:
                audio_segment.export(output_buffer, format="wav")
                media_type = "audio/wav" 
                file_extension = "wav"
            
            return output_buffer.getvalue(), media_type, file_extension
            
    except Exception as e:
        print(f"⚠️ Audio conversion failed, falling back to WAV: {e}")
        # Fallback to original soundfile method
        with BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav_data, sample_rate, format='WAV')
            return wav_buffer.getvalue(), "audio/wav", "wav"


# Gemini translation helpers
def _get_gemini_model_name() -> str:
    return os.getenv(GEMINI_MODEL_ENV_VAR, DEFAULT_GEMINI_MODEL_NAME)


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    match = JSON_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()
    # Fallback: extract JSON array if present
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text


def _coerce_gemini_json_payload(raw: str) -> Optional[str]:
    trimmed = (raw or "").strip()
    if not trimmed:
        return None
    match = COERCE_SPEAKER_SEGMENTS_PATTERN.match(trimmed)
    if match:
        speakers_part, segments_part = match.groups()
        return f'{{"speakers": {speakers_part}, "segments": {segments_part}}}'
    return None


SPEAKER_ID_PATTERN = re.compile(r"^speaker\s*(\d+)$", re.IGNORECASE)


def _canonicalize_speaker_id(raw_value: Any, fallback_index: int) -> str:
    fallback_index = max(1, int(fallback_index))
    if isinstance(raw_value, (int, float)) and raw_value >= 1:
        return f"speaker{int(raw_value)}"
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return f"speaker{fallback_index}"
        match = SPEAKER_ID_PATTERN.match(candidate)
        if match:
            return f"speaker{int(match.group(1))}"
        digits = re.findall(r"\d+", candidate)
        if digits:
            return f"speaker{int(digits[0])}"
    return f"speaker{fallback_index}"


def _normalize_speaker_profiles(raw_profiles: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_profiles, list):
        return []
    normalized: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()
    for idx, entry in enumerate(raw_profiles, start=1):
        entry_dict: Dict[str, Any]
        if isinstance(entry, dict):
            entry_dict = entry
        else:
            entry_dict = {"description": str(entry)}
        candidate_id = entry_dict.get("id") or entry_dict.get("speaker") or entry_dict.get("label") or entry_dict.get("name")
        speaker_id = _canonicalize_speaker_id(candidate_id, idx)
        while speaker_id in used_ids:
            idx += 1
            speaker_id = _canonicalize_speaker_id(None, idx)
        used_ids.add(speaker_id)
        description = str(
            entry_dict.get("description")
            or entry_dict.get("summary")
            or entry_dict.get("traits")
            or entry_dict.get("role")
            or ""
        ).strip()
        if not description:
            description = f"{speaker_id.title()} voice"
        display_name = str(entry_dict.get("label") or entry_dict.get("name") or speaker_id.title()).strip()
        normalized.append(
            {
                "id": speaker_id,
                "label": display_name or speaker_id.title(),
                "description": description,
            }
        )
    return normalized


def _normalize_speaker_overrides(
    raw_overrides: Optional[Dict[str, Any]],
    speaker_profiles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_overrides, dict):
        return {}
    valid_ids = {profile.get("id") for profile in (speaker_profiles or []) if profile.get("id")}
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in raw_overrides.items():
        if not isinstance(key, str):
            continue
        speaker_id = _canonicalize_speaker_id(key, fallback_index=1).lower()
        if valid_ids and speaker_id not in valid_ids:
            continue
        if not isinstance(value, dict):
            continue
        preset_name = str(value.get("preset_name") or "").strip()
        volume_value = value.get("volume_percent")
        has_volume_override = volume_value is not None
        if not preset_name and not has_volume_override:
            continue
        entry: Dict[str, Any] = {}
        if preset_name:
            entry["preset_name"] = preset_name
            entry["use_emotion_prompt"] = bool(value.get("use_emotion_prompt"))
            entry["emotion_weight"] = _coerce_emotion_weight(
                value.get("emotion_weight"),
                DEFAULT_EMOTION_WEIGHT,
            )
        if has_volume_override:
            entry["volume_percent"] = _coerce_volume_percent(
                volume_value,
                DEFAULT_GENERATED_VOLUME_PERCENT,
            )
        normalized[speaker_id] = entry
    return normalized


def _parse_gemini_json(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        coerced_payload = _coerce_gemini_json_payload(cleaned)
        if coerced_payload:
            try:
                data = json.loads(coerced_payload)
            except json.JSONDecodeError as coerced_exc:
                preview = cleaned if len(cleaned) <= 4000 else f"{cleaned[:4000]}...<truncated>"
                print("❌ Gemini JSON parse error (after coercion). Raw response preview:\n", preview)
                raise ValueError(f"Gemini returned invalid JSON: {coerced_exc}") from coerced_exc
        else:
            preview = cleaned if len(cleaned) <= 4000 else f"{cleaned[:4000]}...<truncated>"
            print("❌ Gemini JSON parse error. Raw response preview:\n", preview)
            raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc
    if isinstance(data, list):
        return data, []
    if isinstance(data, dict):
        segments = data.get("segments")
        if not isinstance(segments, list):
            raise ValueError("Gemini response must include a 'segments' array.")
        speaker_profiles = _normalize_speaker_profiles(data.get("speakers"))
        return segments, speaker_profiles
    raise ValueError("Gemini response must be a JSON object or array.")


def _extract_text_from_gemini_response(response: Any) -> str:
    if response is None:
        return ""

    for attr in ("text", "output_text", "output_texts"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            concatenated = "\n".join(str(item) for item in value if item)
            if concatenated.strip():
                return concatenated.strip()

    parts_text: List[str] = []

    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())
            if isinstance(candidate, dict):
                for part in candidate.get("content", {}).get("parts", []):
                    part_text = part.get("text")
                    if isinstance(part_text, str) and part_text.strip():
                        parts_text.append(part_text.strip())

    if not parts_text:
        contents = getattr(response, "contents", None)
        if contents:
            for content in contents:
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())
                elif isinstance(content, dict):
                    for part in content.get("parts", []):
                        part_text = part.get("text")
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())

    return "\n".join(parts_text).strip()


def _parse_timestamp_to_ms(timestamp_value: Any) -> Optional[int]:
    if timestamp_value is None:
        return None
    if isinstance(timestamp_value, (int, float)):
        # Treat integer millisecond values separately
        if isinstance(timestamp_value, int) and timestamp_value >= 1000:
            return max(0, int(timestamp_value))
        seconds = float(timestamp_value)
        return max(0, int(round(seconds * 1000)))
    value = str(timestamp_value).strip()
    if not value:
        return None
    value = value.replace(",", ".")
    # Try simple numeric parse
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        seconds = float(value)
        return max(0, int(seconds * 1000))
    # Handle colon-delimited formats (e.g., hh:mm:ss.xxx or mm:ss)
    parts = value.split(":")
    try:
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return None
        total_seconds = 0.0
        multiplier = 1.0
        for part in reversed(parts):
            if "." in part:
                seconds = float(part)
            else:
                seconds = float(int(part))
            total_seconds += seconds * multiplier
            multiplier *= 60.0
        return max(0, int(round(total_seconds * 1000)))
    except ValueError:
        return None


def _format_ms_to_timestamp(milliseconds: int) -> str:
    ms = max(0, int(milliseconds))
    total_seconds = ms / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    if abs(seconds - round(seconds)) < 1e-3:
        seconds_str = f"{int(round(seconds)):02d}"
    else:
        seconds_str = f"{seconds:06.3f}".rstrip("0").rstrip(".")
        if seconds < 10 and not seconds_str.startswith("0"):
            seconds_str = "0" + seconds_str
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_str}"
    return f"{minutes:02d}:{seconds_str}"


def _coerce_segment_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parsed = _parse_timestamp_to_ms(stripped)
        if parsed is not None:
            return parsed
        try:
            numeric = float(stripped)
            return max(0, int(numeric))
        except ValueError:
            return None
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return None


def _format_ms_to_srt_timestamp(milliseconds: int) -> str:
    ms = max(0, int(milliseconds))
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    seconds = (ms % 60_000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _parse_srt_timestamp_to_ms(timestamp: str) -> int:
    """
    Parse SRT timestamp to milliseconds.
    Supports formats like:
      - 00:01:23,456 (SRT standard)
      - 00:01:23.456 (VTT style)
      - 01:23,456 (mm:ss,ms)
      - 01:23.456 (mm:ss.ms)
    """
    timestamp = timestamp.strip()
    # Replace comma with period for uniform parsing
    timestamp = timestamp.replace(",", ".")
    
    parts = timestamp.split(":")
    if len(parts) == 3:
        # HH:MM:SS.mmm
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_ms = parts[2].split(".")
        seconds = int(sec_ms[0])
        millis = int(sec_ms[1].ljust(3, "0")[:3]) if len(sec_ms) > 1 else 0
    elif len(parts) == 2:
        # MM:SS.mmm
        hours = 0
        minutes = int(parts[0])
        sec_ms = parts[1].split(".")
        seconds = int(sec_ms[0])
        millis = int(sec_ms[1].ljust(3, "0")[:3]) if len(sec_ms) > 1 else 0
    else:
        return 0
    
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


def _parse_srt_file_with_timestamps(srt_content: str) -> List[Dict[str, Any]]:
    """
    Parse SRT/VTT file content and return list of entries with timestamps and text.
    Each entry has: sequence (sequence number), start_ms, end_ms, text
    
    Supports:
    - Standard SRT format
    - WebVTT format (with WEBVTT header)
    """
    entries: List[Dict[str, Any]] = []
    lines = srt_content.strip().split("\n")
    
    i = 0
    current_sequence: Optional[int] = None
    auto_sequence = 0  # Fallback counter if no sequence numbers
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Skip VTT header and metadata
        if line.upper().startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:") or line.startswith("NOTE"):
            i += 1
            continue
        
        # Check for sequence number (just a digit)
        if line.isdigit():
            current_sequence = int(line)
            i += 1
            continue
        
        # Check for timestamp line (contains "-->")
        if "-->" in line:
            # Parse timestamps
            timestamp_parts = line.split("-->")
            if len(timestamp_parts) == 2:
                start_ts = timestamp_parts[0].strip()
                end_ts = timestamp_parts[1].strip()
                # Remove any position info after the timestamp
                if " " in end_ts:
                    end_ts = end_ts.split(" ")[0]
                
                start_ms = _parse_srt_timestamp_to_ms(start_ts)
                end_ms = _parse_srt_timestamp_to_ms(end_ts)
                
                # Collect text lines until empty line or next sequence number
                i += 1
                text_lines: List[str] = []
                while i < len(lines):
                    text_line = lines[i].strip()
                    if not text_line:
                        i += 1
                        break
                    # Check if this looks like a sequence number for next entry
                    if text_line.isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
                        break
                    text_lines.append(text_line)
                    i += 1
                
                text = "\n".join(text_lines).strip()
                if text:
                    auto_sequence += 1
                    entries.append({
                        "sequence": current_sequence if current_sequence is not None else auto_sequence,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": text,
                    })
                    current_sequence = None  # Reset for next entry
                continue
        
        i += 1
    
    return entries


def _combine_srt_subtitles_to_segments(
    original_srt_entries: List[Dict[str, Any]],
    translated_srt_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Combine original language SRT and translated SRT into segment format compatible with generation.
    
    Matching is done by sequence number (1, 2, 3, etc.) - the standard SRT numbering.
    Timestamps are taken from the original SRT file.
    
    Since subtitles don't have speaker information, all segments are assigned to 'speaker1'.
    
    Returns tuple of (segments_data, speaker_profiles).
    """
    # Use original SRT timestamps as primary
    segments: List[Dict[str, Any]] = []
    
    # Build a lookup for translated entries by sequence number
    translated_lookup: Dict[int, Dict[str, Any]] = {}
    for entry in translated_srt_entries:
        seq = entry.get("sequence")
        if seq is not None:
            translated_lookup[seq] = entry
    
    for orig_entry in original_srt_entries:
        start_ms = orig_entry["start_ms"]
        end_ms = orig_entry["end_ms"]
        source_text = orig_entry["text"]
        sequence = orig_entry.get("sequence")
        
        # Find matching translated entry by sequence number
        translated_text = ""
        if sequence is not None and sequence in translated_lookup:
            translated_text = translated_lookup[sequence].get("text", "")
        
        # Format timestamps in mm:ss.xxx format for compatibility
        start_formatted = _format_ms_to_timestamp(start_ms)
        end_formatted = _format_ms_to_timestamp(end_ms)
        
        segments.append({
            "start": start_formatted,
            "end": end_formatted,
            "source_text": source_text,
            "translated_text": translated_text,
            "speaker": "speaker1",  # Subtitles don't have speaker info
        })
    
    # Create single speaker profile
    speaker_profiles = [
        {
            "id": "speaker1",
            "description": "Single speaker (from subtitle)",
        }
    ]
    
    return segments, speaker_profiles


def _combine_srt_translated_only_to_segments(
    translated_srt_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create segments from translated SRT only (when no original SRT is provided).
    Source text will be empty.
    
    Returns tuple of (segments_data, speaker_profiles).
    """
    segments: List[Dict[str, Any]] = []
    
    for entry in translated_srt_entries:
        start_ms = entry["start_ms"]
        end_ms = entry["end_ms"]
        translated_text = entry["text"]
        
        start_formatted = _format_ms_to_timestamp(start_ms)
        end_formatted = _format_ms_to_timestamp(end_ms)
        
        segments.append({
            "start": start_formatted,
            "end": end_formatted,
            "source_text": "",
            "translated_text": translated_text,
            "speaker": "speaker1",
        })
    
    speaker_profiles = [
        {
            "id": "speaker1",
            "description": "Single speaker (from subtitle)",
        }
    ]
    
    return segments, speaker_profiles


def _combine_srt_original_only_to_segments(
    original_srt_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create segments from original SRT only (for transcription mode without translation).
    Translated text will be the same as source text.
    
    Returns tuple of (segments_data, speaker_profiles).
    """
    segments: List[Dict[str, Any]] = []
    
    for entry in original_srt_entries:
        start_ms = entry["start_ms"]
        end_ms = entry["end_ms"]
        source_text = entry["text"]
        
        start_formatted = _format_ms_to_timestamp(start_ms)
        end_formatted = _format_ms_to_timestamp(end_ms)
        
        segments.append({
            "start": start_formatted,
            "end": end_formatted,
            "source_text": source_text,
            "translated_text": source_text,  # Use source as translated for generation
            "speaker": "speaker1",
        })
    
    speaker_profiles = [
        {
            "id": "speaker1",
            "description": "Single speaker (from subtitle)",
        }
    ]
    
    return segments, speaker_profiles


def _parse_srt_input_to_segments(
    original_srt_content: Optional[str],
    translated_srt_content: Optional[str],
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Parse SRT subtitle inputs and convert to segment format.
    
    Args:
        original_srt_content: Content of original language SRT file (optional)
        translated_srt_content: Content of translated SRT file (optional)
    
    Returns:
        Tuple of (segments, speaker_profiles) or None if no valid SRT content provided.
    
    Raises:
        ValueError: If SRT parsing fails.
    """
    original_entries: List[Dict[str, Any]] = []
    translated_entries: List[Dict[str, Any]] = []
    
    if original_srt_content:
        original_srt_str = original_srt_content.strip()
        if original_srt_str:
            original_entries = _parse_srt_file_with_timestamps(original_srt_str)
    
    if translated_srt_content:
        translated_srt_str = translated_srt_content.strip()
        if translated_srt_str:
            translated_entries = _parse_srt_file_with_timestamps(translated_srt_str)
    
    # Return None if no SRT content provided
    if not original_entries and not translated_entries:
        return None
    
    # Combine based on what's available
    if original_entries and translated_entries:
        return _combine_srt_subtitles_to_segments(original_entries, translated_entries)
    elif translated_entries:
        return _combine_srt_translated_only_to_segments(translated_entries)
    elif original_entries:
        return _combine_srt_original_only_to_segments(original_entries)
    
    return None


async def _read_srt_upload_text(
    upload_file: Optional[UploadFile],
    *,
    label: str,
    log_prefix: str,
) -> Optional[str]:
    if upload_file is None:
        return None

    print(f"[{log_prefix}] Reading {label} SRT: {upload_file.filename}")
    srt_bytes = await upload_file.read()
    if not srt_bytes:
        return None

    try:
        content = srt_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = srt_bytes.decode("latin-1")

    print(f"[{log_prefix}] {label.capitalize()} SRT loaded: {len(content)} bytes")
    return content


async def _load_srt_segments_override(
    original_srt_file: Optional[UploadFile],
    translated_srt_file: Optional[UploadFile],
    *,
    log_prefix: str,
) -> Tuple[Optional[str], Optional[JSONResponse]]:
    if original_srt_file is None and translated_srt_file is None:
        return None, None

    print(f"[{log_prefix}] Processing SRT subtitle files...")
    srt_parse_start = time.perf_counter()

    try:
        original_srt_content = await _read_srt_upload_text(
            original_srt_file,
            label="original",
            log_prefix=log_prefix,
        )
        translated_srt_content = await _read_srt_upload_text(
            translated_srt_file,
            label="translated",
            log_prefix=log_prefix,
        )

        print(f"[{log_prefix}] Parsing SRT files into segments...")
        srt_result = _parse_srt_input_to_segments(original_srt_content, translated_srt_content)
        if srt_result is None:
            return None, None

        srt_segments, srt_speaker_profiles = srt_result
        srt_parse_elapsed = (time.perf_counter() - srt_parse_start) * 1000
        print(f"[{log_prefix}] SRT parsing complete: {len(srt_segments)} segments in {srt_parse_elapsed:.1f}ms")

        return json.dumps({
            "speakers": srt_speaker_profiles,
            "segments": srt_segments,
        }, ensure_ascii=False), None
    except Exception as srt_exc:
        return None, _status_error(f"Failed to parse SRT subtitle files: {str(srt_exc)}")


def _build_srt_entries_from_segments(
    segments: Optional[List[Dict[str, Any]]],
    text_kind: Literal["translated", "source"] = "translated",
) -> List[Tuple[int, int, str]]:
    entries: List[Tuple[int, int, str]] = []
    if not segments:
        return entries

    def _segment_sort_key(segment: Dict[str, Any]) -> Tuple[int, int]:
        start_ms = _coerce_segment_ms(segment.get("start_ms")) or _coerce_segment_ms(segment.get("start")) or 0
        index = segment.get("index")
        return start_ms, int(index) if isinstance(index, int) else 0

    for segment in sorted(segments, key=_segment_sort_key):
        seg_type = str(segment.get("type") or "speech").lower()
        if seg_type != "speech":
            continue
        start_ms = _coerce_segment_ms(segment.get("start_ms")) or _coerce_segment_ms(segment.get("start")) or 0
        end_ms = _coerce_segment_ms(segment.get("end_ms")) or _coerce_segment_ms(segment.get("end"))
        if end_ms is None:
            duration_ms = _coerce_segment_ms(segment.get("duration_ms")) or 0
            end_ms = start_ms + (duration_ms if duration_ms > 0 else 1000)
        end_ms = max(end_ms, start_ms + 1)
        if text_kind == "source":
            text_value = (segment.get("source_text") or "").strip()
        else:
            text_value = (segment.get("translated_text") or segment.get("source_text") or "").strip()
        if not text_value:
            continue
        sanitized_text = text_value.replace("\r\n", "\n").replace("\r", "\n")
        entries.append((start_ms, end_ms, sanitized_text))
    return entries


def _offset_segments_for_merge(
    segments: Optional[List[Dict[str, Any]]],
    offset_ms: int,
    *,
    max_duration_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    adjusted: List[Dict[str, Any]] = []
    if not segments:
        return adjusted

    for segment in segments:
        seg_type = str(segment.get("type") or "speech").lower()
        if seg_type != "speech":
            continue
        local_start = _coerce_segment_ms(segment.get("start_ms")) or _coerce_segment_ms(segment.get("start")) or 0
        local_end = _coerce_segment_ms(segment.get("end_ms")) or _coerce_segment_ms(segment.get("end"))
        if local_end is None:
            duration_ms = _coerce_segment_ms(segment.get("duration_ms")) or 0
            local_end = local_start + (duration_ms if duration_ms > 0 else 1000)
        if max_duration_ms is not None:
            limit = max(0, int(max_duration_ms))
            local_start = min(local_start, limit)
            local_end = min(local_end, limit)
        if local_end <= local_start:
            local_end = local_start + 1
        adjusted.append(
            {
                "type": "speech",
                "start_ms": offset_ms + local_start,
                "end_ms": offset_ms + local_end,
                "translated_text": segment.get("translated_text"),
                "source_text": segment.get("source_text"),
            }
        )
    return adjusted


def _export_srt_from_segments(
    segments: Optional[List[Dict[str, Any]]],
    *,
    base_name: str,
    suffix: Optional[str] = None,
    text_kind: Literal["translated", "source"] = "translated",
    empty_note: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        srt_entries = _build_srt_entries_from_segments(segments, text_kind=text_kind)
        safe_base = _sanitize_base_filename(base_name) or DEFAULT_BASE_FILENAME
        srt_stem = safe_base
        if suffix:
            srt_stem = f"{safe_base}_{suffix}"
        srt_filename = f"{srt_stem}.srt"
        srt_path = os.path.join(TRANSLATE_OUTPUT_DIR, srt_filename)
        os.makedirs(TRANSLATE_OUTPUT_DIR, exist_ok=True)
        lines: List[str] = []
        if srt_entries:
            for idx, (start_ms, end_ms, text_value) in enumerate(srt_entries, start=1):
                lines.append(str(idx))
                lines.append(f"{_format_ms_to_srt_timestamp(start_ms)} --> {_format_ms_to_srt_timestamp(end_ms)}")
                lines.append(text_value)
                lines.append("")
        else:
            placeholder = empty_note or "No speech segments were available for subtitle export."
            lines.extend(
                [
                    "1",
                    "00:00:00,000 --> 00:00:01,000",
                    placeholder,
                    "",
                ]
            )
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            srt_file.write("\n".join(lines).strip() + "\n")
        return {
            "format": "srt",
            "filename": srt_filename,
            "url": f"/api/translate_outputs/{srt_filename}",
            "entry_count": len(srt_entries),
        }
    except Exception as exc:
        print(f"⚠️ Failed to export SRT subtitles: {exc}")
        return None


def _find_case_insensitive_key(data: Dict[str, Any], candidates: List[str], ignore: Set[str]) -> Optional[str]:
    lowered = {str(k).lower(): k for k in data.keys()}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lowered:
            if candidate_lower in ignore:
                continue
            return lowered[candidate_lower]
    return None


def _extract_text_fields(entry: Dict[str, Any], dest_language: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    ignore_keys = {
        "start",
        "end",
        "start_time",
        "end_time",
        "start_ms",
        "end_ms",
        "duration",
        "duration_ms",
        "timestamp",
        "time",
        "index",
        "type",
        "speaker",
        "language",
        "source_language",
        "target_language",
    }
    lower_dest = dest_language.lower() if dest_language else ""
    translation_candidates = [
        "translated_text",
        "translation",
        "translation_text",
        "target_text",
        "target",
        "output",
        "output_text",
        "translation_result",
        "english",
        "en",
    ]
    if lower_dest:
        translation_candidates.extend([
            lower_dest,
            lower_dest.replace("-", "_"),
            lower_dest.replace(" ", "_"),
        ])
    source_candidates = [
        "source_text",
        "source",
        "original_text",
        "original",
        "transcript",
        "transcription",
        "input_text",
        "utterance",
        "text",
        "chinese",
        "zh",
        "cn",
    ]
    translation_key = _find_case_insensitive_key(entry, translation_candidates, ignore_keys)
    source_key = _find_case_insensitive_key(entry, source_candidates, ignore_keys)

    def _fallback_key(preferred_key: Optional[str]) -> Optional[str]:
        if preferred_key is not None:
            return preferred_key
        for key, value in entry.items():
            key_lower = str(key).lower()
            if key_lower in ignore_keys:
                continue
            if isinstance(value, str) and value.strip():
                return key
        return None

    source_key = _fallback_key(source_key)
    translation_key = _fallback_key(translation_key)

    source_text = entry.get(source_key) if source_key else None
    translated_text = entry.get(translation_key) if translation_key else None
    return source_text, translated_text, source_key, translation_key


def _coerce_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_positive_int(
    value: Any,
    default: int,
    *,
    min_value: int = 0,
    max_value: Optional[int] = None,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return min_value
    if max_value is not None and parsed > max_value:
        return max_value
    return parsed


def _coerce_clearvoice_parallel_config(
    enabled_value: Any,
    chunk_seconds_value: Any,
    max_workers_value: Any,
) -> ClearVoiceParallelConfig:
    chunk_seconds = _coerce_positive_int(
        chunk_seconds_value,
        CLEARVOICE_PARALLEL_DEFAULT_CHUNK_SECONDS,
        min_value=CLEARVOICE_PARALLEL_MIN_CHUNK_SECONDS,
        max_value=CLEARVOICE_PARALLEL_MAX_CHUNK_SECONDS,
    )
    max_workers = _coerce_positive_int(
        max_workers_value,
        CLEARVOICE_PARALLEL_DEFAULT_WORKERS,
        min_value=1,
        max_value=CLEARVOICE_PARALLEL_MAX_WORKERS,
    )
    enabled = _coerce_to_bool(enabled_value)
    return ClearVoiceParallelConfig(
        enabled=enabled,
        chunk_seconds=chunk_seconds,
        max_workers=max_workers,
    )


def _resolve_audio_preprocess_options(
    *,
    enhance_voice: Any,
    super_resolution_voice: Any,
    enhancement_model: Optional[str],
    audio_separator_enabled: Any = False,
    audio_separator_model: Optional[str] = None,
    audio_separator_use_soundfile: Any = None,
    clearvoice_parallel_enabled: Any = False,
    clearvoice_parallel_chunk_seconds: Any = None,
    clearvoice_parallel_max_workers: Any = None,
    force_enhancement_for_super_resolution: bool = False,
) -> AudioPreprocessOptions:
    apply_enhancement = _coerce_to_bool(enhance_voice)
    apply_super_resolution = _coerce_to_bool(super_resolution_voice)
    if force_enhancement_for_super_resolution and apply_super_resolution:
        apply_enhancement = True

    parallel_config = _coerce_clearvoice_parallel_config(
        clearvoice_parallel_enabled,
        clearvoice_parallel_chunk_seconds,
        clearvoice_parallel_max_workers,
    )
    if not (apply_enhancement or apply_super_resolution):
        parallel_config.enabled = False

    audio_separator_enabled_flag = _coerce_to_bool(audio_separator_enabled or False)
    audio_separator_use_soundfile_flag = (
        AUDIO_SEPARATOR_USE_SOUNDFILE
        if audio_separator_use_soundfile is None
        else _coerce_to_bool(audio_separator_use_soundfile)
    )
    audio_separator_model_key = (audio_separator_model or "").strip().lower()
    if audio_separator_model_key not in AUDIO_SEPARATOR_MODELS:
        audio_separator_model_key = DEFAULT_AUDIO_SEPARATOR_MODEL

    enhancement_model_name = (
        enhancement_model if enhancement_model in AVAILABLE_ENHANCEMENT_MODELS else DEFAULT_ENHANCEMENT_MODEL
    )

    return AudioPreprocessOptions(
        apply_enhancement=apply_enhancement,
        apply_super_resolution=apply_super_resolution,
        enhancement_model_name=enhancement_model_name,
        audio_separator_enabled=audio_separator_enabled_flag,
        audio_separator_model=audio_separator_model_key,
        audio_separator_use_soundfile=audio_separator_use_soundfile_flag,
        parallel_config=parallel_config,
    )


def _parallel_config_from_settings(settings: Optional[Dict[str, Any]]) -> Optional[ClearVoiceParallelConfig]:
    if not settings:
        return None
    if not _coerce_to_bool(settings.get("parallel_enabled")):
        return None
    chunk_seconds = _coerce_positive_int(
        settings.get("parallel_chunk_seconds"),
        CLEARVOICE_PARALLEL_DEFAULT_CHUNK_SECONDS,
        min_value=CLEARVOICE_PARALLEL_MIN_CHUNK_SECONDS,
        max_value=CLEARVOICE_PARALLEL_MAX_CHUNK_SECONDS,
    )
    max_workers = _coerce_positive_int(
        settings.get("parallel_max_workers"),
        CLEARVOICE_PARALLEL_DEFAULT_WORKERS,
        min_value=1,
        max_value=CLEARVOICE_PARALLEL_MAX_WORKERS,
    )
    return ClearVoiceParallelConfig(
        enabled=True,
        chunk_seconds=chunk_seconds,
        max_workers=max_workers,
    )


def _coerce_positive_float(
    value: Any,
    default: float,
    *,
    min_value: float = 0.0,
    max_value: Optional[float] = None,
) -> float:
    if value is None:
        return max(default, min_value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return max(default, min_value)
    if not math.isfinite(parsed):
        return max(default, min_value)
    parsed = max(parsed, min_value)
    if max_value is not None:
        parsed = min(parsed, max_value)
    return parsed


def _resolve_qwen_omnivad_diarization_options(
    transcription_pipeline: Any,
    enable_diarization: Any,
    diarization_min_seconds: Any,
) -> Tuple[bool, float]:
    pipeline = (str(transcription_pipeline or "")).strip().lower()
    enabled = (
        _coerce_to_bool(enable_diarization if enable_diarization is not None else True)
        if pipeline == "qwen_omnivad"
        else False
    )
    min_seconds = _coerce_positive_float(
        diarization_min_seconds if diarization_min_seconds is not None else 0.0,
        0.0,
        min_value=0.0,
    )
    return enabled, min_seconds


def _resolve_qwen_omnivad_diarization_backend_option(
    transcription_pipeline: Any,
    diarization_backend: Any,
) -> str:
    pipeline = (str(transcription_pipeline or "")).strip().lower()
    if pipeline != "qwen_omnivad":
        return "auto"
    value = str(diarization_backend or "auto").strip().lower().replace("-", "_")
    aliases = {
        "pyannote_audio": "pyannote",
        "nvidia_sortformer": "sortformer",
        "sortformer_4spk": "sortformer",
        "sortformer_v2_1": "sortformer",
        "sortformer_v2.1": "sortformer",
    }
    value = aliases.get(value, value)
    return value if value in {"auto", "pyannote", "sortformer"} else "auto"


def _resolve_qwen_omnivad_forced_aligner_option(
    transcription_pipeline: Any,
    enable_forced_aligner: Any,
) -> bool:
    pipeline = (str(transcription_pipeline or "")).strip().lower()
    if pipeline != "qwen_omnivad":
        return False
    return _coerce_to_bool(
        enable_forced_aligner if enable_forced_aligner is not None else True
    )


def _resolve_qwen_omnivad_merge_gap_seconds_option(
    transcription_pipeline: Any,
    merge_gap_seconds: Any,
) -> float:
    pipeline = (str(transcription_pipeline or "")).strip().lower()
    default_value = DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS
    if pipeline != "qwen_omnivad":
        return default_value
    return _coerce_positive_float(
        merge_gap_seconds if merge_gap_seconds is not None else default_value,
        default_value,
        min_value=0.0,
    )


def _coerce_float_range(
    value: Any,
    default: float,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if min_value is not None and parsed < min_value:
        parsed = min_value
    if max_value is not None and parsed > max_value:
        parsed = max_value
    return parsed


def _coerce_volume_percent(
    value: Any,
    default: float = DEFAULT_GENERATED_VOLUME_PERCENT,
) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if parsed < MIN_GENERATED_VOLUME_PERCENT:
        return MIN_GENERATED_VOLUME_PERCENT
    if parsed > MAX_GENERATED_VOLUME_PERCENT:
        return MAX_GENERATED_VOLUME_PERCENT
    return parsed


def _resolve_volume_options(
    generated_value: Any,
    backing_value: Any,
    silence_value: Any,
    *,
    generated_default: float = DEFAULT_GENERATED_VOLUME_PERCENT,
    backing_default: float = DEFAULT_GENERATED_VOLUME_PERCENT,
    silence_default: float = DEFAULT_SILENCE_VOLUME_PERCENT,
) -> VolumeOptions:
    return VolumeOptions(
        generated=_coerce_volume_percent(generated_value, generated_default),
        backing=_coerce_volume_percent(backing_value, backing_default),
        silence=_coerce_volume_percent(silence_value, silence_default),
    )


def _coerce_emotion_weight(value: Any, default: float = DEFAULT_EMOTION_WEIGHT) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _parse_manual_segments_input(
    raw: Optional[str],
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Parse user-supplied Gemini segment JSON.
    Accepts either a JSON array of segment objects, an object with "segments" and
    optional "speakers", or a list of such objects (common transcript export shape:
    [{ "speakers": [...], "segments": [...] }]).
    """
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if not raw_str:
        return None
    try:
        parsed = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"segments_json is not valid JSON: {exc}") from exc

    speaker_profiles: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        speaker_profiles = _normalize_speaker_profiles(parsed.get("speakers"))
        parsed = parsed.get("segments")
    elif isinstance(parsed, list) and parsed:
        # Unwrap [{ "speakers": [...], "segments": [...] }, ...] — not a flat segment list
        if all(
            isinstance(item, dict) and isinstance(item.get("segments"), list)
            for item in parsed
        ):
            merged_speakers_raw: List[Any] = []
            merged_segments: List[Dict[str, Any]] = []
            for item in parsed:
                sp = item.get("speakers")
                if isinstance(sp, list):
                    merged_speakers_raw.extend(sp)
                merged_segments.extend(item["segments"])
            speaker_profiles = _normalize_speaker_profiles(merged_speakers_raw)
            parsed = merged_segments

    if not isinstance(parsed, list):
        raise ValueError("segments_json must be a JSON array of segment objects.")
    if not parsed:
        raise ValueError("segments_json array is empty.")

    cleaned: List[Dict[str, Any]] = []
    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise ValueError(f"segments_json[{idx}] is not an object.")
        cleaned.append(entry)
    return cleaned, speaker_profiles


def _merge_short_speech_segments(
    segments: List[Dict[str, Any]],
    min_duration_ms: int,
    max_merge_interval_ms: int = MAX_MERGE_INTERVAL_MS,
) -> List[Dict[str, Any]]:
    max_merge_interval_ms = max(0, int(max_merge_interval_ms))
    if max_merge_interval_ms == 0:
        no_merge_segments: List[Dict[str, Any]] = []
        for idx, segment in enumerate(segments):
            clone = dict(segment)
            clone["index"] = idx
            no_merge_segments.append(clone)
        return no_merge_segments
    merged_segments: List[Dict[str, Any]] = []
    i = 0
    total = len(segments)

    def _concat_text(a: Optional[str], b: Optional[str]) -> str:
        parts = [part.strip() for part in [a or "", b or ""] if part and part.strip()]
        return " ".join(parts)

    while i < total:
        segment = segments[i]
        if segment.get("type") != "speech":
            merged_segments.append(segment)
            i += 1
            continue

        start_ms = int(segment.get("start_ms", 0))
        end_ms = int(segment.get("end_ms", start_ms))
        source_text = segment.get("source_text", "")
        translated_text = segment.get("translated_text", "")
        raw_chunks: List[Any] = [segment.get("raw_chunk")]
        raw_indices: List[Any] = [segment.get("raw_chunk_index")]
        current_speaker = str(segment.get("speaker") or "").strip().lower() or None

        j = i + 1

        # Merge forward with following silence/speech until threshold reached
        # Only merge if silence intervals are short (<= max_merge_interval_ms)
        while end_ms - start_ms < min_duration_ms and j < total:
            next_segment = segments[j]
            seg_type = next_segment.get("type")
            if seg_type == "silence":
                silence_duration = int(next_segment.get("duration_ms", 0))
                # Only merge if silence is short enough
                if silence_duration <= max_merge_interval_ms:
                    end_ms = int(next_segment.get("end_ms", end_ms))
                    j += 1
                    continue
                else:
                    # Silence gap too long, stop merging forward
                    break
            if seg_type == "speech":
                next_speaker = str(next_segment.get("speaker") or "").strip().lower() or None
                if current_speaker and next_speaker and next_speaker != current_speaker:
                    break
                end_ms = int(next_segment.get("end_ms", end_ms))
                source_text = _concat_text(source_text, next_segment.get("source_text"))
                translated_text = _concat_text(translated_text, next_segment.get("translated_text"))
                raw_chunks.append(next_segment.get("raw_chunk"))
                raw_indices.append(next_segment.get("raw_chunk_index"))
                j += 1
                continue
            break

        duration_ms = end_ms - start_ms

        if duration_ms < min_duration_ms and merged_segments:
            # Merge backward with previous speech if forward merge insufficient
            # Find the last speech segment in merged_segments
            prev_speech_idx = len(merged_segments) - 1
            while prev_speech_idx >= 0 and merged_segments[prev_speech_idx].get("type") != "speech":
                prev_speech_idx -= 1
            
            if prev_speech_idx >= 0:
                prev_segment = merged_segments[prev_speech_idx]
                prev_end_ms = int(prev_segment.get("end_ms", 0))
                gap_ms = start_ms - prev_end_ms
                # Only merge backward if gap is short enough
                prev_speaker = str(prev_segment.get("speaker") or "").strip().lower() or None
                if gap_ms <= max_merge_interval_ms and (
                    not current_speaker
                    or not prev_speaker
                    or prev_speaker == current_speaker
                ):
                    prev_segment["end_ms"] = end_ms
                    prev_segment["duration_ms"] = end_ms - int(prev_segment.get("start_ms", start_ms))
                    prev_segment["end"] = _format_ms_to_timestamp(end_ms)
                    prev_segment["source_text"] = _concat_text(prev_segment.get("source_text"), source_text)
                    prev_segment["translated_text"] = _concat_text(prev_segment.get("translated_text"), translated_text)

                    prev_raw = prev_segment.get("raw_chunk")
                    if isinstance(prev_raw, list):
                        prev_raw.extend(raw_chunks)
                    elif prev_raw is None:
                        prev_segment["raw_chunk"] = raw_chunks
                    else:
                        prev_segment["raw_chunk"] = [prev_raw] + raw_chunks

                    prev_indices = prev_segment.get("raw_chunk_index")
                    raw_indices_clean = [idx for idx in raw_indices if idx is not None]
                    if raw_indices_clean:
                        if isinstance(prev_indices, list):
                            prev_indices.extend(raw_indices_clean)
                        elif prev_indices is None:
                            prev_segment["raw_chunk_index"] = raw_indices_clean
                        else:
                            prev_segment["raw_chunk_index"] = [prev_indices] + raw_indices_clean

                    i = j
                    continue

        new_segment = dict(segment)
        new_segment["start_ms"] = start_ms
        new_segment["end_ms"] = end_ms
        new_segment["duration_ms"] = duration_ms
        new_segment["start"] = _format_ms_to_timestamp(start_ms)
        new_segment["end"] = _format_ms_to_timestamp(end_ms)
        new_segment["source_text"] = source_text
        new_segment["translated_text"] = translated_text

        raw_chunks_clean = [chunk for chunk in raw_chunks if chunk is not None]
        if raw_chunks_clean:
            new_segment["raw_chunk"] = raw_chunks_clean if len(raw_chunks_clean) > 1 else raw_chunks_clean[0]
        raw_indices_clean = [idx for idx in raw_indices if idx is not None]
        if raw_indices_clean:
            new_segment["raw_chunk_index"] = raw_indices_clean if len(raw_indices_clean) > 1 else raw_indices_clean[0]

        merged_segments.append(new_segment)
        i = j

    # Reassign indices
    for idx, seg in enumerate(merged_segments):
        seg["index"] = idx

    return merged_segments


def _save_segment_audio_preview(
    session_id: str,
    segment_index: int,
    audio: AudioSegment,
    start_ms: int,
    end_ms: int,
    fmt: str = "mp3",
    bitrate: str = "128k",
) -> Optional[str]:
    """Save segment audio preview to file and return URL path."""
    start = max(0, int(start_ms))
    end = max(start, int(end_ms))
    audio_len = len(audio)
    if start >= audio_len or start == end:
        return None
    if end > audio_len:
        end = audio_len
    snippet = audio[start:end]
    if len(snippet) == 0:
        return None
    
    # Create filename for this segment preview
    filename = f"{session_id}_segment_{segment_index}.{fmt}"
    file_path = os.path.join(TRANSLATE_SESSION_MEDIA_DIR, filename)
    
    # Export audio to file
    export_kwargs: Dict[str, Any] = {}
    if fmt == "mp3":
        export_kwargs["bitrate"] = bitrate
    
    try:
        snippet.export(file_path, format=fmt, **export_kwargs)
        # Return URL path that will be served by the API endpoint
        return f"/api/segment_preview/{session_id}/{segment_index}"
    except Exception as exc:
        print(f"⚠️ Failed to save segment preview for session {session_id}, segment {segment_index}: {exc}")
        return None


def _save_segment_preview_from_path(
    session_id: str,
    segment_index: int,
    source_path: str,
    start_ms: int,
    end_ms: int,
    fmt: str = "mp3",
    bitrate: str = "128k",
) -> Optional[str]:
    """Save segment preview by extracting from source file using FFmpeg (fast, no full load)."""
    # Create filename for this segment preview
    filename = f"{session_id}_segment_{segment_index}.{fmt}"
    file_path = os.path.join(TRANSLATE_SESSION_MEDIA_DIR, filename)
    
    # Try FFmpeg first (fast, copy if possible)
    try:
        source_ext = os.path.splitext(source_path)[1].lower()
        fmt_ext = (fmt or "mp3").lower()
        # Copy without re-encode only when container matches to avoid slow transcodes
        reencode = not (source_ext == f".{fmt_ext}")
        _ffmpeg_extract_segment(
            source_path,
            file_path,
            start_ms,
            end_ms,
            reencode=reencode,
            bitrate=bitrate,
        )
        return f"/api/segment_preview/{session_id}/{segment_index}"
    except Exception as exc:
        print(f"[segment_preview] FFmpeg extraction failed for segment {segment_index}: {exc}")
        return None


def _serialize_segments_for_ui(
    segments: List[Dict[str, Any]],
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ui_segments: List[Dict[str, Any]] = []
    for segment in segments:
        seg_type = segment.get("type", "speech")
        start_ms = int(segment.get("start_ms", 0))
        end_ms = int(segment.get("end_ms", start_ms))
        duration_ms = int(segment.get("duration_ms", max(0, end_ms - start_ms)))
        segment_index = segment.get("index")
        base_payload = {
            "index": segment_index,
            "type": seg_type,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "source_text": segment.get("source_text", ""),
            "translated_text": segment.get("translated_text", ""),
            "text_keys": segment.get("text_keys", {}),
            "speaker": segment.get("speaker"),
            "volume_percent": segment.get("volume_percent"),
            "emotion_weight": segment.get("emotion_weight"),
        }
        if seg_type == "speech":
            base_payload["generate"] = True
            # Provide preview URL path only - don't generate audio until client requests it
            if session_id is not None and segment_index is not None:
                # Provide a dedicated URL for original-audio playback
                base_payload["audio_preview_url"] = (
                    f"/api/segment_preview/{session_id}/{segment_index}?variant=original"
                )
            # Don't generate base64 previews automatically - too memory intensive
        else:
            base_payload["generate"] = False
        ui_segments.append(base_payload)
    return ui_segments


def _cleanup_expired_translate_sessions_locked(now: Optional[float] = None) -> None:
    if now is None:
        now = time.time()
    expired: List[str] = [
        session_id
        for session_id, session in ADVANCED_TRANSLATE_SESSIONS.items()
        if now - session.created_at > ADVANCED_TRANSLATE_SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        ADVANCED_TRANSLATE_SESSIONS.pop(session_id, None)


async def _create_translate_session(
    original_audio: Optional[AudioSegment],
    dest_language: str,
    prompt: str,
    translate_enabled: bool,
    response_format: str,
    bitrate: str,
    input_mime_type: Optional[str],
    clearvoice_settings: Dict[str, Any],
    base_segments: List[Dict[str, Any]],
    gemini_chunks: List[Dict[str, Any]],
    gemini_model: str,
    gemini_api_key: Optional[str],
    translation_llm_model: Optional[str] = None,
    transcription_pipeline: str = "qwen_omnivad",
    whisperx_proxy_refiner: bool = False,
    qwen_omnivad_enable_diarization: Any = True,
    qwen_omnivad_diarization_backend: Any = "auto",
    qwen_omnivad_diarization_min_seconds: Any = 0.0,
    qwen_omnivad_enable_forced_aligner: Any = True,
    qwen_omnivad_merge_gap_seconds: Any = DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS,
    backing_track_audio: Optional[AudioSegment] = None,
    backing_track_path: Optional[str] = None,
    backing_audio_info: Optional[AudioAssetInfo] = None,
    backing_track_source: str = "none",
    merge_with_backing: bool = False,
    ignore_non_speech: bool = False,
    preserve_silence_audio: bool = False,
    generated_volume_percent: float = DEFAULT_GENERATED_VOLUME_PERCENT,
    backing_volume_percent: float = DEFAULT_GENERATED_VOLUME_PERCENT,
    silence_volume_percent: float = DEFAULT_SILENCE_VOLUME_PERCENT,
    speaker_profiles: Optional[List[Dict[str, Any]]] = None,
    speaker_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    gemini_raw_text: Optional[str] = None,
    chunk_parent_id: Optional[str] = None,
    chunk_index: Optional[int] = None,
    chunk_start_ms: Optional[int] = None,
    chunk_end_ms: Optional[int] = None,
    chunk_cut_reason: Optional[str] = None,
    chunk_silence_midpoint_ms: Optional[int] = None,
    persist_media: bool = True,
    media_format: str = "mp3",
    source_audio_filename: Optional[str] = None,
    source_base_name: Optional[str] = None,
    source_video_path: Optional[str] = None,
    source_video_filename: Optional[str] = None,
    default_speaker_preset: Optional[str] = None,
    default_emotion_weight: Optional[float] = None,
    original_audio_path: Optional[str] = None,
    original_audio_info: Optional[AudioAssetInfo] = None,
    session_id: Optional[str] = None,
) -> TranslateSessionData:
    response_format = _normalize_translate_output_format(response_format)
    session_id = session_id or uuid.uuid4().hex
    resolved_base_name = _normalize_base_filename(
        source_base_name,
        fallback=source_audio_filename,
    )
    normalized_default_speaker = (default_speaker_preset or "").strip() or None
    normalized_default_emotion = _coerce_emotion_weight(
        default_emotion_weight,
        DEFAULT_EMOTION_WEIGHT,
    )
    resolved_transcription_pipeline = _normalize_transcription_pipeline(transcription_pipeline)
    resolved_whisperx_proxy_refiner = (
        _coerce_to_bool(whisperx_proxy_refiner)
        and resolved_transcription_pipeline == "whisperx"
    )
    (
        resolved_qwen_omnivad_enable_diarization,
        resolved_qwen_omnivad_diarization_min_seconds,
    ) = _resolve_qwen_omnivad_diarization_options(
        resolved_transcription_pipeline,
        qwen_omnivad_enable_diarization,
        qwen_omnivad_diarization_min_seconds,
    )
    resolved_qwen_omnivad_diarization_backend = _resolve_qwen_omnivad_diarization_backend_option(
        resolved_transcription_pipeline,
        qwen_omnivad_diarization_backend,
    )
    resolved_qwen_omnivad_enable_forced_aligner = _resolve_qwen_omnivad_forced_aligner_option(
        resolved_transcription_pipeline,
        qwen_omnivad_enable_forced_aligner,
    )
    resolved_qwen_omnivad_merge_gap_seconds = _resolve_qwen_omnivad_merge_gap_seconds_option(
        resolved_transcription_pipeline,
        qwen_omnivad_merge_gap_seconds,
    )

    audio_path = original_audio_path
    audio_info = original_audio_info
    if not audio_path and original_audio is None:
        raise ValueError("Either original_audio or original_audio_path must be provided.")
    should_persist_audio = persist_media or not audio_path or (audio_path and not os.path.exists(audio_path))
    if should_persist_audio:
        target_format = media_format or "wav"
        # If the source is already an MP3 file on disk, reuse it directly to avoid re-encoding/copying.
        if original_audio_path and os.path.exists(original_audio_path) and original_audio_path.lower().endswith(".mp3"):
            print(f"[session] Reusing cached MP3 for vocals (no persistence): {original_audio_path}")
            audio_path = original_audio_path
            try:
                audio_info = _probe_audio_metadata_from_path(audio_path, compute_hash=False)
            except Exception as exc:
                print(f"⚠️ Failed to probe audio metadata for {audio_path}: {exc}")
                audio_info = None
            should_persist_audio = False
        if should_persist_audio:
            # OPTIMIZATION: If we have a source path, use FFmpeg transcode instead of load+export
            if original_audio_path and os.path.exists(original_audio_path):
                print(f"[session] Using fast path-based persistence for vocals (no memory loading)")
                audio_path, audio_info = await _persist_session_audio_from_path(
                    session_id, original_audio_path, "vocals", fmt=target_format
                )
            elif original_audio is not None:
                audio_path, audio_info = await _persist_session_audio_segment(session_id, original_audio, "vocals", fmt=target_format)
            else:
                raise ValueError("Original audio data is required to persist session media.")
    if audio_info is None and audio_path:
        try:
            audio_info = _probe_audio_metadata_from_path(audio_path, compute_hash=False)
        except Exception as exc:
            print(f"⚠️ Failed to probe audio metadata for {audio_path}: {exc}")
            audio_info = None

    backing_path = backing_track_path
    backing_info = backing_audio_info
    if backing_path is None and backing_track_audio is not None:
        # OPTIMIZATION: If we have a source backing path, use FFmpeg transcode
        backing_source_path = getattr(backing_track_audio, '_source_path', None)  # Check for attached path
        if backing_track_path and os.path.exists(backing_track_path):
            print(f"[session] Using fast path-based persistence for backing (no memory loading)")
            backing_path, backing_info = await _persist_session_audio_from_path(
                session_id, backing_track_path, "backing", fmt="mp3"
            )
        else:
            backing_path, backing_info = await _persist_session_audio_segment(session_id, backing_track_audio, "backing", fmt="mp3")
    elif backing_path and os.path.exists(backing_path) and backing_path.lower().endswith(".mp3"):
        # Reuse cached backing mp3 directly if present
        print(f"[session] Reusing cached backing MP3 (no persistence): {backing_path}")
        try:
            backing_info = _probe_audio_metadata_from_path(backing_path, compute_hash=False)
        except Exception as exc:
            print(f"⚠️ Failed to probe backing metadata for {backing_path}: {exc}")
            backing_info = None
    if backing_info is None and backing_path:
        try:
            backing_info = _probe_audio_metadata_from_path(backing_path, compute_hash=False)
        except Exception as exc:
            print(f"⚠️ Failed to probe backing track metadata for {backing_path}: {exc}")
            backing_info = None

    session = TranslateSessionData(
        session_id=session_id,
        dest_language=dest_language,
        prompt=prompt,
        translate_enabled=translate_enabled,
        response_format=response_format,
        bitrate=bitrate,
        input_mime_type=input_mime_type,
        clearvoice_settings=dict(clearvoice_settings or {}),
        base_segments=copy.deepcopy(base_segments),
        gemini_chunks=copy.deepcopy(gemini_chunks),
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        translation_llm_model=_normalize_translation_llm_model(translation_llm_model),
        transcription_pipeline=resolved_transcription_pipeline,
        whisperx_proxy_refiner=resolved_whisperx_proxy_refiner,
        qwen_omnivad_enable_diarization=resolved_qwen_omnivad_enable_diarization,
        qwen_omnivad_diarization_backend=resolved_qwen_omnivad_diarization_backend,
        qwen_omnivad_diarization_min_seconds=resolved_qwen_omnivad_diarization_min_seconds,
        qwen_omnivad_enable_forced_aligner=resolved_qwen_omnivad_enable_forced_aligner,
        qwen_omnivad_merge_gap_seconds=resolved_qwen_omnivad_merge_gap_seconds,
        source_audio_filename=source_audio_filename,
        source_base_name=resolved_base_name,
        source_video_path=source_video_path,
        source_video_filename=source_video_filename,
        original_audio_path=audio_path,
        original_audio_info=audio_info,
        backing_track_path=backing_path,
        backing_audio_info=backing_info,
        backing_track_source=backing_track_source or "none",
        merge_with_backing=merge_with_backing,
        ignore_non_speech=ignore_non_speech,
        preserve_silence_audio=preserve_silence_audio,
        generated_volume_percent=generated_volume_percent,
        backing_volume_percent=backing_volume_percent,
        silence_volume_percent=silence_volume_percent,
        speaker_profiles=copy.deepcopy(speaker_profiles or []),
        speaker_overrides=copy.deepcopy(speaker_overrides or {}),
        default_speaker_preset=normalized_default_speaker,
        default_emotion_weight=normalized_default_emotion,
        gemini_raw_text=gemini_raw_text,
        chunk_parent_id=chunk_parent_id,
        chunk_index=chunk_index,
        chunk_start_ms=chunk_start_ms,
        chunk_end_ms=chunk_end_ms,
        chunk_cut_reason=chunk_cut_reason,
        chunk_silence_midpoint_ms=chunk_silence_midpoint_ms,
    )
    session.original_audio = None
    session.backing_track_audio = None
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        _cleanup_expired_translate_sessions_locked()
        ADVANCED_TRANSLATE_SESSIONS[session_id] = session
    return session


async def _get_translate_session(session_id: str) -> Optional[TranslateSessionData]:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        _cleanup_expired_translate_sessions_locked()
        session = ADVANCED_TRANSLATE_SESSIONS.get(session_id)
        if session:
            session.created_at = time.time()
        return session


async def _resolve_reuse_translate_session(
    reuse_session_id: Optional[str],
) -> Tuple[Optional[str], Optional[TranslateSessionData], Optional[JSONResponse]]:
    normalized_session_id = (reuse_session_id or "").strip()
    if not normalized_session_id:
        return "", None, None

    session = await _get_translate_session(normalized_session_id)
    if session is None:
        return normalized_session_id, None, _status_error(
            "Reuse session not found or expired. Please re-upload the audio.",
            status_code=404,
        )
    return normalized_session_id, session, None


def _normalize_session_ids(raw_session_ids: Optional[List[Any]]) -> List[str]:
    return [sid.strip() for sid in (raw_session_ids or []) if isinstance(sid, str) and sid.strip()]


async def _resolve_chunk_sessions_by_ids(
    session_ids: List[str],
) -> Tuple[List[TranslateSessionData], Optional[JSONResponse]]:
    sessions: List[TranslateSessionData] = []
    for sid in session_ids:
        session = await _get_translate_session(sid)
        if session is None:
            return [], _status_error(f"Chunk session '{sid}' not found or expired.", status_code=404)
        if session.chunk_parent_id is None:
            return [], _status_error(f"Session '{sid}' is not a chunk session.")
        sessions.append(session)
    return sessions, None


async def _update_translate_session_segments(
    session_id: str, segments: List[Dict[str, Any]]
) -> None:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        session = ADVANCED_TRANSLATE_SESSIONS.get(session_id)
        if session:
            session.base_segments = copy.deepcopy(segments)
            session.created_at = time.time()
            _refresh_split_cache_for_session(session)


async def _update_translate_session_speaker_overrides(
    session_id: str, overrides: Dict[str, Dict[str, Any]]
) -> None:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        session = ADVANCED_TRANSLATE_SESSIONS.get(session_id)
        if session:
            session.speaker_overrides = copy.deepcopy(overrides)
            session.created_at = time.time()


async def _update_translate_session_metadata(
    session_id: str,
    *,
    response_format: Optional[str] = None,
    bitrate: Optional[str] = None,
    gemini_model: Optional[str] = None,
    generated_volume_percent: Optional[float] = None,
    backing_volume_percent: Optional[float] = None,
) -> None:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        session = ADVANCED_TRANSLATE_SESSIONS.get(session_id)
        if session:
            if response_format:
                session.response_format = _normalize_translate_output_format(response_format)
            if bitrate:
                session.bitrate = bitrate
            if gemini_model:
                session.gemini_model = gemini_model
            if generated_volume_percent is not None:
                session.generated_volume_percent = generated_volume_percent
        if backing_volume_percent is not None:
            session.backing_volume_percent = backing_volume_percent
            session.created_at = time.time()


async def _list_chunk_sessions(chunk_parent_id: str) -> List[TranslateSessionData]:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        _cleanup_expired_translate_sessions_locked()
        return [
            session
            for session in ADVANCED_TRANSLATE_SESSIONS.values()
            if session.chunk_parent_id == chunk_parent_id
        ]


def _guess_audio_format_from_mime(mime_type: Optional[str]) -> Optional[str]:
    if not mime_type:
        return None
    mime = mime_type.lower()
    mapping = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/wave": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/ogg; codecs=opus": "ogg",
        "audio/webm": "webm",
        "audio/opus": "opus",
        "audio/flac": "flac",
        "audio/aac": "aac",
        "audio/mp4": "mp4",
        "audio/m4a": "mp4",
        "audio/x-m4a": "mp4",
        "audio/vnd.wave": "wav",
    }
    return mapping.get(mime)


def _prepare_translation_segments(
    audio_duration_ms: int,
    chunk_data: List[Dict[str, Any]],
    dest_language: str,
    *,
    speaker_profiles: Optional[List[Dict[str, Any]]] = None,
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    max_merge_interval_ms: int = MAX_MERGE_INTERVAL_MS,
) -> List[Dict[str, Any]]:
    total_duration_ms = max(0, int(audio_duration_ms))
    segments: List[Dict[str, Any]] = []
    current_ms = 0
    normalized_profiles = speaker_profiles or []

    def _get_timestamp(entry: Dict[str, Any], keys: List[str]) -> Optional[int]:
        for key in keys:
            if key in entry:
                parsed = _parse_timestamp_to_ms(entry[key])
                if parsed is not None:
                    return parsed
        return None

    def _ensure_silence(duration_ms: int, position: str) -> Optional[Dict[str, Any]]:
        if duration_ms <= 0:
            return None
        start_ms = current_ms
        end_ms = current_ms + duration_ms
        segment_payload = {
            "type": "silence",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "start": _format_ms_to_timestamp(start_ms),
            "end": _format_ms_to_timestamp(end_ms),
            "position": position,
        }
        return segment_payload

    def _resolve_speaker_label(entry: Dict[str, Any], index: int) -> Optional[str]:
        fallback_index = index + 1
        candidate_keys = ["speaker", "speaker_id", "speaker_label", "voice", "speaker_name"]
        for key in candidate_keys:
            if key in entry and entry[key] not in (None, ""):
                return _canonicalize_speaker_id(entry[key], fallback_index).lower()
        speaker_idx = entry.get("speaker_index")
        if isinstance(speaker_idx, int):
            return f"speaker{max(1, speaker_idx + 1)}"
        if normalized_profiles and index < len(normalized_profiles):
            profile_id = normalized_profiles[index].get("id")
            if profile_id:
                return str(profile_id).strip().lower()
        return None

    if not chunk_data:
        raise ValueError("Gemini returned no transcription segments.")

    for idx, entry in enumerate(chunk_data):
        if not isinstance(entry, dict):
            continue

        start_ms = _get_timestamp(
            entry,
            [
                "start_ms",
                "startMilliseconds",
                "start_milliseconds",
                "start",
                "start_time",
                "begin",
                "from",
            ],
        )
        end_ms = _get_timestamp(
            entry,
            [
                "end_ms",
                "endMilliseconds",
                "end_milliseconds",
                "end",
                "end_time",
                "stop",
                "to",
            ],
        )

        if start_ms is None:
            start_ms = current_ms
        if end_ms is None:
            duration_ms = _get_timestamp(
                entry,
                ["duration_ms", "durationMilliseconds", "duration", "length", "segment_duration"],
            )
            if duration_ms is not None:
                end_ms = start_ms + duration_ms

        if end_ms is None:
            # Fallback: assume contiguous audio up to detected order
            # Use remainder of audio divided equally over remaining segments
            remaining_segments = max(len(chunk_data) - idx, 1)
            remaining_ms = max(total_duration_ms - start_ms, 0)
            avg_duration = remaining_ms // remaining_segments if remaining_segments else remaining_ms
            end_ms = start_ms + avg_duration

        start_ms = max(0, min(start_ms, total_duration_ms))
        end_ms = max(0, min(end_ms, total_duration_ms))
        if start_ms < current_ms:
            start_ms = current_ms
        if end_ms <= start_ms:
            continue

        # Add silence if there is a gap before this segment
        if start_ms > current_ms:
            silence_payload = _ensure_silence(start_ms - current_ms, "leading" if current_ms == 0 else "between")
            if silence_payload:
                silence_payload["index"] = len(segments)
                segments.append(silence_payload)
            current_ms = start_ms

        source_text, translated_text, source_key, translation_key = _extract_text_fields(entry, dest_language)
        if isinstance(source_text, str):
            source_text_value = source_text.strip()
        elif source_text is None:
            source_text_value = ""
        else:
            source_text_value = str(source_text)

        if isinstance(translated_text, str):
            translated_text_value = translated_text.strip()
        elif translated_text is None:
            translated_text_value = ""
        else:
            translated_text_value = str(translated_text)

        segment_payload: Dict[str, Any] = {
            "type": "speech",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "start": _format_ms_to_timestamp(start_ms),
            "end": _format_ms_to_timestamp(end_ms),
            "source_text": source_text_value,
            "translated_text": translated_text_value,
            "text_keys": {
                "source": source_key,
                "translated": translation_key,
            },
            "raw_chunk_index": idx,
            "raw_chunk": entry,
        }
        speaker_label = _resolve_speaker_label(entry, idx)
        if speaker_label:
            segment_payload["speaker"] = speaker_label
        segment_payload["index"] = len(segments)
        segments.append(segment_payload)
        current_ms = end_ms

    # Add trailing silence if needed
    if current_ms < total_duration_ms:
        remaining = total_duration_ms - current_ms
        silence_payload = _ensure_silence(remaining, "trailing")
        if silence_payload:
            silence_payload["index"] = len(segments)
            segments.append(silence_payload)

    segments = _merge_short_speech_segments(
        segments,
        max(0, int(min_speech_duration_ms)),
        max_merge_interval_ms=max(0, int(max_merge_interval_ms)),
    )

    return segments


def _create_silence_segment(duration_ms: int, frame_rate: int, sample_width: int, channels: int) -> AudioSegment:
    duration_ms = max(0, int(duration_ms))
    silence = AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)
    silence = silence.set_sample_width(sample_width)
    silence = silence.set_channels(channels)
    return silence


def _match_segment_duration(
    segment: AudioSegment,
    target_ms: int,
    frame_rate: int,
    sample_width: int,
    channels: int,
    *,
    loop_fill: bool = False,
    allow_trim: bool = False,
) -> AudioSegment:
    target_ms = max(0, int(target_ms))
    segment = segment.set_frame_rate(frame_rate)
    segment = segment.set_sample_width(sample_width)
    segment = segment.set_channels(channels)

    if target_ms == 0:
        return _create_silence_segment(0, frame_rate, sample_width, channels)

    current_ms = len(segment)
    tolerance_ms = 100
    diff = target_ms - current_ms
    if diff > 0:
        if loop_fill and current_ms > 0:
            filler = AudioSegment.silent(duration=0, frame_rate=frame_rate)
            filler = filler.set_sample_width(sample_width).set_channels(channels)
            extended = filler
            while len(extended) < target_ms:
                extended += segment
            if len(extended) > target_ms:
                extended = extended[:target_ms]
            return extended
        if diff <= tolerance_ms:
            return segment + _create_silence_segment(diff, frame_rate, sample_width, channels)
        return segment + _create_silence_segment(diff, frame_rate, sample_width, channels)
    if diff < 0 and allow_trim and target_ms >= 0:
        return segment[:target_ms]
    # Never trim segments by default; return original even if longer than target.
    return segment


def _apply_volume_with_ffmpeg(segment: AudioSegment, volume_percent: float) -> AudioSegment:
    """Adjust segment volume using ffmpeg when available, fallback to pydub gain."""
    try:
        volume_percent = float(volume_percent)
    except (TypeError, ValueError):
        volume_percent = DEFAULT_GENERATED_VOLUME_PERCENT
    if abs(volume_percent - DEFAULT_GENERATED_VOLUME_PERCENT) < 0.01:
        return segment

    volume_factor = max(volume_percent, MIN_GENERATED_VOLUME_PERCENT) / 100.0
    if shutil.which("ffmpeg") is None:
        gain_db = 20 * math.log10(volume_factor) if volume_factor > 0 else -120.0
        return segment.apply_gain(gain_db)

    input_path = None
    output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
            segment.export(tmp_input.name, format="wav")
            input_path = tmp_input.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_output:
            output_path = tmp_output.name

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-filter:a",
            f"volume={volume_factor}",
            output_path,
        ]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"⚠️ FFmpeg volume adjustment failed (code {process.returncode}): {process.stderr.strip()}")
            gain_db = 20 * math.log10(volume_factor) if volume_factor > 0 else -120.0
            return segment.apply_gain(gain_db)

        adjusted = AudioSegment.from_file(output_path, format="wav")
        adjusted = adjusted.set_frame_rate(segment.frame_rate)
        adjusted = adjusted.set_sample_width(segment.sample_width)
        adjusted = adjusted.set_channels(segment.channels)
        return adjusted
    except Exception as exc:
        print(f"⚠️ Volume adjustment fallback due to error: {exc}")
        gain_db = 20 * math.log10(volume_factor) if volume_factor > 0 else -120.0
        return segment.apply_gain(gain_db)
    finally:
        for path in (input_path, output_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


@functools.lru_cache(maxsize=1)
def _ffmpeg_available() -> bool:
    """Check once if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def _ffmpeg_seconds_from_ms(ms: int) -> str:
    return f"{max(0, int(ms)) / 1000:.3f}"


def _ffmpeg_thread_args() -> List[str]:
    return ["-threads", str(FFMPEG_THREADS)]


def _ffmpeg_video_thread_args() -> List[str]:
    return ["-threads:v", str(FFMPEG_THREADS)]


def _ffmpeg_filter_thread_args() -> List[str]:
    thread_count = FFMPEG_THREADS if FFMPEG_THREADS > 0 else max(1, os.cpu_count() or 1)
    return ["-filter_threads", str(thread_count), "-filter_complex_threads", str(thread_count)]


def _existing_path_or_none(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    return normalized if os.path.exists(normalized) else None


def _infer_subtitle_font_name_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    lower_name = os.path.basename(path).lower()
    if "msyh" in lower_name or "yahei" in lower_name:
        return "Microsoft YaHei"
    if "simhei" in lower_name:
        return "SimHei"
    if "simsun" in lower_name:
        return "SimSun"
    if "notosanscjk" in lower_name or "noto sans cjk" in lower_name:
        return "Noto Sans CJK SC"
    if "notosanssc" in lower_name or "noto sans sc" in lower_name:
        return "Noto Sans SC"
    if "wqy" in lower_name or "wenquanyi" in lower_name:
        return "WenQuanYi Micro Hei"
    if "sourcehansans" in lower_name or "source han sans" in lower_name:
        return "Source Han Sans SC"
    return None


def _subtitle_font_candidates() -> List[Tuple[str, List[str]]]:
    local_fonts = os.path.join(current_dir, "fonts")
    bundled_noto_fonts = os.path.join(local_fonts, "noto-cjk")
    return [
        (
            "Noto Sans CJK SC",
            [
                os.path.join(bundled_noto_fonts, "NotoSansCJKsc-Regular.otf"),
                os.path.join(local_fonts, "NotoSansCJK-Regular.ttc"),
                os.path.join(local_fonts, "NotoSansCJKsc-Regular.otf"),
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                "/usr/local/share/fonts/NotoSansCJK-Regular.ttc",
            ],
        ),
        (
            "Noto Sans SC",
            [
                os.path.join(local_fonts, "NotoSansSC-Regular.otf"),
                "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
                "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
                "/usr/local/share/fonts/NotoSansSC-Regular.otf",
            ],
        ),
        (
            "Source Han Sans SC",
            [
                os.path.join(local_fonts, "SourceHanSansSC-Regular.otf"),
                "/usr/share/fonts/opentype/adobe-source-han-sans/SourceHanSansSC-Regular.otf",
                "/usr/local/share/fonts/SourceHanSansSC-Regular.otf",
            ],
        ),
        (
            "WenQuanYi Micro Hei",
            [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
            ],
        ),
        (
            "Microsoft YaHei",
            [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\msyh.ttf",
                r"C:\Windows\Fonts\msyhbd.ttc",
            ],
        ),
        ("SimHei", [r"C:\Windows\Fonts\simhei.ttf"]),
        ("SimSun", [r"C:\Windows\Fonts\simsun.ttc"]),
        (
            "PingFang SC",
            [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
            ],
        ),
    ]


@functools.lru_cache(maxsize=1)
def _resolve_video_subtitle_font() -> Tuple[Optional[str], Optional[str]]:
    configured_font_name = VIDEO_SUBTITLE_FONT or None
    configured_fonts_dir = _existing_path_or_none(VIDEO_SUBTITLE_FONTS_DIR)
    configured_font_file = _existing_path_or_none(VIDEO_SUBTITLE_FONT_FILE)

    if configured_font_file:
        return (
            configured_font_name or _infer_subtitle_font_name_from_path(configured_font_file),
            os.path.dirname(configured_font_file),
        )
    if VIDEO_SUBTITLE_FONT_FILE:
        print(f"⚠️ VIDEO_SUBTITLE_FONT_FILE does not exist: {VIDEO_SUBTITLE_FONT_FILE}")

    if configured_font_name:
        return configured_font_name, configured_fonts_dir

    for font_name, candidate_paths in _subtitle_font_candidates():
        for candidate_path in candidate_paths:
            font_path = _existing_path_or_none(candidate_path)
            if font_path:
                return font_name, os.path.dirname(font_path)

    print(
        "⚠️ No CJK subtitle font found for FFmpeg rendering. "
        "Install a CJK font such as Noto Sans CJK, or set VIDEO_SUBTITLE_FONT_FILE "
        "and optionally VIDEO_SUBTITLE_FONT."
    )
    return None, configured_fonts_dir


def _ffmpeg_escape_filter_path(path: str) -> str:
    normalized = os.path.abspath(path).replace("\\", "/")
    return (
        normalized
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def _ffmpeg_escape_filter_quoted_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _subtitle_style_for_video(video_path: Optional[str]) -> Tuple[int, float, float, int]:
    dimensions = _ffprobe_video_dimensions(video_path) if video_path else None
    if not dimensions:
        return VIDEO_SUBTITLE_FONT_SIZE, 1.5, 0.7, 36
    width, height = dimensions
    short_side = max(1, min(width, height))
    scale = short_side / 1080.0
    font_size = int(round(VIDEO_SUBTITLE_FONT_SIZE * scale))
    font_size = max(12, min(VIDEO_SUBTITLE_FONT_SIZE, font_size))
    outline = round(max(0.8, min(1.5, 1.5 * scale)), 2)
    shadow = round(max(0.35, min(0.7, 0.7 * scale)), 2)
    margin_v = int(round(36 * scale))
    margin_v = max(14, min(36, margin_v))
    return font_size, outline, shadow, margin_v


def _ffmpeg_subtitle_force_style(font_name: Optional[str], video_path: Optional[str] = None) -> str:
    font_size, outline, shadow, margin_v = _subtitle_style_for_video(video_path)
    style_parts = [
        f"FontSize={font_size}",
        "PrimaryColour=&H00FFFFFF",
        "OutlineColour=&H00000000",
        "BorderStyle=1",
        f"Outline={outline}",
        f"Shadow={shadow}",
        f"MarginV={margin_v}",
    ]
    if font_name:
        style_parts.insert(0, f"FontName={font_name}")
    return ",".join(style_parts)


def _ffmpeg_subtitles_filter_arg(subtitle_path: str, video_path: Optional[str] = None) -> str:
    font_name, fonts_dir = _resolve_video_subtitle_font()
    options = [f"filename='{_ffmpeg_escape_filter_path(subtitle_path)}'"]
    ext = os.path.splitext(subtitle_path)[1].lstrip(".").lower()
    if ext in {"srt", "vtt"}:
        options.append("charenc=UTF-8")
    if fonts_dir:
        options.append(f"fontsdir='{_ffmpeg_escape_filter_path(fonts_dir)}'")
    force_style = _ffmpeg_subtitle_force_style(font_name, video_path)
    if force_style:
        options.append(f"force_style='{_ffmpeg_escape_filter_quoted_value(force_style)}'")
    return "subtitles=" + ":".join(options)


def _ffmpeg_codec_args_for_format(fmt: Optional[str], bitrate: Optional[str]) -> List[str]:
    fmt_normalized = (fmt or "").lower()
    codec_map = {
        "mp3": ["-c:a", "libmp3lame"],
        "ogg": ["-c:a", "libvorbis"],
        "opus": ["-c:a", "libopus"],
        "aac": ["-c:a", "aac"],
        "webm": ["-c:a", "libopus"],
        "wav": ["-c:a", "pcm_s16le"],
        "flac": ["-c:a", "flac"],
    }
    codec_args = codec_map.get(fmt_normalized, ["-c:a", "libmp3lame"])
    if bitrate and fmt_normalized in {"mp3", "ogg", "opus", "aac", "webm"}:
        codec_args = codec_args + ["-b:a", bitrate]
    return codec_args + _ffmpeg_thread_args()


def _run_ffmpeg_command(cmd: List[str]) -> None:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        cmd_str = " ".join(cmd)
        raise RuntimeError(
            f"ffmpeg failed (code {process.returncode})\ncmd: {cmd_str}\n"
            f"stderr: {stderr or '<empty>'}\nstdout: {stdout or '<empty>'}"
        )


VIDEO_DOWNLOAD_EXTENSIONS = {"mp4", "mkv", "webm", "mov", "m4v"}
VIDEO_DOWNLOAD_FORMATS = {
    "best": "bestvideo*+bestaudio/bestvideo+bestaudio/best",
    "2160p": "bestvideo*[height<=2160]+bestaudio/bestvideo[height<=2160]+bestaudio/best[height<=2160]/best",
    "1440p": "bestvideo*[height<=1440]+bestaudio/bestvideo[height<=1440]+bestaudio/best[height<=1440]/best",
    "1080p": "bestvideo*[height<=1080]+bestaudio/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
    "720p": "bestvideo*[height<=720]+bestaudio/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
    "480p": "bestvideo*[height<=480]+bestaudio/bestvideo[height<=480]+bestaudio/best[height<=480]/best",
    "360p": "bestvideo*[height<=360]+bestaudio/bestvideo[height<=360]+bestaudio/best[height<=360]/best",
    "worst": "worstvideo*+worstaudio/worst",
}


def _dedupe_preserving_order(values: List[Optional[str]]) -> List[Optional[str]]:
    seen: Set[Optional[str]] = set()
    result: List[Optional[str]] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _video_download_format_fallbacks(format_selector: str, quality_key: str) -> List[Optional[str]]:
    """Return format selectors from most-specific to safest."""
    candidates: List[Optional[str]] = [format_selector]
    if quality_key in VIDEO_DOWNLOAD_FORMATS:
        height_match = re.fullmatch(r"(\d+)p", quality_key)
        if height_match:
            height = height_match.group(1)
            candidates.extend(
                [
                    f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best",
                    f"bestvideo*[height<={height}]/best[height<={height}]/best",
                ]
            )
        elif quality_key == "best":
            candidates.extend(["bestvideo+bestaudio/best", "bestvideo*/best"])
        elif quality_key == "worst":
            candidates.extend(["worstvideo+worstaudio/worst", "worstvideo*/worst"])
    else:
        candidates.append("bestvideo+bestaudio/best")
    candidates.extend(["best", None])
    return _dedupe_preserving_order(candidates)


def _ytdlp_setup_hint() -> str:
    return (
        "YouTube currently requires yt-dlp's external JavaScript challenge solver setup. "
        "Upgrade with `pip install -U \"yt-dlp[default]\"`, install a supported JS runtime "
        "such as Deno or Node, enable that runtime for the yt-dlp Python API, and use a "
        "running PO Token provider for affected YouTube videos."
    )


def _ytdlp_auth_hint() -> str:
    return (
        "YouTube is asking this server to sign in. Import fresh youtube.com cookies exported from "
        "a private/incognito YouTube session, then retry. For CLI verification, pass the same file "
        "with `--cookies /path/to/youtube_com_cookies.txt`."
    )


def _yt_dlp_common_opts() -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    js_runtimes: Dict[str, Dict[str, str]] = {}
    configured_runtimes = (os.environ.get("YTDLP_JS_RUNTIMES") or "").strip()
    configured_node = (os.environ.get("YTDLP_NODE_PATH") or os.environ.get("NODE_BINARY") or "").strip()

    def find_executable(name: str) -> Optional[str]:
        found = shutil.which(name)
        if found:
            return found
        # Uvicorn services often have a thinner PATH than an interactive shell.
        # Ask a login shell as a last resort so it can see nvm/homebrew setup.
        try:
            process = subprocess.run(
                ["bash", "-lc", f"command -v {name}"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            candidate = (process.stdout or "").strip().splitlines()[0] if process.stdout.strip() else ""
            if candidate and os.path.exists(candidate):
                return candidate
        except Exception:
            pass
        if name == "node":
            try:
                process = subprocess.run(
                    ["bash", "-lc", "node -p 'process.execPath'"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                candidate = (process.stdout or "").strip().splitlines()[0] if process.stdout.strip() else ""
                if candidate and os.path.exists(candidate):
                    return candidate
            except Exception:
                pass
        if name == "node":
            node_candidates: List[str] = []
            home = Path.home()
            node_candidates.extend(str(path) for path in (home / ".nvm" / "versions" / "node").glob("*/bin/node"))
            node_candidates.extend(str(path) for path in Path("/home").glob("*/.nvm/versions/node/*/bin/node"))
            node_candidates.extend(
                [
                    "/usr/local/bin/node",
                    "/usr/bin/node",
                    "/snap/bin/node",
                    "/opt/node/bin/node",
                    "/home/linuxbrew/.linuxbrew/bin/node",
                ]
            )
            existing = [path for path in node_candidates if os.path.exists(path)]
            if existing:
                existing.sort(reverse=True)
                return existing[0]
        return None

    runtime_paths = {
        "node": configured_node if configured_node and os.path.exists(configured_node) else find_executable("node"),
        "deno": find_executable("deno"),
        "bun": find_executable("bun"),
        "quickjs": find_executable("qjs") or find_executable("quickjs"),
    }
    if configured_runtimes:
        runtime_items: List[Tuple[str, Optional[str]]] = []
        for item in configured_runtimes.split(","):
            runtime_spec = item.strip()
            if not runtime_spec:
                continue
            name, _, explicit_path = runtime_spec.partition(":")
            runtime_items.append((name.strip(), explicit_path.strip() or runtime_paths.get(name.strip())))
    elif runtime_paths.get("node"):
        # Mirror the known-good CLI path: --js-runtimes node.
        runtime_items = [("node", runtime_paths["node"])]
    else:
        runtime_items = [(name, path) for name, path in runtime_paths.items()]

    for runtime, executable in runtime_items:
        if runtime and executable:
            js_runtimes[runtime] = {"path": executable}
    if js_runtimes:
        opts["js_runtimes"] = js_runtimes

    remote_components = [
        item.strip()
        for item in (os.environ.get("YTDLP_REMOTE_COMPONENTS") or "").split(",")
        if item.strip()
    ]
    if remote_components:
        opts["remote_components"] = set(remote_components)
    return opts


def _raw_format_is_direct(fmt: Dict[str, Any]) -> bool:
    protocol = str(fmt.get("protocol") or "").lower()
    ext = str(fmt.get("ext") or "").lower()
    if not fmt.get("url") or fmt.get("has_drm"):
        return False
    if protocol in {"mhtml", "storyboard"} or ext in {"mhtml", "json"}:
        return False
    return True


def _raw_format_score(fmt: Dict[str, Any]) -> Tuple[int, float, float, int]:
    height = int(fmt.get("height") or 0)
    tbr = float(fmt.get("tbr") or 0)
    fps = float(fmt.get("fps") or 0)
    size = int(fmt.get("filesize") or fmt.get("filesize_approx") or 0)
    return (height, tbr, fps, size)


def _extract_format_ids_from_selector(selector: str) -> List[str]:
    tokens = re.split(r"[+/]", selector or "")
    ids: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token.startswith("best") or token.startswith("worst"):
            continue
        token = re.sub(r"\[.*$", "", token).strip()
        if token:
            ids.append(token)
    return ids


def _select_raw_video_formats(
    raw_info: Dict[str, Any],
    quality_key: str,
    format_selector: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    raw_formats = [fmt for fmt in (raw_info.get("formats") or []) if isinstance(fmt, dict) and _raw_format_is_direct(fmt)]
    by_id = {str(fmt.get("format_id")): fmt for fmt in raw_formats if fmt.get("format_id") is not None}
    audio_formats = [
        fmt for fmt in raw_formats
        if (fmt.get("acodec") or "none") != "none" and (fmt.get("vcodec") or "none") == "none"
    ]
    audio_formats.sort(key=_raw_format_score, reverse=True)

    for format_id in _extract_format_ids_from_selector(format_selector):
        fmt = by_id.get(format_id)
        if not fmt or (fmt.get("vcodec") or "none") == "none":
            continue
        if (fmt.get("acodec") or "none") != "none":
            return fmt, None
        return fmt, audio_formats[0] if audio_formats else None

    height_limit: Optional[int] = None
    height_match = re.fullmatch(r"(\d+)p", quality_key)
    if height_match:
        height_limit = int(height_match.group(1))

    video_formats = [fmt for fmt in raw_formats if (fmt.get("vcodec") or "none") != "none"]
    if height_limit:
        limited = [fmt for fmt in video_formats if int(fmt.get("height") or 0) <= height_limit]
        if limited:
            video_formats = limited
    reverse = quality_key != "worst"
    combined_formats = [fmt for fmt in video_formats if (fmt.get("acodec") or "none") != "none"]
    video_only_formats = [fmt for fmt in video_formats if (fmt.get("acodec") or "none") == "none"]
    combined_formats.sort(key=_raw_format_score, reverse=reverse)
    video_only_formats.sort(key=_raw_format_score, reverse=reverse)

    if video_only_formats and audio_formats and _ffmpeg_available():
        return video_only_formats[0], audio_formats[0]
    if combined_formats:
        return combined_formats[0], None
    if video_only_formats:
        return video_only_formats[0], None
    return None, None


def _safe_video_output_path(raw_info: Dict[str, Any], ext: str) -> str:
    title = _sanitize_base_filename(str(raw_info.get("title") or raw_info.get("id") or "downloaded_video"))
    video_id = _sanitize_base_filename(str(raw_info.get("id") or "")) or uuid.uuid4().hex[:8]
    base = (title or "downloaded_video")[:160].strip()
    safe_ext = re.sub(r"[^A-Za-z0-9]", "", ext or "").lower() or "mp4"
    if safe_ext not in VIDEO_DOWNLOAD_EXTENSIONS:
        safe_ext = "mp4"
    filename = f"{base} [{video_id}].{safe_ext}"
    path = os.path.join(VIDEO_DOWNLOAD_DIR, filename)
    if not os.path.exists(path):
        return path
    for index in range(2, 1000):
        candidate = os.path.join(VIDEO_DOWNLOAD_DIR, f"{base} [{video_id}] ({index}).{safe_ext}")
        if not os.path.exists(candidate):
            return candidate
    return os.path.join(VIDEO_DOWNLOAD_DIR, f"{base} [{video_id}] {uuid.uuid4().hex[:6]}.{safe_ext}")


def _download_direct_format(
    fmt: Dict[str, Any],
    output_path: str,
    emit: Callable[..., None],
    message: str,
) -> None:
    headers = dict(fmt.get("http_headers") or {})
    headers.setdefault("User-Agent", "Mozilla/5.0")
    request = urllib.request.Request(str(fmt["url"]), headers=headers)
    downloaded = 0
    with urllib.request.urlopen(request, timeout=60) as response, open(output_path, "wb") as fh:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            percent = round((downloaded / total) * 100, 2) if total else None
            emit(
                "progress",
                message=message,
                percent=percent,
                downloaded_bytes=downloaded,
                total_bytes=total,
                speed=None,
                eta=None,
            )


def _manual_video_download_from_raw_info(
    url: str,
    quality_key: str,
    format_selector: str,
    emit: Callable[..., None],
) -> str:
    raw_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,
        "check_formats": False,
        "format": None,
        **_yt_dlp_common_opts(),
        **_get_cookie_opts_for_url(url),
    }
    with yt_dlp.YoutubeDL(raw_opts) as ydl:
        raw_info = ydl.extract_info(url, download=False, process=False)
    if not isinstance(raw_info, dict):
        raise RuntimeError("yt-dlp returned no raw video metadata for manual fallback.")

    video_fmt, audio_fmt = _select_raw_video_formats(raw_info, quality_key, format_selector)
    if not video_fmt:
        raise RuntimeError(
            "yt-dlp could fetch metadata, but did not expose a direct downloadable video stream. "
            f"{_ytdlp_setup_hint()}"
        )

    if audio_fmt:
        if not _ffmpeg_available():
            raise RuntimeError("ffmpeg is required to merge separate video and audio streams.")
        output_path = _safe_video_output_path(raw_info, "mkv")
        with tempfile.TemporaryDirectory(prefix="video_stream_", dir=VIDEO_DOWNLOAD_DIR) as tmp_dir:
            video_ext = str(video_fmt.get("ext") or "video")
            audio_ext = str(audio_fmt.get("ext") or "audio")
            video_path = os.path.join(tmp_dir, f"video.{video_ext}")
            audio_path = os.path.join(tmp_dir, f"audio.{audio_ext}")
            _download_direct_format(video_fmt, video_path, emit, "Downloading video stream...")
            _download_direct_format(audio_fmt, audio_path, emit, "Downloading audio stream...")
            emit("status", message="Merging video and audio streams...")
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                output_path,
            ]
            _run_ffmpeg_command(cmd)
        return output_path

    output_ext = str(video_fmt.get("ext") or "mp4").lower()
    output_path = _safe_video_output_path(raw_info, output_ext)
    _download_direct_format(video_fmt, output_path, emit, "Downloading video stream...")
    return output_path


# ---------------------------------------------------------------------------
# Cookie helpers (ported from downloader.py CookieSettings)
# ---------------------------------------------------------------------------

def _extract_domain_from_url(url: str) -> str:
    """Extract the main domain from a video URL."""
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    if match:
        domain = match.group(1)
        parts = domain.split(".")
        if len(parts) > 2:
            domain = ".".join(parts[-2:])
        if domain == "youtu.be":
            domain = "youtube.com"
        return domain
    return "unknown"


def _extract_domain_from_curl(curl_text: str) -> str:
    """Extract the main domain from a cURL command."""
    url_match = re.search(r"curl\s+['\"]?(https?://[^'\"\s]+)", curl_text)
    if not url_match:
        url_match = re.search(r"https?://([^\s/'\"]+)", curl_text)
    if url_match:
        url = url_match.group(1) if "://" in url_match.group(1) else url_match.group(0)
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1)
            parts = domain.split(".")
            if len(parts) > 2:
                domain = ".".join(parts[-2:])
            if domain == "youtu.be":
                domain = "youtube.com"
            return domain
    return "unknown"


def _parse_curl_cookies(curl_text: str) -> Dict[str, str]:
    """Parse cookies from a cURL command string."""
    cookies: Dict[str, str] = {}
    patterns = [
        r"-H\s+['\"]Cookie:\s*([^'\"]+)['\"]",
        r"--header\s+['\"]Cookie:\s*([^'\"]+)['\"]",
        r"-b\s+['\"]([^'\"]+)['\"]",
        r"--cookie\s+['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, curl_text, re.IGNORECASE)
        for match in matches:
            pairs = match.split(";")
            for pair in pairs:
                pair = pair.strip()
                if "=" in pair:
                    name, value = pair.split("=", 1)
                    cookies[name.strip()] = value.strip()
    return cookies


def _get_saved_cookie_sites() -> Dict[str, int]:
    """Get list of saved cookie sites with cookie counts."""
    sites: Dict[str, int] = {}
    for cookies_dir in _candidate_cookie_dirs():
        if not os.path.exists(cookies_dir):
            continue
        for f in os.listdir(cookies_dir):
            if f.endswith("_cookies.txt"):
                domain = f.replace("_cookies.txt", "").replace("_", ".")
                filepath = os.path.join(cookies_dir, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        lines = [line for line in fh.readlines() if line.strip() and not line.startswith("#")]
                        sites[domain] = max(sites.get(domain, 0), len(lines))
                except Exception:
                    sites.setdefault(domain, 0)
    return sites


def _candidate_cookie_dirs() -> List[str]:
    """Cookie directories from both possible output roots, deduped."""
    candidates = [
        COOKIES_DIR,
        str((APP_DIR / "outputs" / "cookies").resolve()),
        str((APP_DIR.parent / "outputs" / "cookies").resolve()),
    ]
    result: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _get_cookie_file_for_url(url: str) -> Optional[str]:
    """Get the appropriate cookie file for a given video URL."""
    domain = _extract_domain_from_url(url)
    # Try exact match first
    filename = domain.replace(".", "_") + "_cookies.txt"
    for cookies_dir in _candidate_cookie_dirs():
        filepath = os.path.join(cookies_dir, filename)
        if os.path.exists(filepath):
            return filepath
    # Try partial matches (e.g., for subdomains)
    for cookies_dir in _candidate_cookie_dirs():
        if not os.path.exists(cookies_dir):
            continue
        for f in os.listdir(cookies_dir):
            if f.endswith("_cookies.txt"):
                saved_domain = f.replace("_cookies.txt", "").replace("_", ".")
                if saved_domain in domain or domain in saved_domain:
                    return os.path.join(cookies_dir, f)
    return None


def _get_cookie_opts_for_url(url: str) -> Dict[str, Any]:
    """Return yt-dlp cookie options dict for a URL, or empty dict."""
    cookie_file = _get_cookie_file_for_url(url)
    if cookie_file:
        return {"cookiefile": cookie_file}
    return {}


def _get_cookie_path_for_url(url: str) -> Optional[str]:
    """Return the saved cookie file path for diagnostics."""
    return _get_cookie_file_for_url(url)


def _safe_video_download_filename(filename: str) -> str:
    safe_name = os.path.basename(filename or "")
    if not safe_name or safe_name != filename:
        raise ValueError("Invalid video filename.")
    return safe_name


DOWNLOADED_VIDEO_ID_PREFIX = "v_"


def _encode_downloaded_video_id(filename: str) -> str:
    safe_name = _safe_video_download_filename(filename)
    encoded = base64.urlsafe_b64encode(safe_name.encode("utf-8")).decode("ascii")
    return DOWNLOADED_VIDEO_ID_PREFIX + encoded.rstrip("=")


def _decode_downloaded_video_id(video_id_or_filename: str) -> str:
    value = (video_id_or_filename or "").strip()
    if value.startswith(DOWNLOADED_VIDEO_ID_PREFIX):
        payload = value[len(DOWNLOADED_VIDEO_ID_PREFIX):]
        padding = "=" * (-len(payload) % 4)
        try:
            return base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8")
        except Exception as exc:
            raise ValueError("Invalid downloaded video id.") from exc
    return value


def _downloaded_video_path(video_id_or_filename: str) -> str:
    safe_name = _safe_video_download_filename(_decode_downloaded_video_id(video_id_or_filename))
    path = os.path.abspath(os.path.join(VIDEO_DOWNLOAD_DIR, safe_name))
    root = os.path.abspath(VIDEO_DOWNLOAD_DIR)
    if os.path.commonpath([root, path]) != root:
        raise ValueError("Invalid video path.")
    return path


def _ffprobe_duration_seconds(path: str) -> Optional[float]:
    if shutil.which("ffprobe") is None or not os.path.exists(path):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if process.returncode != 0:
            return None
        value = (process.stdout or "").strip()
        return float(value) if value else None
    except Exception:
        return None


def _ffprobe_video_dimensions(path: str) -> Optional[Tuple[int, int]]:
    if shutil.which("ffprobe") is None or not os.path.exists(path):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        path,
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if process.returncode != 0:
            return None
        value = (process.stdout or "").strip().splitlines()[0]
        width_text, height_text = value.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception:
        return None


def _format_video_duration(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return ""
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_size_mb(size_bytes: int) -> float:
    return round(max(0, int(size_bytes or 0)) / (1024 * 1024), 2)


def _downloaded_video_entry(path: str) -> Dict[str, Any]:
    safe_name = os.path.basename(path)
    video_id = _encode_downloaded_video_id(safe_name)
    stat = os.stat(path)
    duration_seconds = _ffprobe_duration_seconds(path)
    return {
        "id": video_id,
        "filename": safe_name,
        "title": os.path.splitext(safe_name)[0],
        "extension": os.path.splitext(safe_name)[1].lstrip(".").lower(),
        "size_bytes": int(stat.st_size),
        "size_mb": _format_size_mb(stat.st_size),
        "mtime": stat.st_mtime,
        "duration_seconds": duration_seconds,
        "duration_label": _format_video_duration(duration_seconds),
        "url": f"/api/downloaded_videos/{video_id}",
        "poster_url": f"/api/downloaded_videos/{video_id}/snapshot",
    }


def _list_downloaded_video_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not os.path.exists(VIDEO_DOWNLOAD_DIR):
        return entries
    for entry in os.scandir(VIDEO_DOWNLOAD_DIR):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lstrip(".").lower()
        if ext not in VIDEO_DOWNLOAD_EXTENSIONS:
            continue
        try:
            entries.append(_downloaded_video_entry(entry.path))
        except Exception as exc:
            print(f"⚠️ Failed to inspect downloaded video '{entry.name}': {exc}")
    entries.sort(key=lambda item: item.get("mtime") or 0, reverse=True)
    return entries


def _find_recent_downloaded_video(start_time: float, info: Optional[Dict[str, Any]] = None) -> Optional[str]:
    candidates: List[str] = []
    video_id = str((info or {}).get("id") or "").strip()
    for entry in os.scandir(VIDEO_DOWNLOAD_DIR):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lstrip(".").lower()
        if ext not in VIDEO_DOWNLOAD_EXTENSIONS:
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        if video_id and video_id not in entry.name:
            continue
        if stat.st_mtime >= start_time - 5:
            candidates.append(entry.path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: os.path.getmtime(item), reverse=True)
    return candidates[0]


def _downloaded_video_audio_cache_path(video_path: str) -> str:
    stat = os.stat(video_path)
    cache_key = hashlib.md5(
        f"{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    stem = _sanitize_base_filename(os.path.basename(video_path)) or "downloaded_video"
    return os.path.join(VIDEO_AUDIO_CACHE_DIR, f"{stem}_{cache_key}.mp3")


def _video_snapshot_cache_path(video_path: str) -> str:
    stat = os.stat(video_path)
    cache_key = hashlib.md5(
        f"{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    stem = _sanitize_base_filename(os.path.basename(video_path)) or "video"
    return os.path.join(VIDEO_SNAPSHOT_CACHE_DIR, f"{stem}_{cache_key}.jpg")


def _generate_video_snapshot_sync(video_path: str) -> str:
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is required to generate video snapshots.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = _video_snapshot_cache_path(video_path)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temp_output = f"{output_path}.{uuid.uuid4().hex[:8]}.tmp.jpg"
    try:
        last_error: Optional[Exception] = None
        for seek_seconds in ("1", "0"):
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                seek_seconds,
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-vf",
                "format=yuvj420p",
                "-q:v",
                "3",
                temp_output,
            ]
            try:
                _run_ffmpeg_command(cmd)
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    os.replace(temp_output, output_path)
                    return output_path
            except Exception as exc:
                last_error = exc
                _safe_remove_file(temp_output)
        raise RuntimeError(f"Failed to generate video snapshot: {last_error}")
    finally:
        _safe_remove_file(temp_output)


def _extract_audio_from_downloaded_video_sync(video_path: str) -> str:
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is required to extract audio from downloaded videos.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Downloaded video not found: {video_path}")
    output_path = _downloaded_video_audio_cache_path(video_path)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        *_ffmpeg_thread_args(),
        output_path,
    ]
    _run_ffmpeg_command(cmd)
    return output_path


async def _prepare_downloaded_video_audio_request(
    downloaded_video_id: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[bytes], Optional[str], Optional[JSONResponse]]:
    video_id = (downloaded_video_id or "").strip()
    if not video_id:
        return None, None, None, None
    try:
        video_path = _downloaded_video_path(video_id)
    except ValueError as exc:
        return None, None, None, _status_error(str(exc), status_code=400)
    if not os.path.exists(video_path):
        return None, None, None, _status_error("Downloaded video not found.", status_code=404)
    try:
        audio_path = await _run_blocking(_extract_audio_from_downloaded_video_sync, video_path)
        audio_bytes = await _run_blocking(lambda: _read_file_bytes(audio_path))
        if not audio_bytes:
            return None, None, None, _status_error("Failed to extract audio from downloaded video.", status_code=500)
        source = _downloaded_video_entry(video_path)
        source["path"] = video_path
        source["audio_filename"] = os.path.basename(audio_path)
        source["audio_path"] = audio_path
        return source, audio_bytes, os.path.basename(audio_path), None
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, _status_error(
            f"Failed to extract audio from downloaded video: {str(exc)}",
            status_code=500,
        )


def _public_downloaded_video_source(source: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not source:
        return None
    return {
        key: value
        for key, value in source.items()
        if key not in {"path", "audio_path"}
    }


def _replace_video_audio_sync(
    video_path: str,
    audio_path: Optional[str],
    output_path: str,
    subtitle_path: Optional[str] = None,
    audio_mode: str = "translated",
    embedded_subtitle_paths: Optional[List[Tuple[str, str]]] = None,
) -> None:
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is required to render video outputs.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")
    normalized_audio_mode = (audio_mode or "translated").strip().lower()
    if normalized_audio_mode not in {"translated", "original", "both"}:
        raise ValueError(f"Unsupported video audio mode: {audio_mode}")
    needs_translated_audio = normalized_audio_mode in {"translated", "both"}
    if needs_translated_audio and (not audio_path or not os.path.exists(audio_path)):
        raise FileNotFoundError(f"Translated audio not found: {audio_path}")
    if subtitle_path and not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
    embedded_subtitles = embedded_subtitle_paths or []
    for embedded_path, _embedded_label in embedded_subtitles:
        if not os.path.exists(embedded_path):
            raise FileNotFoundError(f"Embedded subtitle file not found: {embedded_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    source_has_audio = _video_has_audio_stream_sync(video_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if subtitle_path:
        cmd += _ffmpeg_filter_thread_args()
    cmd += ["-i", video_path]
    if needs_translated_audio:
        cmd += ["-i", str(audio_path)]
    first_subtitle_input_index = 2 if needs_translated_audio else 1
    for embedded_path, _embedded_label in embedded_subtitles:
        cmd += ["-i", embedded_path]
    cmd += ["-map", "0:v:0"]
    if normalized_audio_mode == "original":
        cmd += ["-map", "0:a:0?"]
    elif normalized_audio_mode == "translated":
        cmd += ["-map", "1:a:0"]
    else:
        if source_has_audio:
            cmd += ["-map", "0:a:0"]
        cmd += ["-map", "1:a:0"]
    for subtitle_offset, (_embedded_path, _embedded_label) in enumerate(embedded_subtitles):
        cmd += ["-map", f"{first_subtitle_input_index + subtitle_offset}:0"]
    if subtitle_path:
        cmd += [
            "-vf",
            _ffmpeg_subtitles_filter_arg(subtitle_path, video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            *_ffmpeg_video_thread_args(),
        ]
    else:
        cmd += ["-c:v", "copy"]
    if normalized_audio_mode == "original":
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
        if normalized_audio_mode == "both":
            if source_has_audio:
                cmd += [
                    "-metadata:s:a:0",
                    "title=Original",
                    "-metadata:s:a:1",
                    "title=Translated",
                    "-disposition:a:0",
                    "0",
                    "-disposition:a:1",
                    "default",
                ]
            else:
                cmd += ["-metadata:s:a:0", "title=Translated", "-disposition:a:0", "default"]
    if embedded_subtitles:
        cmd += ["-c:s", "mov_text"]
        for subtitle_index, (_embedded_path, embedded_label) in enumerate(embedded_subtitles):
            label = embedded_label or f"Subtitle {subtitle_index + 1}"
            language = "eng" if label.lower() == "translated" else "und"
            cmd += [
                f"-metadata:s:s:{subtitle_index}",
                f"title={label}",
                f"-metadata:s:s:{subtitle_index}",
                f"handler_name={label}",
                f"-metadata:s:s:{subtitle_index}",
                f"language={language}",
                f"-disposition:s:{subtitle_index}",
                "default" if subtitle_index == 0 else "0",
            ]
    cmd += [*_ffmpeg_thread_args()]
    if not embedded_subtitles:
        cmd += ["-shortest"]
    cmd += ["-movflags", "+faststart", output_path]
    _run_ffmpeg_command(cmd)


def _video_has_audio_stream_sync(video_path: str) -> bool:
    """Return whether ffprobe can see at least one audio stream in a video."""
    if shutil.which("ffprobe") is None:
        return True
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    return process.returncode == 0 and bool(process.stdout.strip())


def _embedded_subtitle_label(filename: str) -> str:
    lower_name = filename.lower()
    if "original" in lower_name:
        return "Original"
    if "translated" in lower_name:
        return "Translated"
    return os.path.splitext(filename)[0] or "Subtitle"


def _source_video_metadata_from_session(session: Optional[TranslateSessionData]) -> Optional[Dict[str, Any]]:
    if not session or not session.source_video_filename:
        return None
    try:
        video_path = session.source_video_path or _downloaded_video_path(session.source_video_filename)
        if not os.path.exists(video_path):
            return None
        return _downloaded_video_entry(video_path)
    except Exception:
        return None


def _apply_source_video_metadata(metadata: Dict[str, Any], session: Optional[TranslateSessionData]) -> None:
    source_video = _source_video_metadata_from_session(session)
    if source_video:
        metadata["source_video"] = source_video
        metadata["source_video_filename"] = source_video.get("filename")


async def _log_overlay_progress(label: str, start_time: float, interval_seconds: float = 30.0) -> None:
    """Emit periodic overlay progress logs while ffmpeg runs."""
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            elapsed = time.perf_counter() - start_time
            PERF_LOGGER.info("⏳ [%s] ffmpeg overlay running (%.1fs elapsed)", label, elapsed)
    except asyncio.CancelledError:
        return


def _ffmpeg_extract_segment(
    input_path: str,
    output_path: str,
    start_ms: int,
    end_ms: int,
    *,
    reencode: bool = False,
    bitrate: Optional[str] = None,
) -> None:
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is not available on PATH.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio not found at {input_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    duration_ms = max(0, int(end_ms) - int(start_ms))
    if duration_ms <= 0:
        raise ValueError("Segment duration must be positive for ffmpeg extraction.")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        _ffmpeg_seconds_from_ms(start_ms),
        "-i",
        input_path,
        "-t",
        _ffmpeg_seconds_from_ms(duration_ms),
    ]
    if reencode:
        fmt = os.path.splitext(output_path)[1].lstrip(".") or TRANSLATE_DEFAULT_OUTPUT_FORMAT
        cmd += _ffmpeg_codec_args_for_format(fmt, bitrate or TRANSLATE_DEFAULT_BITRATE)
    else:
        cmd += ["-c", "copy"]
    cmd.append(output_path)

    try:
        _run_ffmpeg_command(cmd)
    except RuntimeError as exc:
        _safe_remove_file(output_path)
        if not reencode:
            _ffmpeg_extract_segment(
                input_path,
                output_path,
                start_ms,
                end_ms,
                reencode=True,
                bitrate=bitrate,
            )
        else:
            raise exc


def _ffmpeg_concat_files(
    input_paths: List[str],
    output_path: str,
    *,
    copy_codec: bool = True,
    target_format: Optional[str] = None,
    bitrate: Optional[str] = None,
) -> None:
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is not available on PATH.")
    if not input_paths:
        raise ValueError("No input files provided for ffmpeg concat.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    list_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            list_file = tmp.name
            for path in input_paths:
                abs_path = os.path.abspath(path)
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"Input audio for concat not found: {abs_path}")
                sanitized = abs_path.replace("'", "'\\''")
                tmp.write(f"file '{sanitized}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
        ]
        if copy_codec:
            cmd += ["-c", "copy"]
        else:
            fmt = target_format or os.path.splitext(output_path)[1].lstrip(".") or TRANSLATE_DEFAULT_OUTPUT_FORMAT
            cmd += _ffmpeg_codec_args_for_format(fmt, bitrate or TRANSLATE_DEFAULT_BITRATE)
        cmd.append(output_path)

        try:
            _run_ffmpeg_command(cmd)
        except RuntimeError as exc:
            _safe_remove_file(output_path)
            if copy_codec:
                _ffmpeg_concat_files(
                    input_paths,
                    output_path,
                    copy_codec=False,
                    target_format=target_format,
                    bitrate=bitrate,
                )
            else:
                raise exc
    finally:
        _safe_remove_file(list_file)


def _ffmpeg_overlay_tracks(
    vocals_path: str,
    backing_path: str,
    output_path: str,
    *,
    vocals_volume: float = 1.0,
    backing_volume: float = 1.0,
    output_format: Optional[str] = None,
    bitrate: Optional[str] = None,
) -> None:
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg is not available on PATH.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    filter_complex = (
        f"[0:a]volume={vocals_volume}[vocals];"
        f"[1:a]volume={backing_volume}[backing];"
        f"[vocals][backing]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        vocals_path,
        "-i",
        backing_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
    ]
    fmt = output_format or os.path.splitext(output_path)[1].lstrip(".") or TRANSLATE_DEFAULT_OUTPUT_FORMAT
    cmd += _ffmpeg_codec_args_for_format(fmt, bitrate or TRANSLATE_DEFAULT_BITRATE)
    cmd.append(output_path)

    try:
        _run_ffmpeg_command(cmd)
    except RuntimeError as exc:
        _safe_remove_file(output_path)
        raise exc


def _ffmpeg_concat_to_tempfile(
    input_paths: List[str],
    *,
    output_format: str,
    bitrate: Optional[str] = None,
) -> str:
    suffix = f".{output_format}" if output_format else ".mp3"
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=TRANSLATE_SESSION_MEDIA_DIR)
    tmp_file.close()
    try:
        _ffmpeg_concat_files(
            input_paths,
            tmp_file.name,
            copy_codec=True,
            target_format=output_format,
            bitrate=bitrate,
        )
        return tmp_file.name
    except Exception:
        _safe_remove_file(tmp_file.name)
        raise


def _ffmpeg_overlay_to_tempfile(
    vocals_path: str,
    backing_path: str,
    *,
    output_format: str,
    bitrate: Optional[str] = None,
    vocals_volume: float = 1.0,
    backing_volume: float = 1.0,
) -> str:
    suffix = f".{output_format}" if output_format else ".mp3"
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=TRANSLATE_SESSION_MEDIA_DIR)
    tmp_file.close()
    try:
        _ffmpeg_overlay_tracks(
            vocals_path,
            backing_path,
            tmp_file.name,
            vocals_volume=vocals_volume,
            backing_volume=backing_volume,
            output_format=output_format,
            bitrate=bitrate,
        )
        return tmp_file.name
    except Exception:
        _safe_remove_file(tmp_file.name)
        raise


async def _synthesize_translated_audio(
    original_audio: AudioSegment,
    segments: List[Dict[str, Any]],
    dest_language: str,
    response_format: str = TRANSLATE_DEFAULT_OUTPUT_FORMAT,
    bitrate: str = TRANSLATE_DEFAULT_BITRATE,
    input_mime_type: Optional[str] = None,
    clearvoice_settings: Optional[Dict[str, Any]] = None,
    backing_track_audio: Optional[AudioSegment] = None,
    backing_track_source: str = "none",
    merge_with_backing: bool = False,
    preserve_silence_audio: bool = False,
    generated_volume_percent: float = DEFAULT_GENERATED_VOLUME_PERCENT,
    silence_volume_percent: float = DEFAULT_SILENCE_VOLUME_PERCENT,
    speaker_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    backing_volume_percent: float = DEFAULT_GENERATED_VOLUME_PERCENT,
    pad_to_original: bool = True,
    default_speaker_preset: Optional[str] = None,
    default_emotion_weight: Optional[float] = None,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    return_unmixed_audio: bool = False,
) -> Tuple[bytes, str, Dict[str, Any]]:
    tts = tts_manager.get_tts()
    frame_rate = int(original_audio.frame_rate or 22050)
    sample_width = int(original_audio.sample_width or 2)
    channels = int(original_audio.channels or 1)

    combined_audio = _create_silence_segment(0, frame_rate, sample_width, channels)
    generation_log: List[Dict[str, Any]] = []
    tts_concurrency = max(1, int(TRANSLATION_TTS_CONCURRENCY or 1))
    semaphore = asyncio.Semaphore(tts_concurrency)
    override_map: Dict[str, Dict[str, Any]] = {
        str(key).lower(): value for key, value in (speaker_overrides or {}).items()
    }
    normalized_default_speaker = (default_speaker_preset or "").strip()
    normalized_default_emotion = _coerce_emotion_weight(
        default_emotion_weight,
        DEFAULT_EMOTION_WEIGHT,
    )
    silence_volume_percent = _coerce_volume_percent(
        silence_volume_percent,
        DEFAULT_SILENCE_VOLUME_PERCENT,
    )

    async def process_segment(index: int, segment: Dict[str, Any]):
        seg_type = segment.get("type")
        start_ms = int(segment.get("start_ms", 0))
        end_ms = int(segment.get("end_ms", start_ms))
        duration_ms = max(0, int(segment.get("duration_ms", max(0, end_ms - start_ms))))

        if seg_type == "silence":
            if preserve_silence_audio:
                chunk_audio = original_audio[start_ms:end_ms]
                chunk_audio = _apply_volume_with_ffmpeg(chunk_audio, silence_volume_percent)
                audio_seg = _match_segment_duration(chunk_audio, duration_ms, frame_rate, sample_width, channels)
                log_entry = {
                    "index": index,
                    "type": "silence",
                    "status": "preserved",
                    "duration_ms": duration_ms,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            else:
                audio_seg = _create_silence_segment(duration_ms, frame_rate, sample_width, channels)
                log_entry = {
                    "index": index,
                    "type": "silence",
                    "duration_ms": duration_ms,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            return index, audio_seg, log_entry

        translated_text = (segment.get("translated_text") or "").strip()
        source_text = (segment.get("source_text") or "").strip()
        generate_segment = segment.get("generate", True)
        keep_original = segment.get("keep_original", False) or (not generate_segment)

        chunk_audio = original_audio[start_ms:end_ms]
        speaker_label = str(segment.get("speaker") or "").strip().lower()
        override_config = override_map.get(speaker_label) if speaker_label else None
        preset_name = None
        use_emotion_prompt = False
        speaker_emotion_weight = DEFAULT_EMOTION_WEIGHT
        speaker_volume_percent: Optional[float] = None
        if override_config:
            raw_volume = override_config.get("volume_percent")
            if raw_volume is not None:
                speaker_volume_percent = _coerce_volume_percent(
                    raw_volume,
                    DEFAULT_GENERATED_VOLUME_PERCENT,
                )
        if override_config:
            preset_candidate = override_config.get("preset_name")
            if isinstance(preset_candidate, str) and preset_candidate.strip():
                preset_name = preset_candidate.strip()
                use_emotion_prompt = bool(override_config.get("use_emotion_prompt"))
                speaker_emotion_weight = _coerce_emotion_weight(
                    override_config.get("emotion_weight"),
                    DEFAULT_EMOTION_WEIGHT,
                )
        if not preset_name and normalized_default_speaker:
            preset_name = normalized_default_speaker
            use_emotion_prompt = True
            speaker_emotion_weight = normalized_default_emotion
        segment_volume_percent: Optional[float] = None
        raw_segment_volume = segment.get("volume_percent")
        if raw_segment_volume is not None:
            segment_volume_percent = _coerce_volume_percent(
                raw_segment_volume,
                DEFAULT_GENERATED_VOLUME_PERCENT,
            )
        segment_emotion_weight: Optional[float] = None
        raw_segment_emotion = segment.get("emotion_weight")
        if raw_segment_emotion is not None:
            segment_emotion_weight = _coerce_emotion_weight(
                raw_segment_emotion,
                DEFAULT_EMOTION_WEIGHT,
            )
        resolved_emotion_weight = (
            segment_emotion_weight
            if segment_emotion_weight is not None
            else speaker_emotion_weight
        )
        resolved_volume_percent = (
            segment_volume_percent
            if segment_volume_percent is not None
            else (
                speaker_volume_percent
                if speaker_volume_percent is not None
                else generated_volume_percent
            )
        )

        if keep_original:
            audio_seg = _match_segment_duration(chunk_audio, duration_ms, frame_rate, sample_width, channels)
            log_entry = {
                "index": index,
                "type": "speech",
                "status": "preserved",
                "duration_ms": duration_ms,
                "source_text": source_text,
                "translated_text": translated_text,
            }
            return index, audio_seg, log_entry

        if not translated_text:
            audio_seg = _create_silence_segment(duration_ms, frame_rate, sample_width, channels)
            log_entry = {
                "index": index,
                "type": "speech",
                "status": "skipped",
                "reason": "empty_translation",
                "source_text": source_text,
                "duration_ms": duration_ms,
            }
            return index, audio_seg, log_entry

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            chunk_audio.export(tmp_chunk.name, format="wav")
            chunk_path = tmp_chunk.name

        generation_target_ms = max(0, duration_ms - AUDIO_GENERATION_MARGIN_MS)
        generated_path = os.path.join("outputs", f"translate_{uuid.uuid4().hex}.wav")

        status = "success"
        error_message = None
        generated_audio: Optional[AudioSegment] = None
        spk_prompt_value = chunk_path if not preset_name else ""
        emo_prompt_value = chunk_path if preset_name and use_emotion_prompt else None

        try:
            async with semaphore:
                inference_path = await tts.infer(
                    spk_audio_prompt=spk_prompt_value,
                    text=translated_text,
                    output_path=generated_path,
                    interval_silence=0,
                    speech_length=generation_target_ms,
                    diffusion_steps=10,
                    verbose=SETTINGS.verbose,
                    speaker_preset=preset_name,
                    emo_audio_prompt=emo_prompt_value,
                    emo_alpha=resolved_emotion_weight,
                )
            generated_audio = AudioSegment.from_file(inference_path)
            if abs(resolved_volume_percent - DEFAULT_GENERATED_VOLUME_PERCENT) >= 0.01:
                generated_audio = _apply_volume_with_ffmpeg(generated_audio, resolved_volume_percent)
            generated_audio = _match_segment_duration(generated_audio, duration_ms, frame_rate, sample_width, channels)
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            print(f"⚠️ Translation synthesis failed for segment {index}: {error_message}")
            generated_audio = _create_silence_segment(duration_ms, frame_rate, sample_width, channels)
        finally:
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass
            if os.path.exists(generated_path):
                try:
                    await async_remove_file(generated_path)
                except Exception:
                    pass

        log_entry: Dict[str, Any] = {
            "index": index,
            "type": "speech",
            "status": status,
            "duration_ms": duration_ms,
            "source_text": source_text,
            "translated_text": translated_text,
        }
        if speaker_label:
            log_entry["speaker"] = speaker_label
        if preset_name:
            log_entry["speaker_preset"] = preset_name
            log_entry["emotion_prompt"] = use_emotion_prompt
            log_entry["emotion_weight"] = resolved_emotion_weight
        if speaker_volume_percent is not None or segment_volume_percent is not None:
            log_entry["volume_percent"] = resolved_volume_percent
        if status == "error" and error_message:
            log_entry["error"] = error_message

        return index, generated_audio, log_entry

    # Count speech segments for progress tracking
    speech_segments = [seg for seg in segments if seg.get("type") == "speech"]
    total_speech = len(speech_segments)
    completed_count = 0
    completed_lock = asyncio.Lock()
    
    async def process_segment_with_progress(idx: int, segment: Dict[str, Any]):
        nonlocal completed_count
        result = await process_segment(idx, segment)
        
        # Track progress for speech segments only
        if segment.get("type") == "speech":
            async with completed_lock:
                completed_count += 1
                current = completed_count
            
            # Emit progress update
            if emit_status and total_speech > 0:
                await emit_status(
                    stage="tts",
                    message=f"🎙️ Generating speech segment {current}/{total_speech}...",
                )
        
        return result

    async def run_segment_tasks_windowed() -> List[Tuple[int, AudioSegment, Dict[str, Any]]]:
        results_accumulator: List[Tuple[int, AudioSegment, Dict[str, Any]]] = []
        pending: Set[asyncio.Task] = set()
        next_index = 0

        def schedule_next() -> bool:
            nonlocal next_index
            if next_index >= len(segments):
                return False
            task = asyncio.create_task(
                process_segment_with_progress(next_index, segments[next_index])
            )
            pending.add(task)
            next_index += 1
            return True

        for _ in range(min(tts_concurrency, len(segments))):
            schedule_next()

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    results_accumulator.append(task.result())
                    schedule_next()
        except Exception:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise

        return results_accumulator

    print(
        f"[synthesize] Scheduling {len(segments)} segment tasks "
        f"with sliding concurrency={tts_concurrency}..."
    )
    results = await run_segment_tasks_windowed()
    
    print(f"[synthesize] All {len(results)} segment tasks completed. Sorting results...")
    sort_start = time.perf_counter()
    results.sort(key=lambda item: item[0])
    sort_elapsed = (time.perf_counter() - sort_start) * 1000
    print(f"[synthesize] Results sorted in {sort_elapsed:.1f}ms")
    
    # Emit completion message
    if emit_status:
        await emit_status(
            stage="tts_complete",
            message=f"✅ Generated {total_speech} speech segments. Preparing for concatenation...",
        )

    # Collect audio segments and log entries
    audio_chunks: List[AudioSegment] = []
    for idx, (_, audio_segment, log_entry) in enumerate(results):
        audio_chunks.append(audio_segment)
        generation_log.append(log_entry)

    original_duration_ms = len(original_audio)
    audio_format = _normalize_translate_output_format(response_format)
    final_duration_ms = 0
    unmixed_audio_bytes: Optional[bytes] = None
    
    # Use FFmpeg for fast concatenation and post-processing (much faster than pydub for large files)
    use_ffmpeg_concat = len(audio_chunks) > 100  # Use FFmpeg for larger files
    
    if use_ffmpeg_concat:
        print(f"[synthesize] Using FFmpeg for fast concatenation of {len(audio_chunks)} segments...")
        if emit_status:
            await emit_status(
                stage="concatenation",
                message=f"🚀 Using FFmpeg for fast concatenation ({len(audio_chunks)} segments)...",
            )
        
        concat_start = time.perf_counter()
        
        # Create temp directory for segment files
        temp_dir = tempfile.mkdtemp(prefix="synth_concat_")
        paths_to_cleanup: List[str] = []
        try:
            # Save segments to temp files and create concat list
            segment_files: List[str] = []
            
            print(f"[synthesize] Writing {len(audio_chunks)} segment files...")
            write_start = time.perf_counter()
            
            def _write_segments_to_files():
                """Write all segments to temp files - runs in thread pool"""
                files = []
                for idx, audio_seg in enumerate(audio_chunks):
                    seg_path = os.path.join(temp_dir, f"seg_{idx:05d}.wav")
                    audio_seg.export(seg_path, format="wav")
                    files.append(seg_path)
                    if (idx + 1) % 200 == 0:
                        print(f"[synthesize] Written {idx + 1}/{len(audio_chunks)} segment files...")
                return files
            
            segment_files = await _run_blocking(_write_segments_to_files)
            write_elapsed = (time.perf_counter() - write_start) * 1000
            print(f"[synthesize] Segment files written in {write_elapsed:.1f}ms")
            
            if emit_status:
                await emit_status(
                    stage="concatenation",
                    message=f"🔗 Running FFmpeg concat on {len(segment_files)} files...",
                )
            
            # Use shared concat helper (copy codec)
            concat_output_path = await _run_blocking(
                _ffmpeg_concat_to_tempfile,
                segment_files,
                output_format="wav",
                bitrate=None,
            )
            paths_to_cleanup.append(concat_output_path)
            ffmpeg_concat_elapsed = (time.perf_counter() - concat_start) * 1000
            print(f"[synthesize] FFmpeg concat complete in {ffmpeg_concat_elapsed:.1f}ms")
            
            # Track current WAV file path - avoid loading into memory unless necessary
            current_wav_path = concat_output_path
            combined_audio: Optional[AudioSegment] = None  # Only load when needed
            
            # Get duration using ffprobe (fast) instead of loading entire file
            def _get_wav_duration_ms(path: str) -> int:
                try:
                    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and result.stdout.strip():
                        return int(float(result.stdout.strip()) * 1000)
                except Exception:
                    pass
                # Fallback: load and check (slower)
                return -1
            
            concat_duration_ms = await _run_blocking(lambda: _get_wav_duration_ms(concat_output_path))
            if concat_duration_ms < 0:
                # Fallback: load to get duration
                combined_audio = await _load_audio_segment_from_path(concat_output_path)
                concat_duration_ms = len(combined_audio)
            
            concat_elapsed = (time.perf_counter() - concat_start) * 1000
            print(f"[synthesize] Total concatenation time: {concat_elapsed:.1f}ms. Duration: {concat_duration_ms / 1000:.1f}s")
            
            # Pad to original duration if needed
            if pad_to_original and concat_duration_ms < original_duration_ms:
                print(f"[synthesize] Padding audio from {concat_duration_ms}ms to {original_duration_ms}ms...")
                silence_duration = original_duration_ms - concat_duration_ms
                # Need to load for padding
                if combined_audio is None:
                    combined_audio = await _load_audio_segment_from_path(current_wav_path)
                combined_audio = combined_audio + _create_silence_segment(silence_duration, frame_rate, sample_width, channels)
                # Save padded audio
                padded_path = os.path.join(temp_dir, "padded.wav")
                await _run_blocking(lambda: combined_audio.export(padded_path, format="wav"))
                current_wav_path = padded_path
                paths_to_cleanup.append(padded_path)
                concat_duration_ms = original_duration_ms

            if return_unmixed_audio:
                print("[synthesize] Exporting dry translated vocal preview before backing mix...")
                unmixed_output_path = os.path.join(temp_dir, f"unmixed.{audio_format}")
                unmixed_export_cmd = ["ffmpeg", "-y", "-i", current_wav_path]
                unmixed_export_cmd += _ffmpeg_codec_args_for_format(audio_format, bitrate or TRANSLATE_DEFAULT_BITRATE)
                unmixed_export_cmd.append(unmixed_output_path)
                unmixed_result = await _run_blocking(
                    lambda: subprocess.run(unmixed_export_cmd, capture_output=True, text=True, timeout=600)
                )
                if unmixed_result.returncode == 0 and os.path.exists(unmixed_output_path):
                    with open(unmixed_output_path, "rb") as infile:
                        unmixed_audio_bytes = infile.read()
                    paths_to_cleanup.append(unmixed_output_path)
                else:
                    print(f"⚠️ Failed to export dry translated audio: {unmixed_result.stderr[:300]}")
            
            # Mix with backing track using FFmpeg if needed
            backing_applied = False
            if merge_with_backing and backing_track_audio is not None:
                if emit_status:
                    await emit_status(
                        stage="backing_merge",
                        message=f"🎵 Mixing with backing track using FFmpeg...",
                    )
                print(f"[synthesize] Mixing with backing track using FFmpeg...")
                merge_start = time.perf_counter()
                
                try:
                    # Save backing track to temp file
                    backing_path = os.path.join(temp_dir, "backing.wav")
                    
                    # Prepare backing track (need to load backing into memory for processing)
                    def _prepare_backing():
                        prepared_backing = (
                            backing_track_audio.set_frame_rate(frame_rate)
                            .set_sample_width(sample_width)
                            .set_channels(channels)
                        )
                        # Match duration
                        target_duration = concat_duration_ms
                        if len(prepared_backing) < target_duration:
                            if backing_track_source == "custom":
                                # Loop to fill
                                loops_needed = (target_duration // len(prepared_backing)) + 1
                                prepared_backing = prepared_backing * loops_needed
                            else:
                                # Pad with silence
                                prepared_backing = prepared_backing + AudioSegment.silent(duration=target_duration - len(prepared_backing))
                        prepared_backing = prepared_backing[:target_duration]
                        prepared_backing.export(backing_path, format="wav")
                        return backing_path
                    
                    await _run_blocking(_prepare_backing)
                    paths_to_cleanup.append(backing_path)
                    
                    # Mix using shared overlay helper (FFmpeg)
                    backing_vol = max(backing_volume_percent, MIN_GENERATED_VOLUME_PERCENT) / 100.0
                    vocals_vol = max(generated_volume_percent, MIN_GENERATED_VOLUME_PERCENT) / 100.0
                    mixed_output_path = await _run_blocking(
                        _ffmpeg_overlay_to_tempfile,
                        current_wav_path,
                        backing_path,
                        output_format="wav",
                        bitrate=None,
                        vocals_volume=vocals_vol,
                        backing_volume=backing_vol,
                    )
                    paths_to_cleanup.append(mixed_output_path)
                    current_wav_path = mixed_output_path
                    backing_applied = True
                    merge_elapsed = (time.perf_counter() - merge_start) * 1000
                    print(f"[synthesize] FFmpeg backing track mix complete in {merge_elapsed:.1f}ms")
                except Exception as merge_error:
                    print(f"⚠️ Failed to merge backing track: {merge_error}")
                    # Fall back to pydub overlay - need to load
                    print("[synthesize] Falling back to pydub overlay...")
                    if combined_audio is None:
                        combined_audio = await _load_audio_segment_from_path(current_wav_path)
                    prepared_backing = backing_track_audio.set_frame_rate(frame_rate).set_sample_width(sample_width).set_channels(channels)
                    if len(prepared_backing) < len(combined_audio):
                        prepared_backing = prepared_backing + AudioSegment.silent(duration=len(combined_audio) - len(prepared_backing))
                    prepared_backing = prepared_backing[:len(combined_audio)]
                    combined_audio = await _run_blocking(lambda: prepared_backing.overlay(combined_audio))
                    # Save back to path
                    mixed_output_path = os.path.join(temp_dir, "mixed.wav")
                    await _run_blocking(lambda: combined_audio.export(mixed_output_path, format="wav"))
                    paths_to_cleanup.append(mixed_output_path)
                    current_wav_path = mixed_output_path
                    backing_applied = True
            
            # Export final audio using FFmpeg - use current_wav_path directly!
            if emit_status:
                await emit_status(
                    stage="export",
                    message=f"💾 Exporting final audio ({concat_duration_ms / 1000:.1f}s) to {audio_format}...",
                )
            print(f"[synthesize] Exporting final audio using FFmpeg (from {os.path.basename(current_wav_path)})...")
            export_start = time.perf_counter()
            
            # Use current_wav_path directly - no need to re-export to final.wav!
            final_output_path = os.path.join(temp_dir, f"output.{audio_format}")
            
            export_cmd = ["ffmpeg", "-y", "-i", current_wav_path]
            export_cmd += _ffmpeg_codec_args_for_format(audio_format, bitrate or TRANSLATE_DEFAULT_BITRATE)
            export_cmd.append(final_output_path)
            
            export_result = await _run_blocking(
                lambda: subprocess.run(export_cmd, capture_output=True, text=True, timeout=600)
            )
            
            if export_result.returncode != 0:
                print(f"⚠️ FFmpeg export failed: {export_result.stderr[:300]}")
                # Fall back to pydub export - need to load if not already
                if combined_audio is None:
                    combined_audio = await _load_audio_segment_from_path(current_wav_path)
                buffer = BytesIO()
                await _run_blocking(lambda: combined_audio.export(buffer, format=audio_format, bitrate=bitrate or "192k"))
                audio_bytes = buffer.getvalue()
            else:
                with open(final_output_path, "rb") as f:
                    audio_bytes = f.read()
            
            export_elapsed = (time.perf_counter() - export_start) * 1000
            print(f"[synthesize] Export complete in {export_elapsed:.1f}ms ({len(audio_bytes) / 1024 / 1024:.1f} MB)")
            
            final_duration_ms = len(combined_audio) if combined_audio is not None else concat_duration_ms
            
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                print(f"⚠️ Failed to cleanup temp dir: {cleanup_error}")
            for path in paths_to_cleanup:
                _safe_remove_file(path)
    
    else:
        # For smaller files, use pydub (simpler and sufficient)
        print(f"[synthesize] Using pydub for concatenation of {len(audio_chunks)} segments...")
        concat_start = time.perf_counter()
        
        if emit_status:
            await emit_status(
                stage="concatenation",
                message=f"🔗 Concatenating {len(audio_chunks)} audio segments...",
            )
        
        # Simple sum for smaller files
        combined_audio = await _run_blocking(lambda: sum(audio_chunks, combined_audio))
        
        concat_elapsed = (time.perf_counter() - concat_start) * 1000
        print(f"[synthesize] Concatenation complete in {concat_elapsed:.1f}ms. Duration: {len(combined_audio) / 1000:.1f}s")

        original_duration_ms = len(original_audio)
        final_duration_ms = len(combined_audio)
        if pad_to_original and final_duration_ms < original_duration_ms:
            print(f"[synthesize] Padding audio from {final_duration_ms}ms to {original_duration_ms}ms...")
            combined_audio = combined_audio + _create_silence_segment(original_duration_ms - final_duration_ms, frame_rate, sample_width, channels)

        if return_unmixed_audio:
            print("[synthesize] Exporting dry translated vocal preview before backing mix...")
            export_kwargs: Dict[str, Any] = {}
            if audio_format == "mp3" and bitrate:
                export_kwargs["bitrate"] = bitrate

            def _blocking_unmixed_export():
                buffer = BytesIO()
                combined_audio.export(buffer, format=audio_format, **export_kwargs)
                return buffer.getvalue()

            unmixed_audio_bytes = await _run_blocking(_blocking_unmixed_export)

        backing_applied = False
        if merge_with_backing and backing_track_audio is not None:
            if emit_status:
                await emit_status(
                    stage="backing_merge",
                    message=f"🎵 Mixing with backing track ({len(backing_track_audio) / 1000:.1f}s)...",
                )
            print(f"[synthesize] Mixing with backing track ({len(backing_track_audio) / 1000:.1f}s)...")
            merge_start = time.perf_counter()
            try:
                def _blocking_backing_merge(audio_to_mix: AudioSegment) -> AudioSegment:
                    prepared_backing = (
                        backing_track_audio.set_frame_rate(frame_rate)
                        .set_sample_width(sample_width)
                        .set_channels(channels)
                    )
                    if abs(backing_volume_percent - DEFAULT_GENERATED_VOLUME_PERCENT) >= 0.01:
                        volume_factor = max(backing_volume_percent, MIN_GENERATED_VOLUME_PERCENT) / 100.0
                        gain_db = 20 * math.log10(volume_factor) if volume_factor > 0 else -120.0
                        prepared_backing = prepared_backing.apply_gain(gain_db)
                    prepared_backing = _match_segment_duration(
                        prepared_backing,
                        len(audio_to_mix),
                        frame_rate,
                        sample_width,
                        channels,
                        loop_fill=backing_track_source == "custom",
                        allow_trim=backing_track_source == "custom",
                    )
                    if backing_track_source == "custom":
                        fade_duration = min(2000, len(prepared_backing))
                        if fade_duration > 0:
                            prepared_backing = prepared_backing.fade_out(fade_duration)
                    return prepared_backing.overlay(audio_to_mix)
                
                combined_audio = await _run_blocking(_blocking_backing_merge, combined_audio)
                backing_applied = True
                merge_elapsed = (time.perf_counter() - merge_start) * 1000
                print(f"[synthesize] Backing track mixed in {merge_elapsed:.1f}ms")
            except Exception as merge_error:
                print(f"⚠️ Failed to merge translated audio with backing track: {merge_error}")

        # Export final audio
        if emit_status:
            await emit_status(
                stage="export",
                message=f"💾 Exporting final audio ({len(combined_audio) / 1000:.1f}s) to {audio_format}...",
            )
        print(f"[synthesize] Exporting final audio ({len(combined_audio) / 1000:.1f}s) to {audio_format}...")
        export_start = time.perf_counter()
        
        export_kwargs: Dict[str, Any] = {}
        if audio_format == "mp3" and bitrate:
            export_kwargs["bitrate"] = bitrate
        
        def _blocking_export():
            buffer = BytesIO()
            combined_audio.export(buffer, format=audio_format, **export_kwargs)
            return buffer.getvalue()
        
        audio_bytes = await _run_blocking(_blocking_export)
        
        export_elapsed = (time.perf_counter() - export_start) * 1000
        print(f"[synthesize] Export complete in {export_elapsed:.1f}ms ({len(audio_bytes) / 1024 / 1024:.1f} MB)")
        final_duration_ms = len(combined_audio)

    media_type = _audio_media_type(audio_format)

    metadata = {
        "dest_language": dest_language,
        "segment_count": len(segments),
        "speech_segment_count": sum(1 for s in segments if s["type"] == "speech"),
        "silence_segment_count": sum(1 for s in segments if s["type"] == "silence"),
        "original_duration_ms": original_duration_ms,
        "generated_duration_ms": final_duration_ms,
        "generation_log": generation_log,
        "padded_to_original": pad_to_original,
        "backing_volume_percent": backing_volume_percent,
    }
    if input_mime_type:
        metadata["input_mime_type"] = input_mime_type
    if clearvoice_settings:
        metadata["clearvoice"] = clearvoice_settings
    metadata["backing_track"] = {
        "available": backing_track_audio is not None,
        "merged": backing_applied,
        "volume_percent": backing_volume_percent,
        "source": backing_track_source,
    }
    if unmixed_audio_bytes:
        metadata["_unmixed_audio_bytes"] = unmixed_audio_bytes
    metadata["preserve_silence_audio"] = preserve_silence_audio
    metadata["generated_volume_percent"] = generated_volume_percent

    return audio_bytes, media_type, metadata


async def _gemini_transcribe_translate(
    audio_bytes: bytes,
    mime_type: str,
    dest_language: str,
    prompt_text: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    api_key_override: Optional[str] = None,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
    if genai is None:
        raise RuntimeError(
            "The google-genai package is required for translation. Install it with `pip install google-genai`."
        )

    api_key = (api_key_override or "").strip() or os.getenv(GEMINI_API_KEY_ENV_VAR) or os.getenv(GOOGLE_API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Neither {GEMINI_API_KEY_ENV_VAR} nor {GOOGLE_API_KEY_ENV_VAR} environment variables are set."
        )

    prompt = (prompt_text or "").strip() or TRANSLATION_PROMPT_TEMPLATE.format(dest_language=dest_language)
    model_name = (model_name or "").strip() or _get_gemini_model_name()

    if types is None:
        raise RuntimeError(
            "google-genai types module is unavailable. Ensure the `google-genai` package is installed and up to date."
        )

    client = _get_gemini_client(api_key)
    audio_hash, cache_key = _gemini_cache_key(
        audio_bytes,
        dest_language=dest_language,
        prompt_text=prompt,
        model_name=model_name,
    )
    cache_info: Dict[str, Any] = {
        "audio_md5": audio_hash,
        "hit": False,
        "force_refresh": False,
    }
    if force_refresh:
        cache_info["force_refresh"] = True
    else:
        cached = _load_gemini_cache_entry(cache_key)
        cached_segments = cached.get("segments") if cached else None
        if cached and isinstance(cached_segments, list):
            cache_info["hit"] = True
            cache_info["cache_file"] = os.path.basename(_gemini_cache_path(cache_key))
            cache_info["created_at"] = cached.get("created_at")
            print(f"♻️ Gemini cache hit for audio md5={audio_hash} (model={model_name}).")
            return (
                cached_segments,
                cached.get("speaker_profiles") or [],
                cached.get("raw_text"),
                cache_info,
            )

    user_content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )

    def _call_gemini() -> str:
        response = client.models.generate_content(
            model=model_name,
            contents=[user_content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=DEFAULT_GEMINI_TEMPERATURE,
                top_p=DEFAULT_GEMINI_TOP_P,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,
                    thinkingBudget = -1
                )
            ),
        )

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason:
                raise RuntimeError(f"Gemini blocked the prompt: {block_reason}")

        text = _extract_text_from_gemini_response(response)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    loop = asyncio.get_event_loop()
    max_attempts = 10
    retry_delay_seconds = 10
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_text = await loop.run_in_executor(executor, _call_gemini)
            break
        except Exception as exc:
            last_error = exc
            print(
                f"⚠️ Gemini call failed (attempt {attempt}/{max_attempts}): {exc}. "
                f"Retrying in {retry_delay_seconds}s..."
            )
            if attempt < max_attempts:
                await asyncio.sleep(retry_delay_seconds)
    else:
        raise RuntimeError(f"Gemini request failed after {max_attempts} attempts: {last_error}")

    segments, speaker_profiles = _parse_gemini_json(raw_text)
    cache_record = {
        "version": GEMINI_CACHE_VERSION,
        "created_at": time.time(),
        "audio_md5": audio_hash,
        "dest_language": dest_language,
        "model": model_name,
        "prompt_hash": hashlib.md5(prompt.encode("utf-8")).hexdigest(),
        "segments": segments,
        "speaker_profiles": speaker_profiles,
        "raw_text": raw_text,
    }
    cache_path = _write_gemini_cache_entry(cache_key, cache_record)
    if cache_path:
        cache_info["cache_file"] = os.path.basename(cache_path)
        cache_info["stored"] = True
        print(f"💾 Gemini cache stored for audio md5={audio_hash} (model={model_name}).")
    return segments, speaker_profiles, raw_text, cache_info

# Global TTS manager
class TTSManager:
    _instance = None
    _initialized = False
    
    def __init__(self):
        self.tts = None
        self.speaker_manager = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def initialize(self):
        if not self._initialized:
            try:
                print("🚀 Initializing IndexTTS2 vLLM v2...")
                
                # Check model directory
                if not os.path.exists(SETTINGS.model_dir):
                    raise FileNotFoundError(f"Model directory {SETTINGS.model_dir} does not exist")
                
                # Check required files
                required_files = [
                    "bpe.model",
                    "gpt.pth", 
                    "config.yaml",
                    "s2mel.pth",
                    "wav2vec2bert_stats.pt"
                ]
                
                for file in required_files:
                    file_path = os.path.join(SETTINGS.model_dir, file)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Required file {file_path} does not exist")
                
                # Initialize IndexTTS2
                self.tts = IndexTTS2(
                    model_dir=SETTINGS.model_dir,
                    is_fp16=SETTINGS.is_fp16,
                    use_torch_compile=SETTINGS.use_torch_compile,
                    gpu_memory_utilization=SETTINGS.gpu_memory_utilization
                )
                
                # Initialize speaker preset manager
                self.speaker_manager = initialize_preset_manager(self.tts)
                
                # Initialize speaker API wrapper
                global speaker_api
                speaker_api = SpeakerAPIWrapper(self.speaker_manager)
                
                self._initialized = True
                print("✅ IndexTTS2 vLLM v2 initialized successfully!")
                print(f"🎭 Speaker preset manager initialized with {len(self.speaker_manager.list_presets())} existing presets")
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to initialize IndexTTS2: {e}")
                traceback.print_exc()
                self._initialized = False
                return False
        return True
    
    def get_tts(self):
        if not self._initialized or self.tts is None:
            raise Exception("IndexTTS2 not initialized")
        return self.tts

    async def sleep_for_snapshot(self, level: int = 1):
        tts = self.get_tts()
        if not hasattr(tts, "sleep_vllm"):
            raise RuntimeError("IndexTTS2 does not expose vLLM sleep hooks")
        await tts.sleep_vllm(level=level)

    async def wake_from_snapshot(self):
        tts = self.get_tts()
        if not hasattr(tts, "wake_vllm"):
            raise RuntimeError("IndexTTS2 does not expose vLLM wake hooks")
        await tts.wake_vllm()
        self.refresh_post_snapshot_state()

    def refresh_post_snapshot_state(self):
        if self.tts is not None and hasattr(self.tts, "clear_runtime_caches"):
            self.tts.clear_runtime_caches()

        if self.speaker_manager is not None and hasattr(self.speaker_manager, "_load_presets"):
            self.speaker_manager.presets = self.speaker_manager._load_presets()
            if hasattr(self.speaker_manager, "_memory_cache"):
                valid_names = set(self.speaker_manager.presets.keys())
                for preset_name in list(self.speaker_manager._memory_cache.keys()):
                    if preset_name not in valid_names:
                        del self.speaker_manager._memory_cache[preset_name]
    
    def is_ready(self):
        return self._initialized and self.tts is not None

# Create global TTS manager
tts_manager = TTSManager.get_instance()

# Speaker Management Functions using existing SpeakerPresetManager
class SpeakerAPIWrapper:
    """Wrapper to adapt SpeakerPresetManager for FastAPI endpoints"""
    
    def __init__(self, preset_manager: SpeakerPresetManager):
        self.preset_manager = preset_manager
    
    async def add_speaker(
        self,
        speaker_name: str,
        audio_files: List[bytes],
        filenames: List[str],
        apply_enhancement: bool = False,
        apply_super_resolution: bool = False,
        enhancement_model_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Add a new speaker with audio files and optional ClearVoice processing."""
        try:
            # Check if speaker already exists - run in thread pool
            loop = asyncio.get_event_loop()
            clearvoice_requested = apply_enhancement or apply_super_resolution

            if clearvoice_requested and ClearVoice is None:
                return {
                    "status": "error",
                    "message": "ClearVoice package is required for enhancement or super-resolution. Install the `clearvoice` package to enable these options."
                }

            existing_presets = await loop.run_in_executor(executor, self.preset_manager.list_presets)
            
            if speaker_name in existing_presets:
                preset_info = existing_presets[speaker_name]
                if not _speaker_preview_exists(speaker_name):
                    audio_path = preset_info.get("audio_path")
                    if audio_path and os.path.exists(audio_path):
                        await _create_speaker_preview_mp3(audio_path, speaker_name)
                preview_available = _speaker_preview_exists(speaker_name)
                return {
                    "status": "success", 
                    "message": f"Speaker '{speaker_name}' already exists",
                    "info": "already_exists",
                    "audio_count": 1,  # SpeakerPresetManager uses single audio file
                    "description": preset_info.get("description", ""),
                    "created_at": preset_info.get("created_at", 0),
                    "preview_url": _speaker_preview_url(speaker_name) if preview_available else None,
                }
            
            # Save the first audio file (SpeakerPresetManager expects single file)
            if not audio_files:
                return {"status": "error", "message": "No audio files provided"}
            
            # Create temporary file for the first audio
            audio_data = audio_files[0]
            filename = filenames[0] if filenames else "audio.wav"
            
            # Save to temporary location
            temp_dir = Path("speaker_presets") / "temp"
            # Create directory in thread pool to avoid blocking
            await loop.run_in_executor(executor, lambda: temp_dir.mkdir(parents=True, exist_ok=True))
            
            temp_path = temp_dir / f"{speaker_name}_{filename}"
            await async_write_file(str(temp_path), audio_data)
            
            processed_paths: Set[str] = set()
            processed_paths.add(str(temp_path))
            final_audio_path = str(temp_path)

            try:
                # Cut audio to 10 seconds if it exceeds the limit
                cut_temp_path = await async_cut_audio_to_duration(str(temp_path), max_duration=10.0)
                processed_paths.add(cut_temp_path)
                final_audio_path = cut_temp_path
                
                if clearvoice_requested:
                    try:
                        final_audio_path, clearvoice_paths, _ = await apply_clearvoice_processing(
                            final_audio_path,
                            apply_enhancement,
                            apply_super_resolution,
                            enhancement_model_name=enhancement_model_name,
                        )
                        processed_paths.update(clearvoice_paths)
                    except Exception as cv_error:
                        return {"status": "error", "message": f"ClearVoice processing failed: {str(cv_error)}"}
                
                processed_paths.add(final_audio_path)

                description_parts = [
                    f"Added via API with {len(audio_files)} audio files (auto-cut to 10s)"
                ]
                if clearvoice_requested:
                    cv_features = []
                    if apply_enhancement:
                        effective_model = enhancement_model_name if enhancement_model_name in AVAILABLE_ENHANCEMENT_MODELS else DEFAULT_ENHANCEMENT_MODEL
                        cv_features.append(f"{effective_model} enhancement")
                    if apply_super_resolution:
                        cv_features.append("MossFormer2_SR_48K super-resolution")
                    description_parts.append("ClearVoice: " + " + ".join(cv_features))
                description = " | ".join(description_parts)
                
                # Use SpeakerPresetManager to add the speaker - run in thread pool to avoid blocking
                # This operation processes audio through TTS pipeline and can take several seconds
                success = await loop.run_in_executor(
                    executor,
                    self.preset_manager.add_speaker_preset,
                    speaker_name,
                    final_audio_path,
                    description
                )
                
                if success:
                    response: Dict[str, Any] = {
                        "status": "success", 
                        "message": f"Speaker '{speaker_name}' added successfully",
                        "info": "newly_added",
                        "audio_count": len(audio_files)
                    }
                    if clearvoice_requested:
                        response["clearvoice"] = {
                            "enhancement": apply_enhancement,
                            "super_resolution": apply_super_resolution
                        }
                    preview_result = await _create_speaker_preview_mp3(final_audio_path, speaker_name)
                    if preview_result:
                        response["preview_url"] = _speaker_preview_url(speaker_name)
                    return response
                else:
                    return {"status": "error", "message": f"Failed to add speaker '{speaker_name}'"}
            
            finally:
                # Clean up temporary files asynchronously
                try:
                    for cleanup_path in processed_paths:
                        await async_remove_file(cleanup_path)
                except:
                    pass
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to add speaker: {str(e)}"}
    
    async def delete_speaker(self, speaker_name: str) -> Dict[str, str]:
        """Delete a speaker"""
        try:
            # Run deletion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(executor, self.preset_manager.delete_preset, speaker_name)
            
            if success:
                await _remove_speaker_preview(speaker_name)
                return {
                    "status": "success",
                    "message": f"Speaker '{speaker_name}' deleted successfully"
                }
            else:
                return {"status": "error", "message": f"Speaker '{speaker_name}' not found"}
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to delete speaker: {str(e)}"}
    
    async def list_speakers(self) -> Dict[str, Any]:
        """List all speakers with metadata"""
        try:
            # Run the synchronous operation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            presets = await loop.run_in_executor(executor, self.preset_manager.list_presets)
            
            speaker_info = {}
            for speaker_name, preset_data in presets.items():
                # Calculate cache file size in thread pool to avoid blocking
                cache_file = preset_data.get('cache_file', '')
                total_size = 0
                if cache_file:
                    def get_file_size(filepath):
                        if os.path.exists(filepath):
                            return os.path.getsize(filepath)
                        return 0
                    total_size = await loop.run_in_executor(executor, get_file_size, cache_file)
                
                preview_available = _speaker_preview_exists(speaker_name)
                preview_url = _speaker_preview_url(speaker_name) if preview_available else None

                speaker_info[speaker_name] = {
                    "audio_count": 1,  # SpeakerPresetManager uses single audio file
                    "audio_files": [os.path.basename(preset_data.get('audio_path', ''))],
                    "total_size_mb": total_size / (1024 * 1024),
                    "description": preset_data.get('description', ''),
                    "created_at": preset_data.get('created_at', 0),
                    "last_used": preset_data.get('last_used', 0),
                    "preview_available": preview_available,
                    "preview_url": preview_url,
                }
            
            return {
                "status": "success",
                "speakers": speaker_info,
                "total_speakers": len(speaker_info)
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to list speakers: {str(e)}"}
    
    async def get_speaker_audio_paths(self, speaker_name: str) -> Optional[List[str]]:
        """Get audio paths for a specific speaker"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            presets = await loop.run_in_executor(executor, self.preset_manager.list_presets)
            
            if speaker_name in presets:
                audio_path = presets[speaker_name].get('audio_path', '')
                if audio_path:
                    # Check file existence in thread pool
                    exists = await loop.run_in_executor(executor, os.path.exists, audio_path)
                    if exists:
                        return [audio_path]
            return None
        except Exception as e:
            print(f"Error getting speaker audio paths: {e}")
            return None
    
    def speaker_exists(self, speaker_name: str) -> bool:
        """Check if a speaker exists - simple check, no async needed"""
        try:
            presets = self.preset_manager.list_presets()
            return speaker_name in presets
        except Exception as e:
            print(f"Error checking speaker existence: {e}")
            return False

# Global speaker API wrapper (will be initialized after TTS)
speaker_api = None

# API Models
class TranslateRequest(BaseModel):
    audio: Optional[str] = Field(default=None, description="Base64-encoded audio or download URL.")
    downloaded_video_id: Optional[str] = Field(
        default=None,
        description="Filename/id of a video downloaded through the WebUI video downloader.",
    )
    dest_language: str = Field(..., description="Target language for translation, e.g., 'English'.")
    audio_mime_type: Optional[str] = Field(default=None, description="MIME type of the audio, e.g., 'audio/wav'.")
    base_filename: Optional[str] = Field(
        default=None,
        description="Optional override for the base output filename (no extension).",
    )
    prompt: Optional[str] = Field(default=None, description="Optional custom prompt for Gemini.")
    custom_backing_audio: Optional[str] = Field(
        default=None,
        description="Optional base64 or URL reference for a custom backing track to mix instead of the extracted instrumental.",
    )
    custom_backing_audio_mime_type: Optional[str] = Field(
        default=None,
        description="MIME type for the custom backing track (e.g., 'audio/mp3').",
    )
    response_format: Optional[Literal["mp3", "wav", "flac", "aac", "opus", "ogg", "webm"]] = Field(
        default=None, description="Desired audio format for the translated output."
    )
    bitrate: Optional[str] = Field(
        default=None,
        description="Optional bitrate (e.g., '128k') when using lossy codecs such as MP3.",
    )
    enhance_voice: Optional[bool] = Field(
        default=False,
        description="Apply ClearVoice speech enhancement before translation.",
    )
    enhancement_model: Optional[str] = Field(
        default=None,
        description="ClearVoice speech enhancement model to use. Options: 'MossFormerGAN_SE_16K' (default, smaller/faster), 'FRCRN_SE_16K' (smaller/faster), 'MossFormer2_SE_48K' (larger, higher quality).",
    )
    super_resolution_voice: Optional[bool] = Field(
        default=False,
        description="Apply ClearVoice MossFormer2_SR_48K super-resolution before translation.",
    )
    audio_separator_enabled: Optional[bool] = Field(
        default=False,
        description="Enable audio-separator for vocal/instrumental separation before translation.",
    )
    audio_separator_model: Optional[str] = Field(
        default=None,
        description="Audio-separator model to use: 'fast', 'balance' (default), or 'quality'.",
    )
    clearvoice_parallel_enabled: Optional[bool] = Field(
        default=False,
        description="Enable experimental ClearVoice parallel chunk processing for faster separation.",
    )
    audio_separator_use_soundfile: Optional[bool] = Field(
        default=None,
        description="Use audio-separator's soundfile writer for stems (slower, lower-memory fallback).",
    )
    clearvoice_parallel_chunk_seconds: Optional[int] = Field(
        default=None,
        description="When parallel ClearVoice is enabled, chunk duration in seconds (default 180s).",
    )
    clearvoice_parallel_max_workers: Optional[int] = Field(
        default=None,
        description="When parallel ClearVoice is enabled, maximum concurrent worker processes (max 5).",
    )
    merge_backing_track: Optional[bool] = Field(
        default=False,
        description="When enabled, mix the regenerated speech back onto an instrumental backing track extracted during ClearVoice enhancement.",
    )
    segments_json: Optional[str] = Field(
        default=None,
        description="Optional JSON array of pre-generated Gemini-like segments to skip inference.",
    )
    min_speech_ms: Optional[int] = Field(
        default=None,
        description="Override the minimum speech segment duration (ms) when merging short segments.",
    )
    max_merge_ms: Optional[int] = Field(
        default=None,
        description="Override the maximum silence gap (ms) allowed when merging neighboring segments; use 0 to skip merging.",
    )
    gemini_model: Optional[str] = Field(
        default=None,
        description="Override the Gemini model used for transcription/translation.",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Provide a Gemini API key for this request if environment key is not set.",
    )
    translation_llm_model: Optional[str] = Field(
        default=None,
        description=(
            "Translation model for local transcription pipelines: "
            "tencent/Hy-MT2-1.8B, lightning-ai/gemma-4-31B-it, "
            "lightning-ai/gpt-oss-20b, "
            "lightning-ai/gpt-oss-120b, lightning-ai/minimax-m2.5, "
            "or another allowed local/API model."
        ),
    )
    force_gemini_regenerate: Optional[bool] = Field(
        default=False,
        description="When true, bypass the Gemini cache and force a fresh analysis.",
    )
    ignore_non_speech: Optional[bool] = Field(
        default=False,
        description="When true, ask Gemini to ignore non-speech vocalizations and background voices.",
    )
    preserve_silence_audio: Optional[bool] = Field(
        default=False,
        description="When true, reuse the original audio for any segments labeled as silence.",
    )
    generated_volume_percent: Optional[float] = Field(
        default=None,
        description="Adjust regenerated speech loudness before merging (percentage, default 100%).",
    )
    backing_volume_percent: Optional[float] = Field(
        default=None,
        description="Adjust backing track loudness when merging (percentage, default 100%).",
    )
    silence_volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Adjust preserved-silence loudness when keeping original audio (percentage, default 100%).",
    )
    reuse_session_id: Optional[str] = Field(
        default=None,
        description="Reuse a previous translate session (e.g., from chunk splitting) instead of uploading audio again.",
    )
    default_speaker_preset: Optional[str] = Field(
        default=None,
        description="Optional speaker preset to apply to all detected voices (leave empty to clone originals).",
    )
    default_emotion_weight: Optional[float] = Field(
        default=None,
        description="Emotion weight (0.0-1.0) when using the original emotion prompt for the default speaker.",
    )
    transcription_pipeline: Optional[str] = Field(
        default="qwen_omnivad",
        description="Transcription pipeline to use: 'gemini' (default), 'whisperx', 'qwen_omnivad' (Qwen3-ASR + OmniVAD), or 'parakeet' (NVIDIA Parakeet).",
    )
    whisperx_proxy_refiner: Optional[bool] = Field(
        default=False,
        description="When using transcription_pipeline='whisperx', enable the experimental speaker-aware proxy segment refiner.",
    )
    qwen_omnivad_enable_diarization: Optional[bool] = Field(
        default=True,
        description="When using transcription_pipeline='qwen_omnivad', enable diarization.",
    )
    qwen_omnivad_diarization_backend: Optional[Literal["auto", "pyannote", "sortformer"]] = Field(
        default="auto",
        description="Diarization backend for Qwen OmniVAD: auto, pyannote, or sortformer.",
    )
    qwen_omnivad_enable_forced_aligner: Optional[bool] = Field(
        default=True,
        description="When using transcription_pipeline='qwen_omnivad', enable Qwen3 ForcedAligner timestamps.",
    )
    qwen_omnivad_diarization_min_seconds: Optional[float] = Field(
        default=0.0,
        description="When using transcription_pipeline='qwen_omnivad', minimum span duration to split by diarization.",
    )
    qwen_omnivad_merge_gap_seconds: Optional[float] = Field(
        default=DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS,
        description="When using transcription_pipeline='qwen_omnivad', merge adjacent OmniVAD spans separated by this many seconds or less.",
    )


class VideoInfoRequest(BaseModel):
    url: str = Field(..., description="Video URL supported by yt-dlp.")


class VideoDownloadRequest(BaseModel):
    url: str = Field(..., description="Video URL supported by yt-dlp.")
    quality: Optional[str] = Field(default="best", description="Quality preset: best, 1080p, 720p, 480p, 360p, or worst.")


class VideoReplaceAudioRequest(BaseModel):
    downloaded_video_id: Optional[str] = Field(default=None, description="Downloaded video filename/id.")
    session_id: Optional[str] = Field(default=None, description="Translate session id; used to resolve the source video when available.")
    audio_file_name: Optional[str] = Field(default=None, description="Translated audio filename from /api/translate_outputs.")
    output_filename: Optional[str] = Field(default=None, description="Optional output video base filename.")
    subtitle_file_name: Optional[str] = Field(default=None, description="Optional SRT/VTT filename from /api/translate_outputs to burn into the rendered video.")
    embedded_subtitle_file_names: Optional[List[str]] = Field(
        default=None,
        description="Optional SRT/VTT subtitle filenames from /api/translate_outputs to embed as selectable MP4 subtitle tracks.",
    )
    audio_mode: Optional[Literal["translated", "original", "both"]] = Field(
        default="translated",
        description="Rendered MP4 audio: translated track, original source track, or both selectable tracks.",
    )


class SpeakerOverrideInput(BaseModel):
    preset_name: Optional[str] = Field(
        default=None,
        description="Speaker preset to use for the detected speaker label; leave empty to clone the original voice.",
    )
    use_emotion_prompt: Optional[bool] = Field(
        default=False,
        description="When True, feed the original segment audio as an emotion/style prompt while using the preset.",
    )
    emotion_weight: Optional[float] = Field(
        default=DEFAULT_EMOTION_WEIGHT,
        description="Emotion control weight (0.0 - 1.0) when using the original emotion prompt.",
    )
    volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override regenerated speech loudness for this speaker (percentage).",
    )


class TranslateSegmentInput(BaseModel):
    index: int
    type: Literal["speech", "silence"] = "speech"
    start_ms: int
    end_ms: int
    translated_text: Optional[str] = ""
    source_text: Optional[str] = ""
    generate: Optional[bool] = True
    speaker: Optional[str] = Field(default=None, description="Detected speaker label")
    volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override regenerated speech loudness for this segment (percentage).",
    )
    emotion_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override emotion prompt weight for this segment (0.0-1.0).",
    )


class SegmentPreviewRequest(BaseModel):
    session_id: str = Field(..., description="Active translate session identifier.")
    segment: TranslateSegmentInput = Field(..., description="Segment payload to preview.")
    generated_volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override the global regenerated volume percent for this preview.",
    )
    backing_volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override backing track volume percent for this preview.",
    )
    speaker_overrides: Optional[Dict[str, SpeakerOverrideInput]] = Field(
        default=None,
        description="Optional speaker overrides to apply only for this preview request.",
    )


class TranslateGenerateRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier returned from /api/translate_segments.")
    segments: List[TranslateSegmentInput] = Field(..., description="Segments to render, including edits and selection.")
    response_format: Optional[Literal["mp3", "wav", "flac", "aac", "opus", "ogg", "webm"]] = Field(
        default=None, description="Desired audio format for output. Defaults to the original request value."
    )
    bitrate: Optional[str] = Field(default=None, description="Optional bitrate for lossy codecs.")
    merge_backing_track: Optional[bool] = Field(
        default=None,
        description="Override whether to mix generated speech with the stored backing track (requires ClearVoice enhancement).",
    )
    generated_volume_percent: Optional[float] = Field(
        default=None,
        description="Override regenerated speech loudness before merging (percentage, default 100%).",
    )
    backing_volume_percent: Optional[float] = Field(
        default=None,
        description="Override backing-track loudness before merging (percentage, default 100%).",
    )
    silence_volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override preserved-silence loudness before merging (percentage, default 100%).",
    )
    speaker_overrides: Optional[Dict[str, SpeakerOverrideInput]] = Field(
        default=None,
        description="Optional mapping from detected speaker ids (speaker1, speaker2, …) to preset overrides.",
    )


class MergeChunksRequest(BaseModel):
    chunk_session_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific chunk session IDs to merge, ordered as provided.",
    )
    chunk_batch_id: Optional[str] = Field(
        default=None,
        description="Chunk batch identifier returned from /api/translate_split_audio.",
    )
    response_format: Optional[Literal["mp3", "wav", "flac", "aac", "opus", "ogg", "webm"]] = Field(
        default=None,
        description="Desired format for the merged output (defaults to MP3).",
    )
    bitrate: Optional[str] = Field(
        default=None,
        description="Optional bitrate (e.g., '128k') for lossy codecs.",
    )
    merge_backing_track: Optional[bool] = Field(
        default=None,
        description="Whether to mix the concatenated chunk vocals with their stored backing audio.",
    )


class ChunkBatchGenerateRequest(BaseModel):
    chunk_session_ids: List[str] = Field(
        ...,
        min_items=1,
        description="List of chunk session IDs to generate in parallel.",
    )
    dest_language: Optional[str] = Field(
        default=None,
        description="Override destination language; defaults to the stored chunk language.",
    )
    response_format: Optional[Literal["mp3", "wav", "flac", "aac", "opus", "ogg", "webm"]] = Field(
        default=None,
        description="Desired audio format for each generated chunk.",
    )
    bitrate: Optional[str] = Field(
        default=None,
        description="Optional bitrate (e.g., '128k') when using lossy codecs such as MP3.",
    )
    gemini_model: Optional[str] = Field(
        default=None,
        description="Override the Gemini model used for chunk transcription/translation.",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Provide a Gemini API key for this batch if environment defaults should be bypassed.",
    )
    translation_llm_model: Optional[str] = Field(
        default=None,
        description="Translation model for local transcription pipelines.",
    )
    transcription_pipeline: Optional[str] = Field(
        default=None,
        description="Transcription pipeline to use for selected chunk generation: 'gemini' (default), 'whisperx', 'qwen_omnivad', or 'parakeet'.",
    )
    whisperx_proxy_refiner: Optional[bool] = Field(
        default=None,
        description="When using transcription_pipeline='whisperx', enable the experimental speaker-aware proxy segment refiner.",
    )
    qwen_omnivad_enable_diarization: Optional[bool] = Field(
        default=None,
        description="When using transcription_pipeline='qwen_omnivad', enable diarization.",
    )
    qwen_omnivad_diarization_backend: Optional[Literal["auto", "pyannote", "sortformer"]] = Field(
        default=None,
        description="Diarization backend for Qwen OmniVAD: auto, pyannote, or sortformer.",
    )
    qwen_omnivad_enable_forced_aligner: Optional[bool] = Field(
        default=None,
        description="When using transcription_pipeline='qwen_omnivad', enable Qwen3 ForcedAligner timestamps.",
    )
    qwen_omnivad_diarization_min_seconds: Optional[float] = Field(
        default=None,
        description="When using transcription_pipeline='qwen_omnivad', minimum span duration to split by diarization.",
    )
    qwen_omnivad_merge_gap_seconds: Optional[float] = Field(
        default=None,
        description="When using transcription_pipeline='qwen_omnivad', merge adjacent OmniVAD spans separated by this many seconds or less.",
    )
    merge_backing_track: Optional[bool] = Field(
        default=None,
        description="When true, attempt to mix regenerated speech with the stored backing track if available.",
    )
    ignore_non_speech: Optional[bool] = Field(
        default=None,
        description="When true, ask Gemini to ignore non-speech vocalizations.",
    )
    preserve_silence_audio: Optional[bool] = Field(
        default=None,
        description="When true, reuse the original audio for silence segments.",
    )
    generated_volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override regenerated speech loudness before merging (percentage, default 100%).",
    )
    backing_volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override backing-track loudness when merging (percentage, default 100%).",
    )
    silence_volume_percent: Optional[float] = Field(
        default=None,
        ge=MIN_GENERATED_VOLUME_PERCENT,
        le=MAX_GENERATED_VOLUME_PERCENT,
        description="Override preserved-silence loudness before merging (percentage, default 100%).",
    )
    force_gemini_regenerate: Optional[bool] = Field(
        default=False,
        description="Force Gemini to reprocess audio even if cache entries exist.",
    )
    default_speaker_preset: Optional[str] = Field(
        default=None,
        description="Optional speaker preset to apply across all generated chunks.",
    )
    default_emotion_weight: Optional[float] = Field(
        default=None,
        description="Emotion weight (0.0-1.0) when cloning emotion prompts for the default speaker.",
    )


class CloneRequest(BaseModel):
    text: str = Field(..., description="The text to generate audio for.")
    reference_audio: Optional[str] = Field(default=None, description="Reference audio URL or base64")
    reference_text: Optional[str] = Field(default=None, description="Optional transcript")
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    temperature: float = Field(default=0.9)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)
    repetition_penalty: float = Field(default=1.0)
    max_tokens: int = Field(default=4096)
    length_threshold: int = Field(default=50)
    window_size: int = Field(default=50)
    stream: bool = Field(default=False)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")
    emotion_text: Optional[str] = Field(default="", description="Emotion description text for emotion control")
    emotion_weight: float = Field(default=0.6, description="Emotion control weight (0.0 to 1.0)")
    speech_length: int = Field(default=0, description="Target audio duration in milliseconds. If 0, uses default duration calculation.")
    diffusion_steps: int = Field(default=10, description="Number of diffusion steps for mel-spectrogram generation (1-50). Higher values improve quality but increase latency.")
    max_text_tokens_per_sentence: int = Field(default=120, ge=80, le=200, description="Maximum tokens per sentence for text splitting (80-200). Higher values = longer sentences but may impact quality.")

    @validator("response_format", pre=True, always=True)
    def _force_mp3(cls, value: Optional[str]) -> str:
        return "mp3"

class SpeakRequest(BaseModel):
    text: str = Field(..., description="The text to generate audio for.")
    name: Optional[str] = Field(default=None, description="The name of the voice character")
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    temperature: float = Field(default=0.9)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)
    repetition_penalty: float = Field(default=1.0)
    max_tokens: int = Field(default=4096)
    length_threshold: int = Field(default=50)
    window_size: int = Field(default=50)
    stream: bool = Field(default=False)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")
    emotion_text: Optional[str] = Field(default="", description="Emotion description text for emotion control")
    emotion_weight: float = Field(default=0.6, description="Emotion control weight (0.0 to 1.0)")
    speech_length: int = Field(default=0, description="Target audio duration in milliseconds. If 0, uses default duration calculation.")
    diffusion_steps: int = Field(default=10, description="Number of diffusion steps for mel-spectrogram generation (1-50). Higher values improve quality but increase latency.")
    max_text_tokens_per_sentence: int = Field(default=120, ge=80, le=200, description="Maximum tokens per sentence for text splitting (80-200). Higher values = longer sentences but may impact quality.")

    @validator("response_format", pre=True, always=True)
    def _force_mp3(cls, value: Optional[str]) -> str:
        return "mp3"

async def warmup_model():
    """Run warmup inferences to fully preload the model"""
    try:
        print("🔥 Running model warmup (2 inferences for full load)...")
        tts = tts_manager.get_tts()
        
        # First warmup inference
        warmup_audio_1 = os.path.join(current_dir, "examples", "voice_01.wav")
        warmup_text_1 = "你好！欢迎使用IndexTTS中文语音合成系统。这是一个功能强大的AI语音生成工具，能够准确处理中文语音合成任务。床前明月光，疑是地上霜。举头望明月，低头思故乡。这首《静夜思》是李白的名作，表达了诗人对故乡的深深思念之情。系统支持多种语音风格，让您的文本转换为自然流畅的语音。今天是2025年1月11日，时间是下午3点30分。这款产品的价格是12,999元，性价比很高。我的电话号码是138-8888-8888，欢迎联系。我正在使用IndexTTS和vLLM技术进行AI语音合成。This system supports both Chinese and English perfectly. 这个系统的RTF约为0.1，比原版快3倍！GPU memory utilization设置为85%。"
        
        # Check if first warmup audio exists
        if not os.path.exists(warmup_audio_1):
            print(f"⚠️ Warmup audio file not found: {warmup_audio_1}")
            return
        
        # Create temporary output file for first warmup
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            warmup_output_1 = tmp_file.name
        
        try:
            # Run first warmup inference
            print("🔥 Warmup 1/2: Modern text with voice_01.wav...")
            await tts.infer(
                spk_audio_prompt=warmup_audio_1,
                text=warmup_text_1,
                output_path=warmup_output_1,
                emo_audio_prompt=None,
                emo_alpha=0.6,
                emo_vector=None,
                use_emo_text=True,
                emo_text="兴奋",
                use_random=False,
                interval_silence=200,
                verbose=False,
                max_text_tokens_per_sentence=120,
                speaker_preset=None,
                speech_length=0,
                diffusion_steps=10
            )
            print("✅ Warmup 1/2 completed!")
        finally:
            # Clean up first temporary warmup file
            if os.path.exists(warmup_output_1):
                os.remove(warmup_output_1)
        
        # Second warmup inference
        warmup_audio_2 = os.path.join(current_dir, "examples", "voice_02.wav")
        warmup_text_2 = "人工智能是百年来最宏大的科技建设项目。它究竟是什么样子的？美国经济已经一分为二。一边是热火朝天的 AI 经济，另一边则是萎靡不振的消费经济。你可以在经济统计数据中看到这一点。上个季度，人工智能领域的支出增长超过了消费者支出的增长。如果没有 AI，美国的经济增长将会微不足道。你可以在股市中看到这一点。在过去两年里，股市增长的约 60% 来自与 AI 相关的公司，如微软、英伟达和 Meta。如果没有 AI 热潮，股市的回报率将惨不忍睹。"
        
        # Check if second warmup audio exists
        if not os.path.exists(warmup_audio_2):
            print(f"⚠️ Warmup audio file not found: {warmup_audio_2}")
            print("✅ Model warmup completed with 1/2 inferences")
            return
        
        # Create temporary output file for second warmup
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            warmup_output_2 = tmp_file.name
        
        try:
            # Run second warmup inference
            print("🔥 Warmup 2/2: Ancient poetry with voice_02.wav...")
            await tts.infer(
                spk_audio_prompt=warmup_audio_2,
                text=warmup_text_2,
                output_path=warmup_output_2,
                emo_audio_prompt=None,
                emo_alpha=0.6,
                emo_vector=None,
                use_emo_text=True,
                emo_text="无聊",
                use_random=False,
                interval_silence=200,
                verbose=False,
                max_text_tokens_per_sentence=120,
                speaker_preset=None,
                speech_length=0,
                diffusion_steps=10
            )
            print("✅ Warmup 2/2 completed!")
        finally:
            # Clean up second temporary warmup file
            if os.path.exists(warmup_output_2):
                os.remove(warmup_output_2)
        
        print("✅ Model warmup fully completed (2/2 inferences)!")
                
    except Exception as e:
        print(f"⚠️ Warmup failed (non-critical): {e}")
        traceback.print_exc()

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting IndexTTS vLLM v2 FastAPI WebUI...")
    await tts_manager.initialize()
    
    # Only run warmup inference when torch_compile is enabled
    # torch_compile benefits from warmup to compile optimized CUDA graphs
    if SETTINGS.use_torch_compile:
        print("🔥 Running warmup (--use_torch_compile enabled)...")
        await warmup_model()
    else:
        print("⏭️ Skipping warmup (--use_torch_compile not enabled)")
    
    yield
    # Shutdown (if needed)
    print("🔄 Shutting down IndexTTS vLLM v2...")
    # Shutdown the thread executor
    executor.shutdown(wait=True)

def create_app() -> FastAPI:
    return FastAPI(
        title="IndexTTS vLLM v2 FastAPI WebUI",
        description="Ultra-fast TTS with vLLM backend, speaker presets, and advanced translate/edit mode with Gemini integration",
        lifespan=lifespan,
    )


app = create_app()


def _require_internal_snapshot_token(request: Request) -> None:
    expected_token = os.environ.get("INDEXTTS_INTERNAL_TOKEN")
    provided_token = request.headers.get("X-IndexTTS-Internal-Token")
    if not expected_token or provided_token != expected_token:
        raise HTTPException(status_code=404, detail="Not found")


@app.post("/internal/snapshot/warmup")
async def internal_snapshot_warmup(request: Request):
    _require_internal_snapshot_token(request)
    if not tts_manager.is_ready():
        raise HTTPException(status_code=503, detail="IndexTTS2 is not initialized")
    await warmup_model()
    tts_manager.refresh_post_snapshot_state()
    return JSONResponse(content={"status": "ok", "action": "warmup"})


@app.post("/internal/snapshot/sleep")
async def internal_snapshot_sleep(request: Request, level: int = Query(1, ge=1, le=2)):
    _require_internal_snapshot_token(request)
    await tts_manager.sleep_for_snapshot(level=level)
    return JSONResponse(content={"status": "ok", "action": "sleep", "level": level})


@app.post("/internal/snapshot/wake")
async def internal_snapshot_wake(request: Request):
    _require_internal_snapshot_token(request)
    await tts_manager.wake_from_snapshot()
    return JSONResponse(content={"status": "ok", "action": "wake"})

# Web Interface - HTML file path for hot reload support
_HTML_UI_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_new.html")

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open(_HTML_UI_FILE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        # Inject dynamic configuration values
        html_content = html_content.replace("{{CHUNK_SPLIT_MIN_SILENCE_MS}}", str(CHUNK_SPLIT_MIN_SILENCE_MS))
        return html_content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index_new.html not found</h1><p>Please ensure index_new.html exists in the same directory as fastapi_webui_v2.py</p>",
            status_code=500
        )

# Utility endpoints
@app.post("/api/estimate_duration")
async def api_estimate_duration(request: Request):
    """API: Estimate speech duration from text"""
    try:
        data = await request.json()
        text = data.get("text", "")
        language = data.get("language", "auto")
        
        if not text or not text.strip():
            return JSONResponse(content={
                "status": "error",
                "message": "No text provided"
            })
        
        duration_ms = estimate_speech_duration(text, language)
        duration_s = duration_ms / 1000.0
        
        # Detect language for display
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        detected_lang = "Chinese" if chinese_chars / max(len(text), 1) > 0.3 else "English"
        
        return JSONResponse(content={
            "status": "success",
            "duration_ms": duration_ms,
            "duration_s": round(duration_s, 1),
            "detected_language": detected_lang,
            "char_count": len(text)
        })
        
    except Exception as e:
        print(f"❌ Error estimating duration: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/api/clear_outputs")
async def api_clear_outputs():
    """API: Clear all generated output files"""
    try:
        outputs_dir = "outputs"
        
        if not os.path.exists(outputs_dir):
            return {
                "status": "success",
                "message": "Outputs directory does not exist",
                "files_deleted": 0,
                "space_freed_mb": 0
            }
        
        # Count files and size before deletion
        files_deleted = 0
        total_size = 0
        
        # Remove all files in outputs directory and subdirectories
        # Run file deletion in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def delete_files_sync():
            deleted = 0
            size = 0
            for root, dirs, files in os.walk(outputs_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted += 1
                        size += file_size
                        print(f"🗑️ Deleted: {file_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file_path}: {e}")
            return deleted, size
        
        files_deleted, total_size = await loop.run_in_executor(executor, delete_files_sync)
        
        space_freed_mb = total_size / (1024 * 1024)
        
        return {
            "status": "success",
            "message": f"Successfully cleared outputs directory",
            "files_deleted": files_deleted,
            "space_freed_mb": round(space_freed_mb, 2)
        }
        
    except Exception as e:
        print(f"❌ Error clearing outputs: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to clear outputs: {str(e)}"
        }


@app.get("/api/prompt_templates")
async def api_prompt_templates():
    """Return the default Gemini prompt templates so users can run them elsewhere."""
    return {
        "translation": TRANSLATION_PROMPT_TEMPLATE,
        "transcription": TRANSCRIPTION_PROMPT_TEMPLATE,
        "ignore_non_speech_instruction": IGNORE_NON_SPEECH_PROMPT_SUFFIX,
    }


def _normalize_stable_audio_output_format(value: Any) -> str:
    audio_format = str(value or "mp3").strip().lower()
    return audio_format if audio_format in STABLE_AUDIO3_OUTPUT_FORMATS else "mp3"


async def _save_stable_audio_upload(upload: Any, prefix: str) -> Optional[str]:
    if upload is None or not hasattr(upload, "read"):
        return None
    filename = getattr(upload, "filename", "") or ""
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix) as tmp:
        tmp_path = tmp.name
    try:
        data = await upload.read()
        await async_write_file(tmp_path, data)
        return tmp_path
    finally:
        close_fn = getattr(upload, "close", None)
        if close_fn is not None:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result


async def _parse_stable_audio_generate_request(
    request: Request,
) -> Tuple[StableAudio3GenerateOptions, str, List[str], int]:
    content_type = (request.headers.get("content-type") or "").lower()
    temp_paths: List[str] = []
    payload: Dict[str, Any] = {}
    init_upload = None
    inpaint_upload = None

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                continue
            payload[key] = value
        init_upload = form.get("init_audio_file") or form.get("init_audio")
        inpaint_upload = form.get("inpaint_audio_file") or form.get("inpaint_audio")
    else:
        try:
            raw_payload = await request.json()
        except Exception:
            raw_payload = {}
        if isinstance(raw_payload, dict):
            payload = raw_payload

    init_audio_path = await _save_stable_audio_upload(init_upload, "stable_audio_init_")
    if init_audio_path:
        temp_paths.append(init_audio_path)
    inpaint_audio_path = await _save_stable_audio_upload(inpaint_upload, "stable_audio_inpaint_")
    if inpaint_audio_path:
        temp_paths.append(inpaint_audio_path)

    variant_key = stable_audio3_manager.normalize_variant_key(payload.get("variant_key") or payload.get("model"))
    variant_info = next(
        (model for model in stable_audio3_manager.list_models() if model.get("key") == variant_key),
        None,
    )
    default_duration = int((variant_info or {}).get("default_duration") or 60)
    output_format = _normalize_stable_audio_output_format(payload.get("response_format") or payload.get("output_format"))
    batch_count = _coerce_positive_int(
        payload.get("batch_count") or payload.get("num_outputs") or payload.get("count"),
        1,
        min_value=1,
        max_value=4,
    )

    options = StableAudio3GenerateOptions(
        variant_key=variant_key,
        prompt=str(payload.get("prompt") or ""),
        negative_prompt=str(payload.get("negative_prompt") or ""),
        duration=_coerce_positive_int(payload.get("duration"), default_duration, min_value=1, max_value=3600),
        steps=_coerce_positive_int(payload.get("steps"), 8, min_value=1, max_value=500),
        cfg_scale=_coerce_positive_float(payload.get("cfg_scale"), 1.0, min_value=0.0, max_value=25.0),
        sampler_type=stable_audio3_manager.normalize_sampler(payload.get("sampler_type")),
        seed=_coerce_positive_int(payload.get("seed"), 0, min_value=0),
        sigma_max=_coerce_positive_float(payload.get("sigma_max"), 1.0, min_value=0.0, max_value=1.0),
        apg_scale=_coerce_positive_float(payload.get("apg_scale"), 1.0, min_value=0.0, max_value=1.0),
        duration_padding_sec=_coerce_positive_float(
            payload.get("duration_padding_sec"),
            6.0,
            min_value=0.0,
            max_value=30.0,
        ),
        cut_to_seconds_total=_coerce_to_bool(payload.get("cut_to_seconds_total", True)),
        init_audio_path=init_audio_path,
        init_noise_level=_coerce_positive_float(payload.get("init_noise_level"), 0.9, min_value=0.01, max_value=1.0),
        inpaint_audio_path=inpaint_audio_path,
        mask_start_sec=_coerce_positive_float(payload.get("mask_start_sec"), 0.0, min_value=0.0),
        mask_end_sec=_coerce_positive_float(payload.get("mask_end_sec"), 0.0, min_value=0.0),
        output_dir=str((ROOT_OUTPUT_DIR / "stable_audio").resolve()),
    )
    return options, output_format, temp_paths, batch_count


async def _stable_audio_generation_to_payload(
    result_path: str,
    response_format: str,
    filename_stem: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        audio_bytes = await _read_generated_audio_bytes(result_path, response_format)
    finally:
        await async_remove_file(result_path)
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    return {
        "filename": f"{filename_stem}.{response_format}",
        "media_type": _audio_media_type(response_format),
        "audio_base64": encoded_audio,
        "metadata": metadata,
    }


@app.get("/api/stable-audio/status")
async def api_stable_audio_status():
    """API: Stable Audio 3 dependency, checkpoint, and loaded-model status."""
    return JSONResponse(content={"status": "success", **stable_audio3_manager.status()})


@app.get("/api/stable-audio/models")
async def api_stable_audio_models():
    """API: List Stable Audio 3 variants and checkpoint readiness."""
    return JSONResponse(
        content={
            "status": "success",
            "default_variant": STABLE_AUDIO3_DEFAULT_VARIANT,
            "samplers": list(STABLE_AUDIO3_SAMPLERS),
            "models": stable_audio3_manager.list_models(),
        }
    )


@app.post("/api/stable-audio/unload")
async def api_stable_audio_unload(request: Request):
    """API: Unload Stable Audio 3 models from GPU memory."""
    payload: Dict[str, Any] = {}
    if _request_has_json_body(request):
        try:
            raw_payload = await request.json()
            if isinstance(raw_payload, dict):
                payload = raw_payload
        except Exception:
            payload = {}
    removed = await _run_blocking(stable_audio3_manager.unload, payload.get("variant_key"))
    return JSONResponse(content={"status": "success", "unloaded": removed})


@app.post("/api/stable-audio/generate")
async def api_stable_audio_generate(request: Request):
    """API: Generate music or sound effects with Stable Audio 3."""
    temp_paths: List[str] = []
    generated_paths: List[str] = []
    try:
        options, response_format, temp_paths, batch_count = await _parse_stable_audio_generate_request(request)
        print(
            f"[stable-audio] Generating {batch_count}x {options.variant_key} "
            f"{options.duration}s/{options.steps} steps: {options.prompt[:80]!r}"
        )
        base_seed = int(options.seed or 0)
        generation_options = [
            replace(options, seed=(base_seed + idx if base_seed > 0 else 0))
            for idx in range(batch_count)
        ]
        results = await _run_blocking(stable_audio3_manager.generate_many, generation_options)
        generated_paths = [result.output_path for result in results]

        if batch_count == 1:
            filename_stem = f"stable_audio_{results[0].variant_key}_{int(time.time())}"
            generated_paths = []
            return await _generated_audio_attachment_response(
                results[0].output_path,
                response_format,
                filename_stem=filename_stem,
            )

        generated_at = int(time.time())
        payload_items = []
        for idx, result in enumerate(results, start=1):
            filename_stem = f"stable_audio_{result.variant_key}_{generated_at}_{idx}"
            payload_items.append(
                await _stable_audio_generation_to_payload(
                    result.output_path,
                    response_format,
                    filename_stem,
                    {
                        "index": idx,
                        "variant_key": result.variant_key,
                        "duration_seconds": result.duration_seconds,
                        "elapsed_seconds": result.elapsed_seconds,
                        "seed": result.seed,
                        "source": result.source,
                    },
                )
            )
            generated_paths = [path for path in generated_paths if path != result.output_path]

        return JSONResponse(
            content={
                "status": "success",
                "count": len(payload_items),
                "response_format": response_format,
                "items": payload_items,
            }
        )
    except Exception as exc:
        print(f"Stable Audio 3 generation failed: {exc}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )
    finally:
        for temp_path in temp_paths:
            await async_remove_file(temp_path)
        for generated_path in generated_paths:
            if generated_path and os.path.exists(generated_path):
                await async_remove_file(generated_path)


# ---------------------------------------------------------------------------
# Cookie management API endpoints
# ---------------------------------------------------------------------------

class CookieImportCurlRequest(BaseModel):
    curl_text: str = Field(..., description="cURL command string copied from browser DevTools.")
    domain: Optional[str] = Field(default=None, description="Override auto-detected domain.")


@app.get("/api/cookies")
async def api_cookies_list():
    """API: List all saved cookie domains with cookie counts."""
    sites = _get_saved_cookie_sites()
    return JSONResponse(content={
        "status": "ok",
        "sites": {domain: {"count": count} for domain, count in sites.items()},
        "cookie_dirs": _candidate_cookie_dirs(),
    })


@app.get("/api/video_ytdlp_diagnostics")
async def api_video_ytdlp_diagnostics(url: Optional[str] = Query(default=None)):
    """API: Show what yt-dlp-related setup the running FastAPI process can see."""
    _ensure_yt_dlp()

    def shell_command_output(command: str) -> Dict[str, Any]:
        try:
            process = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return {
                "returncode": process.returncode,
                "stdout": (process.stdout or "").strip(),
                "stderr": (process.stderr or "").strip(),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def package_version(name: str) -> Optional[str]:
        try:
            import importlib.metadata as importlib_metadata

            return importlib_metadata.version(name)
        except Exception:
            return None

    def executable_version(executable: Optional[str]) -> Optional[str]:
        if not executable:
            return None
        try:
            process = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version_text = (process.stdout or process.stderr or "").strip()
            return version_text or None
        except Exception as exc:
            return f"error: {exc}"

    def port_open(host: str, port: int) -> bool:
        try:
            import socket

            with socket.create_connection((host, port), timeout=2):
                return True
        except Exception:
            return False

    cookie_file = _get_cookie_path_for_url(url or "") if url else None
    cookie_count: Optional[int] = None
    if cookie_file and os.path.exists(cookie_file):
        try:
            with open(cookie_file, "r", encoding="utf-8") as fh:
                cookie_count = len([line for line in fh if line.strip() and not line.startswith("#")])
        except Exception:
            cookie_count = None

    common_opts = _yt_dlp_common_opts()
    common_opts_payload = dict(common_opts)
    if isinstance(common_opts_payload.get("remote_components"), set):
        common_opts_payload["remote_components"] = sorted(common_opts_payload["remote_components"])

    js_runtime_opts = (common_opts.get("js_runtimes") or {}) if isinstance(common_opts.get("js_runtimes"), dict) else {}
    node_path = (js_runtime_opts.get("node") or {}).get("path") or shutil.which("node")
    deno_path = (js_runtime_opts.get("deno") or {}).get("path") or shutil.which("deno")
    nvm_node_candidates = [str(path) for path in Path("/home").glob("*/.nvm/versions/node/*/bin/node")]
    return JSONResponse(content={
        "status": "ok",
        "python": sys.executable,
        "cwd": os.getcwd(),
        "path": os.environ.get("PATH", ""),
        "env": {
            "YTDLP_NODE_PATH": os.environ.get("YTDLP_NODE_PATH"),
            "NODE_BINARY": os.environ.get("NODE_BINARY"),
            "YTDLP_JS_RUNTIMES": os.environ.get("YTDLP_JS_RUNTIMES"),
        },
        "shell": {
            "command_v_node": shell_command_output("command -v node"),
            "type_node": shell_command_output("type -a node"),
            "node_exec_path": shell_command_output("node -p 'process.execPath'"),
        },
        "app_dir": str(APP_DIR),
        "root_output_dir": str(ROOT_OUTPUT_DIR),
        "cookies_dir": COOKIES_DIR,
        "candidate_cookie_dirs": _candidate_cookie_dirs(),
        "url_cookie_file": cookie_file,
        "url_cookie_count": cookie_count,
        "node": {
            "path": node_path,
            "version": executable_version(node_path),
            "nvm_candidates": nvm_node_candidates,
        },
        "deno": {"path": deno_path, "version": executable_version(deno_path)},
        "bgutil_http_provider": {"url": "http://127.0.0.1:4416", "port_open": port_open("127.0.0.1", 4416)},
        "yt_dlp": {
            "version": getattr(yt_dlp, "__version__", None) if yt_dlp is not None else None,
            "yt_dlp_ejs_version": package_version("yt-dlp-ejs"),
            "bgutil_provider_version": package_version("bgutil-ytdlp-pot-provider"),
            "common_opts": common_opts_payload,
        },
    })


@app.post("/api/cookies/import_curl")
async def api_cookies_import_curl(payload: CookieImportCurlRequest):
    """API: Import cookies from a cURL command and save for the detected domain."""
    curl_text = (payload.curl_text or "").strip()
    if not curl_text:
        return _status_error("cURL text is required.")

    domain = (payload.domain or "").strip()
    if not domain:
        domain = _extract_domain_from_curl(curl_text)
    if domain == "unknown":
        return _status_error("Could not detect domain. Please provide it explicitly.")

    cookies = _parse_curl_cookies(curl_text)
    if not cookies:
        return _status_error("No cookies found in cURL command.")

    try:
        cookie_lines = [
            "# Netscape HTTP Cookie File",
            f"# Domain: {domain}",
            "# This is a generated file! Do not edit.",
            "",
        ]
        for name, value in cookies.items():
            cookie_lines.append(f".{domain}\tTRUE\t/\tTRUE\t0\t{name}\t{value}")

        filename = domain.replace(".", "_") + "_cookies.txt"
        filepath = os.path.join(COOKIES_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(cookie_lines))

        return JSONResponse(content={
            "status": "ok",
            "message": f"Imported {len(cookies)} cookies for {domain}.",
            "domain": domain,
            "cookie_count": len(cookies),
            "cookie_file": filepath,
        })
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to save cookies: {str(exc)}", status_code=500)


@app.post("/api/cookies/upload")
async def api_cookies_upload(
    file: UploadFile = File(...),
    domain: str = Form(...),
):
    """API: Upload a cookies.txt file directly and save for the specified domain."""
    domain = (domain or "").strip().lower()
    if not domain:
        return _status_error("Domain is required.")
    if domain.startswith("www."):
        domain = domain[4:]

    try:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        return _status_error(f"Failed to read file: {str(exc)}")

    if not content.strip():
        return _status_error("Uploaded file is empty.")

    try:
        filename = domain.replace(".", "_") + "_cookies.txt"
        filepath = os.path.join(COOKIES_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        cookie_count = len([l for l in content.splitlines() if l.strip() and not l.startswith("#")])

        return JSONResponse(content={
            "status": "ok",
            "message": f"Cookies saved for {domain}.",
            "domain": domain,
            "cookie_count": cookie_count,
            "cookie_file": filepath,
        })
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to save cookies file: {str(exc)}", status_code=500)


@app.delete("/api/cookies/{domain}")
async def api_cookies_delete(domain: str):
    """API: Delete saved cookies for a specific domain."""
    domain = (domain or "").strip().lower()
    if not domain:
        return _status_error("Domain is required.")
    filename = domain.replace(".", "_") + "_cookies.txt"
    filepath = os.path.join(COOKIES_DIR, filename)
    if not os.path.exists(filepath):
        return _status_error(f"No cookies found for {domain}.", status_code=404)
    try:
        os.remove(filepath)
        return JSONResponse(content={
            "status": "ok",
            "message": f"Cookies for {domain} deleted.",
            "domain": domain,
        })
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to delete cookies: {str(exc)}", status_code=500)


# ---------------------------------------------------------------------------
# Downloaded video endpoints
# ---------------------------------------------------------------------------

@app.get("/api/downloaded_videos")
async def api_downloaded_videos():
    """API: List videos downloaded through the WebUI downloader."""
    return JSONResponse(
        content={
            "status": "ok",
            "videos": _list_downloaded_video_entries(),
            "download_dir": VIDEO_DOWNLOAD_DIR,
        }
    )


@app.delete("/api/downloaded_videos/{filename}")
async def api_delete_downloaded_video(filename: str):
    """API: Delete a downloaded video file."""
    try:
        file_path = _downloaded_video_path(filename)
    except ValueError as exc:
        return _status_error(str(exc), status_code=400)
    if not os.path.exists(file_path):
        return _status_error("Downloaded video not found.", status_code=404)
    try:
        safe_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        os.remove(file_path)
        # Also remove cached extracted audio if any
        audio_cache = os.path.join(VIDEO_AUDIO_CACHE_DIR, os.path.splitext(safe_name)[0] + ".mp3")
        if os.path.exists(audio_cache):
            os.remove(audio_cache)
        return JSONResponse(content={
            "status": "ok",
            "message": f"Deleted {safe_name}.",
            "filename": safe_name,
            "freed_mb": round(file_size / (1024 * 1024), 2),
        })
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to delete video: {str(exc)}", status_code=500)


@app.get("/api/downloaded_videos/{filename}")
async def api_downloaded_video_file(filename: str):
    """API: Serve a downloaded video for preview/download."""
    try:
        file_path = _downloaded_video_path(filename)
    except ValueError as exc:
        return _status_error(str(exc), status_code=400)
    if not os.path.exists(file_path):
        return _status_error("Downloaded video not found.", status_code=404)
    safe_name = os.path.basename(file_path)
    return FileResponse(
        file_path,
        media_type=_guess_media_type_from_extension(safe_name),
        filename=safe_name,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/downloaded_videos/{filename}/snapshot")
async def api_downloaded_video_snapshot(filename: str):
    """API: Serve a generated snapshot image for a downloaded video."""
    try:
        file_path = _downloaded_video_path(filename)
    except ValueError as exc:
        return _status_error(str(exc), status_code=400)
    if not os.path.exists(file_path):
        return _status_error("Downloaded video not found.", status_code=404)
    try:
        snapshot_path = await _run_blocking(_generate_video_snapshot_sync, file_path)
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to generate video snapshot: {str(exc)}", status_code=500)
    return FileResponse(
        snapshot_path,
        media_type="image/jpeg",
        filename=os.path.basename(snapshot_path),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/downloaded_videos/{filename}/audio")
async def api_downloaded_video_audio(filename: str):
    """API: Extract and serve MP3 audio from a downloaded video."""
    try:
        video_path = _downloaded_video_path(filename)
    except ValueError as exc:
        return _status_error(str(exc), status_code=400)
    if not os.path.exists(video_path):
        return _status_error("Downloaded video not found.", status_code=404)
    try:
        audio_path = await _run_blocking(_extract_audio_from_downloaded_video_sync, video_path)
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to extract video audio: {str(exc)}", status_code=500)
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=os.path.basename(audio_path),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/video_info")
async def api_video_info(payload: VideoInfoRequest):
    """API: Fetch basic yt-dlp metadata before downloading."""
    _ensure_yt_dlp()
    if yt_dlp is None:
        return _status_error("yt-dlp is not installed. Install it with `pip install yt-dlp`.", status_code=500)
    url = (payload.url or "").strip()
    if not url:
        return _status_error("Video URL is required.")

    def extract_info() -> Dict[str, Any]:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "noplaylist": True,
            "check_formats": False,
            **_yt_dlp_common_opts(),
            **_get_cookie_opts_for_url(url),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            if not _is_ytdl_format_unavailable_error(exc):
                raise
            # Metadata can still be useful when yt-dlp's default format picker
            # cannot resolve a downloadable format. This mirrors downloader.py's
            # broad info fetch, while skipping the processing step that raises.
            raw_opts = {**opts, "format": None}
            with yt_dlp.YoutubeDL(raw_opts) as ydl:
                info = ydl.extract_info(url, download=False, process=False)
        if not isinstance(info, dict):
            raise RuntimeError("yt-dlp returned no video metadata.")
        formats: List[Dict[str, Any]] = []
        raw_format_count = 0
        for fmt in info.get("formats") or []:
            if not isinstance(fmt, dict):
                continue
            raw_format_count += 1
            vcodec = fmt.get("vcodec") or "none"
            acodec = fmt.get("acodec") or "none"
            if vcodec == "none" or not fmt.get("format_id") or not _raw_format_is_direct(fmt):
                continue
            height = fmt.get("height")
            formats.append(
                {
                    "format_id": fmt.get("format_id"),
                    "ext": fmt.get("ext"),
                    "resolution": fmt.get("resolution") or (f"{height}p" if height else ""),
                    "height": height,
                    "fps": fmt.get("fps"),
                    "vcodec": vcodec,
                    "acodec": acodec,
                    "filesize": fmt.get("filesize") or fmt.get("filesize_approx") or 0,
                    "tbr": fmt.get("tbr") or 0,
                    "note": fmt.get("format_note") or "",
                }
            )
        formats.sort(key=lambda item: (item.get("height") or 0, item.get("tbr") or 0), reverse=True)
        return {
            "status": "ok",
            "title": info.get("title") or "Untitled",
            "id": info.get("id"),
            "webpage_url": info.get("webpage_url") or url,
            "thumbnail": info.get("thumbnail"),
            "duration_seconds": info.get("duration"),
            "duration_label": _format_video_duration(info.get("duration")),
            "formats": formats[:30],
            "raw_format_count": raw_format_count,
            "downloadable_format_count": len(formats),
            "format_warning": (
                f"Video metadata loaded, but yt-dlp did not expose downloadable stream URLs. {_ytdlp_setup_hint()}"
                if raw_format_count and not formats else ""
            ),
        }

    try:
        return JSONResponse(content=await _run_blocking(extract_info))
    except Exception as exc:
        traceback.print_exc()
        error_msg = _clean_ytdl_error(exc)
        lower_error = error_msg.lower()
        if "confirm you are not a bot" in lower_error or "sign in" in lower_error:
            error_msg += f" ({_ytdlp_auth_hint()})"
        elif "format" in lower_error:
            error_msg += f" ({_ytdlp_setup_hint()})"
        return _status_error(f"Failed to fetch video info: {error_msg}", status_code=400)


@app.post("/api/video_download")
async def api_video_download(payload: VideoDownloadRequest):
    """API: Download a video through yt-dlp and stream progress events."""
    _ensure_yt_dlp()
    if yt_dlp is None:
        return _status_error("yt-dlp is not installed. Install it with `pip install yt-dlp`.", status_code=500)
    url = (payload.url or "").strip()
    if not url:
        return _status_error("Video URL is required.")
    quality_raw = (payload.quality or "best").strip()
    quality_key = quality_raw.lower()
    if quality_key in VIDEO_DOWNLOAD_FORMATS:
        format_selector = VIDEO_DOWNLOAD_FORMATS[quality_key]
    else:
        format_selector = quality_raw

    async def download_stream():
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(event_type: str, **event_payload: Any) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "event": event_type,
                    "timestamp": time.time(),
                    **event_payload,
                },
            )

        def download_sync() -> None:
            start_time = time.time()
            info_result: Optional[Dict[str, Any]] = None
            try:
                emit("status", message="Starting video download...", quality=quality_key)

                def progress_hook(data: Dict[str, Any]) -> None:
                    status = data.get("status")
                    if status == "downloading":
                        total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
                        downloaded = data.get("downloaded_bytes") or 0
                        percent = round((downloaded / total) * 100, 2) if total else None
                        emit(
                            "progress",
                            message="Downloading video...",
                            percent=percent,
                            downloaded_bytes=downloaded,
                            total_bytes=total,
                            speed=data.get("speed"),
                            eta=data.get("eta"),
                        )
                    elif status == "finished":
                        emit("status", message="Download finished; post-processing...")

                base_ydl_opts = {
                    "outtmpl": os.path.join(VIDEO_DOWNLOAD_DIR, "%(title).180B [%(id)s].%(ext)s"),
                    "merge_output_format": "mp4",
                    "noplaylist": True,
                    "quiet": True,
                    "no_warnings": True,
                    "progress_hooks": [progress_hook],
                    **_yt_dlp_common_opts(),
                    **_get_cookie_opts_for_url(url),
                }
                last_format_error: Optional[Exception] = None
                manual_downloaded_path: Optional[str] = None
                format_attempts = _video_download_format_fallbacks(format_selector, quality_key)
                for attempt_index, candidate_format in enumerate(format_attempts):
                    ydl_opts = dict(base_ydl_opts)
                    if candidate_format:
                        ydl_opts["format"] = candidate_format
                    if attempt_index > 0:
                        emit(
                            "status",
                            message=(
                                "Requested format was unavailable; retrying with "
                                f"{candidate_format or 'yt-dlp default'}..."
                            ),
                            quality=quality_key,
                        )
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info_result = ydl.extract_info(url, download=True)
                        break
                    except Exception as exc:
                        if not _is_ytdl_format_unavailable_error(exc):
                            raise
                        last_format_error = exc
                else:
                    emit(
                        "status",
                        message="yt-dlp format selection failed; trying direct stream fallback...",
                    )
                    try:
                        manual_downloaded_path = _manual_video_download_from_raw_info(
                            url,
                            quality_key,
                            format_selector,
                            emit,
                        )
                    except Exception as manual_exc:
                        if last_format_error:
                            raise RuntimeError(
                                f"{manual_exc}; original yt-dlp format error: {_clean_ytdl_error(last_format_error)}"
                            ) from manual_exc
                        raise

                downloaded_path = manual_downloaded_path or _find_recent_downloaded_video(start_time, info_result)
                if not downloaded_path:
                    raise RuntimeError("Download completed, but the output video could not be located.")
                entry = _downloaded_video_entry(downloaded_path)
                emit("complete", message="Video downloaded.", video=entry)
            except Exception as exc:
                traceback.print_exc()
                error_msg = _clean_ytdl_error(exc)
                lower_error = error_msg.lower()
                if "confirm you are not a bot" in lower_error or "sign in" in lower_error:
                    error_msg += f" ({_ytdlp_auth_hint()})"
                elif "format" in lower_error:
                    error_msg += f" ({_ytdlp_setup_hint()})"
                emit("error", message=f"Video download failed: {error_msg}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(download_sync))
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _json_event_bytes(item)

    return _json_stream_response(download_stream())


@app.post("/api/video_replace_audio")
async def api_video_replace_audio(payload: VideoReplaceAudioRequest):
    """API: Replace a downloaded video's audio with a translated audio output."""
    session: Optional[TranslateSessionData] = None
    if payload.session_id:
        session = await _get_translate_session(payload.session_id)

    video_id = (payload.downloaded_video_id or "").strip()
    if not video_id and session and session.source_video_filename:
        video_id = session.source_video_filename
    if not video_id:
        return _status_error("Downloaded video id is required.")
    try:
        video_path = _downloaded_video_path(video_id)
    except ValueError as exc:
        return _status_error(str(exc), status_code=400)
    if not os.path.exists(video_path):
        return _status_error("Downloaded video not found.", status_code=404)

    audio_mode = (payload.audio_mode or "translated").strip().lower()
    if audio_mode not in {"translated", "original", "both"}:
        return _status_error("Audio mode must be translated, original, or both.", status_code=400)
    needs_translated_audio = audio_mode in {"translated", "both"}
    audio_name = os.path.basename((payload.audio_file_name or "").strip())
    audio_path: Optional[str] = None
    if needs_translated_audio:
        if not audio_name:
            return _status_error("Translated audio filename is required.")
        if audio_name != (payload.audio_file_name or "").strip():
            return _status_error("Invalid translated audio filename.", status_code=400)
        audio_path = os.path.join(TRANSLATE_OUTPUT_DIR, audio_name)
        if not os.path.exists(audio_path):
            return _status_error("Translated audio output not found.", status_code=404)

    subtitle_path: Optional[str] = None
    subtitle_name = os.path.basename((payload.subtitle_file_name or "").strip())
    if subtitle_name:
        if subtitle_name != (payload.subtitle_file_name or "").strip():
            return _status_error("Invalid subtitle filename.", status_code=400)
        subtitle_ext = os.path.splitext(subtitle_name)[1].lstrip(".").lower()
        if subtitle_ext not in {"srt", "vtt", "ass", "ssa"}:
            return _status_error("Subtitle file must be SRT, VTT, ASS, or SSA.", status_code=400)
        subtitle_path = os.path.join(TRANSLATE_OUTPUT_DIR, subtitle_name)
        if not os.path.exists(subtitle_path):
            return _status_error("Selected subtitle output not found.", status_code=404)

    embedded_subtitles: List[Tuple[str, str]] = []
    seen_embedded_subtitle_names: Set[str] = set()
    for raw_embedded_name in payload.embedded_subtitle_file_names or []:
        embedded_name_value = (raw_embedded_name or "").strip()
        if not embedded_name_value:
            continue
        embedded_name = os.path.basename(embedded_name_value)
        if embedded_name != embedded_name_value:
            return _status_error("Invalid embedded subtitle filename.", status_code=400)
        if embedded_name in seen_embedded_subtitle_names:
            continue
        embedded_ext = os.path.splitext(embedded_name)[1].lstrip(".").lower()
        if embedded_ext not in {"srt", "vtt"}:
            return _status_error("Embedded subtitle tracks must be SRT or VTT for MP4 output.", status_code=400)
        embedded_path = os.path.join(TRANSLATE_OUTPUT_DIR, embedded_name)
        if not os.path.exists(embedded_path):
            return _status_error("Selected embedded subtitle output not found.", status_code=404)
        seen_embedded_subtitle_names.add(embedded_name)
        embedded_subtitles.append((embedded_path, _embedded_subtitle_label(embedded_name)))

    requested_base = _sanitize_base_filename(payload.output_filename)
    if not requested_base:
        video_base = _sanitize_base_filename(os.path.splitext(os.path.basename(video_path))[0]) or "video"
        audio_base = _sanitize_base_filename(os.path.splitext(audio_name)[0]) or "translated"
        requested_base = f"{video_base}_{audio_base}_video"
    output_filename = f"{requested_base}.mp4"
    output_path = os.path.join(TRANSLATE_OUTPUT_DIR, output_filename)
    if os.path.exists(output_path):
        requested_base = f"{requested_base}_{uuid.uuid4().hex[:6]}"
        output_filename = f"{requested_base}.mp4"
        output_path = os.path.join(TRANSLATE_OUTPUT_DIR, output_filename)

    try:
        await _run_blocking(
            _replace_video_audio_sync,
            video_path,
            audio_path,
            output_path,
            subtitle_path,
            audio_mode,
            embedded_subtitles,
        )
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to create translated video: {str(exc)}", status_code=500)

    return JSONResponse(
        content={
            "status": "ok",
            "message": "Video created.",
            "video_url": f"/api/translate_outputs/{output_filename}",
            "poster_url": f"/api/translate_outputs/{output_filename}/snapshot",
            "file_name": output_filename,
            "source_video": _downloaded_video_entry(video_path),
            "subtitle_file_name": subtitle_name or None,
            "embedded_subtitle_file_names": [os.path.basename(path) for path, _label in embedded_subtitles],
            "audio_mode": audio_mode,
        }
    )


@app.post("/api/translate_split_audio")
async def api_translate_split_audio(
    request: Request,
    dest_language: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    downloaded_video_id: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    base_filename: Optional[str] = Form(None),
    chunk_min_minutes: Optional[float] = Form(None),
    chunk_max_minutes: Optional[float] = Form(None),
    min_silence_ms: Optional[int] = Form(None),
    silence_threshold_db: Optional[float] = Form(None),
    super_resolution_voice: Optional[bool] = Form(False),
    enhance_voice: Optional[bool] = Form(False, description="Apply ClearVoice speech enhancement (optional, recommended for mixed audio)"),
    audio_separator_enabled: Optional[bool] = Form(False, description="Enable audio-separator for vocal/instrumental separation"),
    audio_separator_model: Optional[str] = Form(None, description="Audio-separator model: 'fast', 'balance' (default), or 'quality'"),
    audio_separator_use_soundfile: Optional[bool] = Form(None, description="Use audio-separator soundfile writer for stems (slower fallback)."),
    clearvoice_parallel_enabled: Optional[bool] = Form(False),
    clearvoice_parallel_chunk_seconds: Optional[int] = Form(None),
    clearvoice_parallel_max_workers: Optional[int] = Form(None),
    enhancement_model: Optional[str] = Form(None, description="ClearVoice enhancement model: 'MossFormerGAN_SE_16K' (default), 'FRCRN_SE_16K', or 'MossFormer2_SE_48K'"),
    audio_file: Optional[UploadFile] = File(None),
):
    """API: Split long audio into chunks for reuse. ClearVoice/Audio-Separator optional but recommended for mixed audio."""
    # Note: ClearVoice/Audio-Separator recommended for better silence detection on mixed audio,
    # but not required for vocal-only audio uploads

    try:
        payload: Optional[Dict[str, Any]] = None
        enhance_voice_value = enhance_voice
        audio_separator_enabled_value = audio_separator_enabled
        audio_separator_model_value = audio_separator_model
        audio_separator_use_soundfile_value = audio_separator_use_soundfile
        clearvoice_parallel_enabled_value = clearvoice_parallel_enabled
        clearvoice_parallel_chunk_seconds_value = clearvoice_parallel_chunk_seconds
        clearvoice_parallel_max_workers_value = clearvoice_parallel_max_workers
        payload, payload_error = await _read_optional_json_payload(
            request,
            audio_file is None
            and (audio is None or not audio.strip())
            and not (downloaded_video_id or "").strip()
            and _request_has_json_body(request),
        )
        if payload_error is not None:
            return payload_error

        if payload is not None:
            dest_language = payload.get("dest_language", dest_language)
            audio = payload.get("audio", audio)
            downloaded_video_id = payload.get("downloaded_video_id", downloaded_video_id)
            audio_mime_type = payload.get("audio_mime_type", audio_mime_type)
            base_filename = payload.get("base_filename", base_filename)
            chunk_min_minutes = payload.get("chunk_min_minutes", chunk_min_minutes)
            chunk_max_minutes = payload.get("chunk_max_minutes", chunk_max_minutes)
            min_silence_ms = payload.get("min_silence_ms", min_silence_ms)
            silence_threshold_db = payload.get("silence_threshold_db", silence_threshold_db)
            super_resolution_voice = payload.get("super_resolution_voice", super_resolution_voice)
            enhance_voice_value = payload.get("enhance_voice", enhance_voice_value)
            audio_separator_enabled_value = payload.get("audio_separator_enabled", audio_separator_enabled_value)
            audio_separator_model_value = payload.get("audio_separator_model", audio_separator_model_value)
            audio_separator_use_soundfile_value = payload.get(
                "audio_separator_use_soundfile", audio_separator_use_soundfile_value
            )
            enhancement_model = payload.get("enhancement_model", enhancement_model)
            clearvoice_parallel_enabled_value = payload.get(
                "clearvoice_parallel_enabled", clearvoice_parallel_enabled_value
            )
            clearvoice_parallel_chunk_seconds_value = payload.get(
                "clearvoice_parallel_chunk_seconds", clearvoice_parallel_chunk_seconds_value
            )
            clearvoice_parallel_max_workers_value = payload.get(
                "clearvoice_parallel_max_workers", clearvoice_parallel_max_workers_value
            )

        preprocess_options = _resolve_audio_preprocess_options(
            enhance_voice=enhance_voice_value,
            super_resolution_voice=super_resolution_voice,
            enhancement_model=enhancement_model,
            audio_separator_enabled=audio_separator_enabled_value,
            audio_separator_model=audio_separator_model_value,
            audio_separator_use_soundfile=audio_separator_use_soundfile_value,
            clearvoice_parallel_enabled=clearvoice_parallel_enabled_value,
            clearvoice_parallel_chunk_seconds=clearvoice_parallel_chunk_seconds_value,
            clearvoice_parallel_max_workers=clearvoice_parallel_max_workers_value,
        )
        enhancement_model_name_value = preprocess_options.enhancement_model_name
        audio_separator_enabled_flag = preprocess_options.audio_separator_enabled
        audio_separator_model_key = preprocess_options.audio_separator_model
        audio_reference_value = (audio or "").strip()
        downloaded_video_id_value = (downloaded_video_id or "").strip()
        if audio_file is None and not audio_reference_value and not downloaded_video_id_value:
            return _status_error("Source audio is required for chunk splitting.")

        dest_language_value = (dest_language or "").strip() or "unspecified"
        apply_super_resolution = preprocess_options.apply_super_resolution
        apply_enhancement = preprocess_options.apply_enhancement
        parallel_config = preprocess_options.parallel_config
        min_minutes = _coerce_positive_float(
            chunk_min_minutes,
            CHUNK_SPLIT_DEFAULT_MIN_MINUTES,
            min_value=CHUNK_SPLIT_MIN_MINUTES,
            max_value=CHUNK_SPLIT_MAX_MINUTES,
        )
        max_minutes = _coerce_positive_float(
            chunk_max_minutes,
            CHUNK_SPLIT_DEFAULT_MAX_MINUTES,
            min_value=min_minutes,
            max_value=CHUNK_SPLIT_MAX_MINUTES,
        )
        if max_minutes < min_minutes:
            max_minutes = min_minutes
        min_chunk_ms = int(min_minutes * 60_000)
        max_chunk_ms = int(max_minutes * 60_000)
        min_chunk_ms = max(60_000, min_chunk_ms)
        max_chunk_ms = max(min_chunk_ms, max_chunk_ms)

        min_silence_ms_value = _coerce_positive_int(
            min_silence_ms,
            CHUNK_SPLIT_MIN_SILENCE_MS,
            min_value=500,
            max_value=120_000,
        )
        silence_threshold_value = _coerce_float_range(
            silence_threshold_db,
            CHUNK_SPLIT_SILENCE_THRESHOLD_DB,
            min_value=-80.0,
            max_value=-10.0,
        )

        preloaded_audio_bytes: Optional[bytes] = None
        uploaded_filename: Optional[str] = None
        audio_mime_type_value = audio_mime_type
        audio_file_for_pipeline = audio_file
        base_filename_value = base_filename
        downloaded_video_source: Optional[Dict[str, Any]] = None

        if downloaded_video_id_value:
            (
                downloaded_video_source,
                preloaded_audio_bytes,
                downloaded_audio_filename,
                downloaded_error,
            ) = await _prepare_downloaded_video_audio_request(downloaded_video_id_value)
            if downloaded_error is not None:
                return downloaded_error
            uploaded_filename = downloaded_audio_filename
            audio_mime_type_value = "audio/mpeg"
            audio_reference_value = ""
            audio_file_for_pipeline = None
            if not base_filename_value and downloaded_video_source:
                base_filename_value = os.path.splitext(str(downloaded_video_source.get("filename") or ""))[0]
        elif audio_file is not None:
            preloaded_audio_bytes, audio_error = await _read_audio_request_bytes(
                audio_file,
                None,
                missing_message="Source audio is required for chunk splitting.",
                empty_message="Uploaded audio file is empty.",
            )
            if audio_error is not None:
                return audio_error
            uploaded_filename = audio_file.filename
            audio_mime_type_value = audio_file.content_type or audio_mime_type_value
            audio_file_for_pipeline = None

        resolved_base_name = _determine_output_base_name(
            user_base=base_filename_value,
            upload_filename=uploaded_filename,
            reuse_session=None,
        )
        input_mime_type_value = audio_mime_type_value or "audio/wav"

        split_request_summary = {
            "dest_language": dest_language_value,
            "chunk_min_minutes": min_minutes,
            "chunk_max_minutes": max_minutes,
            "min_silence_ms": min_silence_ms_value,
            "silence_threshold_db": silence_threshold_value,
            "audio_separator_enabled": audio_separator_enabled_flag,
            "audio_separator_model": audio_separator_model_key if audio_separator_enabled_flag else None,
            "audio_separator_use_soundfile": (
                preprocess_options.audio_separator_use_soundfile if audio_separator_enabled_flag else None
            ),
            "enhancement": apply_enhancement,
            "enhancement_model": enhancement_model_name_value if apply_enhancement else None,
            "super_resolution": apply_super_resolution,
            "clearvoice_parallel": parallel_config.to_metadata(),
            "base_output_name": resolved_base_name,
            "downloaded_video": _public_downloaded_video_source(downloaded_video_source),
        }

        split_profiler = PerfLogger("split:pending")

        async def split_stream():
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

            async def emit(event_type: str, **payload: Any) -> None:
                event = {
                    "event": event_type,
                    "timestamp": time.time(),
                    **payload,
                }
                await queue.put(event)

            async def emit_status(**payload: Any) -> None:
                await emit("status", **payload)

            async def heartbeat_task():
                try:
                    while True:
                        await asyncio.sleep(10)
                        await emit_status(stage="heartbeat", message="Still splitting audio...")
                except asyncio.CancelledError:
                    pass

            async def run_pipeline():
                nonlocal resolved_base_name  # Allow reassignment from cached data
                heartbeat = asyncio.create_task(heartbeat_task())
                try:
                    await emit_status(stage="start", message="Split request accepted.", summary=split_request_summary)

                    audio_prepare_start = time.perf_counter()
                    clearvoice_assets = await _prepare_clearvoice_for_split(
                        audio_file=audio_file_for_pipeline,
                        audio_reference=audio_reference_value if audio_reference_value else None,
                        preloaded_audio_bytes=preloaded_audio_bytes,
                        source_audio_filename=uploaded_filename,
                        apply_enhancement=apply_enhancement,
                        apply_super_resolution=apply_super_resolution,
                        emit_status=emit_status,
                        clearvoice_parallel_config=parallel_config if parallel_config.enabled else None,
                        enhancement_model_name=enhancement_model_name_value,
                        audio_separator_enabled=audio_separator_enabled_flag,
                        audio_separator_model=audio_separator_model_key,
                        audio_separator_use_soundfile=preprocess_options.audio_separator_use_soundfile,
                    )
                    split_profiler.mark(
                        "prepare_clearvoice_assets",
                        audio_prepare_start,
                        extra=(
                            f"clearvoice_duration={clearvoice_assets.processed_audio_info.duration_ms/1000:.1f}s, "
                            f"super_resolution={apply_super_resolution}"
                        ),
                    )

                    # Compute cache key based on processed audio + split settings
                    clearvoice_hash = clearvoice_assets.clearvoice_hash
                    split_cache_key = _split_audio_cache_key(
                        clearvoice_hash,
                        min_chunk_minutes=min_minutes,
                        max_chunk_minutes=max_minutes,
                        min_silence_ms=min_silence_ms_value,
                        silence_threshold_db=silence_threshold_value,
                    )
                    
                    print(f"🔍 Checking split audio cache: {split_cache_key}")
                    cache_lookup_start = time.perf_counter()
                    cached_split_data = _load_split_audio_cache(split_cache_key)
                    split_profiler.mark(
                        "split_cache_lookup",
                        cache_lookup_start,
                        extra="hit" if cached_split_data else "miss",
                    )
                    total_duration_ms = max(0, clearvoice_assets.processed_audio_info.duration_ms)
                    chunk_batch_id = uuid.uuid4().hex
                    split_profiler.rename(f"split:{chunk_batch_id}")
                    vocals_full_path: Optional[str] = None
                    processed_audio_buffer: Optional[AudioBuffer] = None

                    async def _ensure_vocals_full_path() -> str:
                        nonlocal vocals_full_path
                        if vocals_full_path is None:
                            vocals_full_path = await _run_blocking(
                                _persist_chunk_batch_media,
                                chunk_batch_id,
                                clearvoice_assets.processed_audio_path,
                                "vocals_full",
                                fmt="wav",
                            )
                        return vocals_full_path

                    async def _ensure_processed_audio_buffer() -> AudioBuffer:
                        nonlocal processed_audio_buffer, total_duration_ms
                        if processed_audio_buffer is None:
                            path = await _ensure_vocals_full_path()
                            processed_audio_buffer = _load_processed_audio(path)
                            if processed_audio_buffer.duration_ms:
                                total_duration_ms = processed_audio_buffer.duration_ms
                        return processed_audio_buffer
                    
                    chunk_entries: List[Dict[str, Any]] = []
                    chunk_specs: List[Dict[str, Any]] = []
                    chunk_session_manifests: List[Dict[str, Any]] = []
                    rehydrated_from_cache = False
                    processed_audio_segment: Optional[AudioSegment] = None

                    async def _ensure_processed_audio_segment() -> AudioSegment:
                        nonlocal processed_audio_segment
                        if processed_audio_segment is None:
                            processed_audio_segment = await _load_audio_segment_from_path(
                                clearvoice_assets.processed_audio_path
                            )
                        return processed_audio_segment

                    if cached_split_data:
                        # Reuse cached split MP3 files and metadata!
                        print(f"♻️ Split audio cache hit! Reusing {cached_split_data['chunk_count']} cached MP3 chunks.")
                        await emit_status(
                            stage="chunking",
                            message=f"Reusing cached split audio ({cached_split_data['chunk_count']} chunks)...",
                        )
                        chunk_ranges = cached_split_data["chunk_ranges"]
                        silence_stats = cached_split_data.get("silence_stats", {"cached": True})
                        # Don't preload chunks - will load on-demand or reference cached files
                        use_cached_chunks = True
                        cached_manifests = cached_split_data.get("chunk_sessions") or []
                        # Use cached base name for artifact discovery if available
                        cached_base_name = _sanitize_base_filename(cached_split_data.get("base_output_name"))
                        if cached_base_name:
                            resolved_base_name = cached_base_name
                            print(f"♻️ Using cached base name for artifact discovery: {resolved_base_name}")
                        elif cached_manifests and cached_manifests[0].get("source_base_name"):
                            manifest_base_name = _sanitize_base_filename(cached_manifests[0].get("source_base_name"))
                            if manifest_base_name:
                                resolved_base_name = manifest_base_name
                            print(f"♻️ Using manifest base name for artifact discovery: {resolved_base_name}")
                        if cached_manifests:
                            manifest_lang = (cached_manifests[0].get("dest_language") or "").strip().lower()
                            requested_lang = dest_language_value.strip().lower()
                            if not manifest_lang or manifest_lang == requested_lang:
                                try:
                                    rehydrate_start = time.perf_counter()
                                    restored_sessions = await asyncio.gather(
                                        *[_rehydrate_session_from_manifest(manifest) for manifest in cached_manifests]
                                    )
                                    # Update chunk_parent_id to match new batch ID so upload/lookup works
                                    for session in restored_sessions:
                                        session.chunk_parent_id = chunk_batch_id
                                    chunk_entries = [_serialize_chunk_session(session) for session in restored_sessions]
                                    rehydrated_from_cache = True
                                    chunk_session_manifests = cached_manifests
                                    split_profiler.mark(
                                        "rehydrate_chunk_sessions",
                                        rehydrate_start,
                                        extra=f"sessions={len(restored_sessions)}",
                                    )
                                except Exception as manifest_exc:
                                    print(f"⚠️ Failed to rehydrate chunk sessions from cache: {manifest_exc}")
                    else:
                        # No cache, analyze and split
                        await emit_status(stage="chunking", message="Analyzing silence to plan chunk ranges...")
                        plan_ranges_start = time.perf_counter()
                        processed_buffer = await _ensure_processed_audio_buffer()
                        chunk_ranges, silence_stats = _plan_chunk_ranges(
                            processed_buffer,
                            min_chunk_ms=min_chunk_ms,
                            max_chunk_ms=max_chunk_ms,
                            min_silence_ms=min_silence_ms_value,
                            silence_threshold_db=silence_threshold_value,
                        )
                        split_profiler.mark(
                            "plan_chunk_ranges",
                            plan_ranges_start,
                            extra=f"ranges={len(chunk_ranges)}, silence_points={silence_stats.get('silence_count', 0)}",
                        )
                        use_cached_chunks = False

                    if not chunk_ranges:
                        chunk_ranges = [
                            {
                                "start_ms": 0,
                                "end_ms": total_duration_ms,
                                "cut_reason": "full_audio",
                                "silence_midpoint_ms": None,
                            }
                        ]

                    gemini_model_name = _get_gemini_model_name()
                    use_ffmpeg_split = TRANSLATE_USE_FFMPEG_SPLIT_MERGE and _ffmpeg_available()
                    backing_track_source = clearvoice_assets.backing_track_source or "none"
                    backing_available = bool(clearvoice_assets.backing_track_path)
                    persist_tasks: List[asyncio.Task] = []
                    if backing_available:
                        persist_tasks.append(
                            asyncio.create_task(
                                _run_blocking(
                                    _persist_chunk_batch_media,
                                    chunk_batch_id,
                                    clearvoice_assets.backing_track_path,
                                    "backing_full",
                                    fmt="mp3",
                                )
                            )
                        )

                    await emit_status(
                        stage="chunking",
                        message=f"Creating up to {len(chunk_ranges)} chunk session(s)...",
                    )

                    cached_chunk_files = []
                    if use_cached_chunks:
                        cached_chunk_files = cached_split_data.get("chunk_files", [])

                    if not rehydrated_from_cache:
                        for chunk_idx, entry in enumerate(chunk_ranges, start=1):
                            start_ms = int(entry["start_ms"])
                            end_ms = int(entry["end_ms"])
                            duration_ms = max(0, end_ms - start_ms)
                            if duration_ms < CHUNK_SPLIT_MIN_CHUNK_MS:
                                continue

                            if use_cached_chunks:
                                cached_path = None
                                if cached_chunk_files and chunk_idx - 1 < len(cached_chunk_files):
                                    cached_path = cached_chunk_files[chunk_idx - 1].get("file_path")
                                chunk_mp3_path = cached_path or _split_audio_cache_path(split_cache_key, chunk_idx)
                            else:
                                chunk_mp3_path = _split_audio_cache_path(split_cache_key, chunk_idx)

                            chunk_specs.append(
                                {
                                    "chunk_idx": chunk_idx,
                                    "chunk_mp3_path": chunk_mp3_path,
                                    "start_ms": start_ms,
                                    "end_ms": end_ms,
                                    "cut_reason": entry.get("cut_reason"),
                                    "silence_midpoint_ms": entry.get("silence_midpoint_ms"),
                                    "silence_duration_ms": entry.get("silence_duration_ms"),
                                }
                            )

                    if chunk_specs and not use_cached_chunks:
                        try:
                            if use_ffmpeg_split:
                                await emit_status(
                                    stage="chunking",
                                    message="Extracting chunk audio files with ffmpeg...",
                                )
                                materialize_start = time.perf_counter()
                                await _materialize_split_chunks_ffmpeg(
                                    await _ensure_vocals_full_path(),
                                    chunk_specs,
                                    emit_status=emit_status,
                                )
                                split_profiler.mark(
                                    "materialize_chunks_ffmpeg",
                                    materialize_start,
                                    extra=f"chunks={len(chunk_specs)}",
                                )
                            else:
                                await emit_status(
                                    stage="chunking",
                                    message="Exporting chunk audio segments...",
                                )
                                materialize_start = time.perf_counter()
                                await _materialize_split_chunks_pydub(
                                await _ensure_processed_audio_segment(),
                                    chunk_specs,
                                    emit_status=emit_status,
                                )
                                split_profiler.mark(
                                    "materialize_chunks_pydub",
                                    materialize_start,
                                    extra=f"chunks={len(chunk_specs)}",
                                )
                        except Exception as exc:
                            if use_ffmpeg_split:
                                print(f"⚠️ FFmpeg chunk extraction failed: {exc}")
                                await emit_status(
                                    stage="chunking",
                                    message="FFmpeg extraction failed, falling back to in-memory chunk export...",
                                )
                                materialize_start = time.perf_counter()
                                await _materialize_split_chunks_pydub(
                                await _ensure_processed_audio_segment(),
                                    chunk_specs,
                                    emit_status=emit_status,
                                )
                                split_profiler.mark(
                                    "materialize_chunks_pydub_fallback",
                                    materialize_start,
                                    extra=f"chunks={len(chunk_specs)}",
                                )
                            else:
                                raise

                    if chunk_specs:
                        chunk_counter_lock = asyncio.Lock()
                        created_chunk_counter = 0
                        semaphore = asyncio.Semaphore(CHUNK_SESSION_CREATION_CONCURRENCY)

                        async def _build_chunk_session(spec: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                            nonlocal created_chunk_counter
                            async with semaphore:
                                chunk_audio_path = spec["chunk_mp3_path"]
                                chunk_audio_segment: Optional[AudioSegment] = None
                                chunk_audio_info: Optional[AudioAssetInfo] = None
                                path_exists = chunk_audio_path and os.path.exists(chunk_audio_path)
                                if path_exists:
                                    try:
                                        chunk_audio_info = _probe_audio_metadata_from_path(chunk_audio_path, compute_hash=False)
                                    except Exception as meta_exc:
                                        print(f"⚠️ Failed to read metadata for cached chunk {chunk_audio_path}: {meta_exc}")
                                else:
                                    base_audio = await _ensure_processed_audio_segment()
                                    chunk_audio_segment = base_audio[spec["start_ms"]:spec["end_ms"]]
                                    if chunk_audio_path:
                                        try:
                                            await _export_audio_segment_to_path(
                                                chunk_audio_segment,
                                                chunk_audio_path,
                                                fmt="mp3",
                                                bitrate=TRANSLATE_DEFAULT_BITRATE,
                                            )
                                            path_exists = True
                                        except Exception as export_exc:
                                            print(
                                                f"⚠️ Failed to persist chunk {spec['chunk_idx']} audio to {chunk_audio_path}: {export_exc}"
                                            )
                                            path_exists = False
                                    chunk_audio_info = _audio_segment_metadata(chunk_audio_segment)
                                if chunk_audio_info is None:
                                    duration_ms = max(0, int(spec["end_ms"]) - int(spec["start_ms"]))
                                    base_audio = await _ensure_processed_audio_segment()
                                    frame_rate = int(getattr(base_audio, "frame_rate", 44100) or 44100)
                                    channels = int(getattr(base_audio, "channels", 1) or 1)
                                    sample_width = int(getattr(base_audio, "sample_width", 2) or 2)
                                    frame_count = int(frame_rate * (duration_ms / 1000.0))
                                    chunk_audio_info = AudioAssetInfo(
                                        duration_ms=duration_ms,
                                        sample_rate=frame_rate,
                                        channels=channels,
                                        sample_width=sample_width,
                                        frame_count=frame_count,
                                        hash_md5=None,
                                    )

                                chunk_clearvoice_settings = {
                                    "enhancement": True,
                                    "super_resolution": apply_super_resolution,
                                }
                                if parallel_config.enabled:
                                    chunk_clearvoice_settings.update(parallel_config.to_metadata())

                                chunk_session = await _create_translate_session(
                                    chunk_audio_segment if not path_exists else None,
                                    dest_language_value,
                                    prompt="",
                                    translate_enabled=True,
                                    response_format=TRANSLATE_DEFAULT_OUTPUT_FORMAT,
                                    bitrate=TRANSLATE_DEFAULT_BITRATE,
                                    input_mime_type=input_mime_type_value,
                                    clearvoice_settings=chunk_clearvoice_settings,
                                    base_segments=[],
                                    gemini_chunks=[],
                                    gemini_model=gemini_model_name,
                                    gemini_api_key=None,
                                    backing_track_audio=None,
                                    backing_track_source="none",
                                    merge_with_backing=False,
                                    ignore_non_speech=False,
                                    preserve_silence_audio=False,
                                    generated_volume_percent=DEFAULT_GENERATED_VOLUME_PERCENT,
                                    backing_volume_percent=DEFAULT_GENERATED_VOLUME_PERCENT,
                                    chunk_parent_id=chunk_batch_id,
                                    chunk_index=spec["chunk_idx"],
                                    chunk_start_ms=spec["start_ms"],
                                    chunk_end_ms=spec["end_ms"],
                                    chunk_cut_reason=spec["cut_reason"],
                                    chunk_silence_midpoint_ms=spec["silence_midpoint_ms"],
                                    persist_media=not path_exists,
                                    source_audio_filename=uploaded_filename,
                                    source_base_name=resolved_base_name,
                                    source_video_path=downloaded_video_source.get("path") if downloaded_video_source else None,
                                    source_video_filename=downloaded_video_source.get("filename") if downloaded_video_source else None,
                                    original_audio_path=chunk_audio_path if path_exists else None,
                                    original_audio_info=chunk_audio_info,
                                )

                                chunk_manifest = _session_to_manifest(chunk_session)
                                chunk_entry = _serialize_chunk_session(chunk_session)

                            async with chunk_counter_lock:
                                created_chunk_counter += 1
                                progress_idx = created_chunk_counter
                                chunk_session_manifests.append(chunk_manifest)

                            await emit_status(
                                stage="chunking",
                                message=f"Chunk {progress_idx} created (target {len(chunk_specs)}).",
                            )
                            return spec["chunk_idx"], chunk_entry

                        build_chunks_start = time.perf_counter()
                        chunk_results = await asyncio.gather(
                            *[asyncio.create_task(_build_chunk_session(spec)) for spec in chunk_specs]
                        )
                        chunk_results.sort(key=lambda item: item[0])
                        chunk_entries = [entry for _, entry in chunk_results]
                        split_profiler.mark(
                            "create_chunk_sessions",
                            build_chunks_start,
                            extra=f"sessions={len(chunk_entries)}",
                        )
                    
                    artifact_discovery_start = time.perf_counter()
                    artifacts_by_chunk = _discover_existing_chunk_artifacts(resolved_base_name)
                    split_profiler.mark(
                        "discover_existing_artifacts",
                        artifact_discovery_start,
                        extra=f"chunk_files={len(artifacts_by_chunk)}",
                    )
                    reused_artifacts = 0
                    if artifacts_by_chunk and chunk_entries:
                        requested_lang_label = (dest_language_value or "").strip().lower()
                        lang_suffix = _language_code_from_label(dest_language_value).lower()
                        allow_wildcard_suffix = not requested_lang_label or requested_lang_label == "unspecified"
                        manifest_by_index = {
                            manifest.get("chunk_index"): manifest
                            for manifest in chunk_session_manifests
                            if manifest.get("chunk_index") is not None
                        }
                        for entry in chunk_entries:
                            idx = entry.get("chunk_index")
                            if idx is None:
                                continue
                            artifact_bundle = artifacts_by_chunk.get(int(idx))
                            if not artifact_bundle:
                                continue
                            audio_candidates = [
                                item for item in artifact_bundle.get("audio", [])
                                if item.get("suffix", "").lower() == lang_suffix
                            ]
                            subtitle_candidates = [
                                item for item in artifact_bundle.get("subtitles", [])
                                if item.get("suffix", "").lower() == lang_suffix
                            ]
                            if allow_wildcard_suffix and not audio_candidates:
                                audio_candidates = artifact_bundle.get("audio", [])
                            if allow_wildcard_suffix and not subtitle_candidates:
                                subtitle_candidates = artifact_bundle.get("subtitles", [])
                            audio_entry = audio_candidates[0] if audio_candidates else None
                            subtitle_entry = subtitle_candidates[0] if subtitle_candidates else None
                            if audio_entry:
                                entry["existing_audio_url"] = audio_entry["url"]
                                entry["existing_audio_filename"] = audio_entry["filename"]
                            if subtitle_entry:
                                entry["existing_subtitle_url"] = subtitle_entry["url"]
                                entry["existing_subtitle_filename"] = subtitle_entry["filename"]
                            manifest = manifest_by_index.get(idx)
                            session_id = manifest.get("session_id") if manifest else entry.get("session_id")
                            already_generated = False
                            if manifest and subtitle_entry:
                                already_generated = _srt_matches_manifest(manifest, subtitle_entry["path"])
                            if not already_generated and audio_entry and subtitle_entry:
                                already_generated = True
                            entry["already_generated"] = already_generated
                            if already_generated and audio_entry:
                                entry["generated"] = True
                                entry["generated_audio_url"] = audio_entry["url"]
                                entry["audio_url"] = audio_entry["url"]
                                entry["output_filename"] = audio_entry["filename"]
                                entry["output_format"] = audio_entry.get("format") or entry.get("output_format")
                                if subtitle_entry:
                                    entry["subtitle_url"] = subtitle_entry["url"]
                                    entry["subtitle_file_name"] = subtitle_entry["filename"]
                                # If we are reusing cached audio/SRT, hydrate the session/manifest so merge has segments.
                                try:
                                    audio_format = (
                                        audio_entry.get("format")
                                        or (os.path.splitext(audio_entry.get("filename", ""))[1].lstrip(".") or TRANSLATE_DEFAULT_OUTPUT_FORMAT)
                                    )
                                    if manifest is not None:
                                        manifest["chunk_output_path"] = audio_entry["path"]
                                        manifest["chunk_output_filename"] = audio_entry["filename"]
                                        manifest["chunk_output_format"] = audio_format
                                        manifest["chunk_generated"] = True
                                    cached_session = await _get_translate_session(session_id) if session_id else None
                                    if cached_session:
                                        cached_session.chunk_generated = True
                                        cached_session.chunk_output_path = audio_entry["path"]
                                        cached_session.chunk_output_filename = audio_entry["filename"]
                                        cached_session.chunk_output_format = audio_format
                                        cached_session.chunk_generated_at = cached_session.chunk_generated_at or time.time()
                                    have_segments = bool(
                                        (manifest and manifest.get("base_segments"))
                                        or (cached_session and getattr(cached_session, "base_segments", None))
                                    )
                                    if subtitle_entry and not have_segments:
                                        translated_srt_content: Optional[str] = None
                                        original_srt_content: Optional[str] = None
                                        try:
                                            with open(subtitle_entry["path"], "r", encoding="utf-8", errors="ignore") as srt_file:
                                                translated_srt_content = srt_file.read()
                                        except Exception as srt_exc:
                                            print(f"⚠️ Failed to read cached subtitle for chunk {idx}: {srt_exc}")
                                        original_candidate = next(
                                            (item for item in artifact_bundle.get("subtitles", []) if str(item.get("suffix", "")).lower() == "original"),
                                            None,
                                        )
                                        if original_candidate:
                                            try:
                                                with open(original_candidate["path"], "r", encoding="utf-8", errors="ignore") as orig_file:
                                                    original_srt_content = orig_file.read()
                                            except Exception as orig_exc:
                                                print(f"⚠️ Failed to read cached original subtitle for chunk {idx}: {orig_exc}")
                                        if translated_srt_content or original_srt_content:
                                            srt_result = _parse_srt_input_to_segments(original_srt_content, translated_srt_content)
                                            if srt_result:
                                                segments_data, speaker_profiles = srt_result
                                                if manifest is not None:
                                                    manifest["base_segments"] = copy.deepcopy(segments_data)
                                                if session_id:
                                                    try:
                                                        await _update_translate_session_segments(session_id, segments_data)
                                                    except Exception as seg_exc:
                                                        print(f"⚠️ Failed to update cached chunk segments for chunk {idx}: {seg_exc}")
                                                    else:
                                                        refreshed = await _get_translate_session(session_id)
                                                        if refreshed:
                                                            refreshed.chunk_generated = True
                                                            refreshed.chunk_output_path = audio_entry["path"]
                                                            refreshed.chunk_output_filename = audio_entry["filename"]
                                                            refreshed.chunk_output_format = audio_format
                                                            refreshed.chunk_generated_at = refreshed.chunk_generated_at or time.time()
                                                            if speaker_profiles:
                                                                refreshed.speaker_profiles = copy.deepcopy(speaker_profiles)
                                                            _refresh_split_cache_for_session(refreshed)
                                except Exception as hydrate_exc:
                                    print(f"⚠️ Failed to hydrate cached chunk artifacts for chunk {idx}: {hydrate_exc}")
                                reused_artifacts += 1
                    if reused_artifacts:
                        print(f"♻️ [{resolved_base_name}] Found {reused_artifacts} pre-generated chunk(s) matching existing artifacts.")

                    # Save to cache if we just split the audio (not using cache)
                    if not use_cached_chunks and chunk_specs:
                        chunk_ranges_for_cache = [
                            {
                                "start_ms": spec["start_ms"],
                                "end_ms": spec["end_ms"],
                                "cut_reason": spec.get("cut_reason"),
                                "silence_midpoint_ms": spec.get("silence_midpoint_ms"),
                                "silence_duration_ms": spec.get("silence_duration_ms"),
                            }
                            for spec in chunk_specs
                        ]
                        chunk_audio_paths = [spec["chunk_mp3_path"] for spec in chunk_specs]
                        await emit_status(stage="chunking", message="Caching split audio chunks...")
                        try:
                            # Run cache save in thread pool to avoid blocking
                            cache_save_start = time.perf_counter()
                            await asyncio.get_event_loop().run_in_executor(
                                executor,
                                functools.partial(
                                    _save_split_audio_cache,
                                    split_cache_key,
                                    chunk_ranges_for_cache,
                                    None,
                                    silence_stats,
                                    chunk_audio_paths=chunk_audio_paths,
                                    chunk_session_manifests=chunk_session_manifests,
                                    chunk_batch_id=chunk_batch_id,
                                    base_output_name=resolved_base_name,
                                ),
                            )
                            split_profiler.mark(
                                "cache_split_artifacts",
                                cache_save_start,
                                extra=f"chunks={len(chunk_specs)}",
                            )
                        except Exception as exc:
                            print(f"⚠️ Failed to cache split audio chunks: {exc}")

                    if not chunk_entries and total_duration_ms >= CHUNK_SPLIT_MIN_CHUNK_MS:
                        await emit_status(stage="chunking", message="Fallback: keeping full audio as single chunk.")
                        fallback_session = await _create_translate_session(
                            await _ensure_processed_audio_segment(),
                            dest_language_value,
                            prompt="",
                            translate_enabled=True,
                            response_format=TRANSLATE_DEFAULT_OUTPUT_FORMAT,
                            bitrate=TRANSLATE_DEFAULT_BITRATE,
                            input_mime_type=input_mime_type_value,
                            clearvoice_settings={"enhancement": True, "super_resolution": apply_super_resolution},
                            base_segments=[],
                            gemini_chunks=[],
                            gemini_model=gemini_model_name,
                            gemini_api_key=None,
                            backing_track_audio=None,
                            backing_track_source="none",
                            merge_with_backing=False,
                            ignore_non_speech=False,
                            preserve_silence_audio=False,
                            generated_volume_percent=DEFAULT_GENERATED_VOLUME_PERCENT,
                            backing_volume_percent=DEFAULT_GENERATED_VOLUME_PERCENT,
                            chunk_parent_id=chunk_batch_id,
                            chunk_index=1,
                            chunk_start_ms=0,
                            chunk_end_ms=total_duration_ms,
                            chunk_cut_reason="full_audio",
                            persist_media=True,
                            source_audio_filename=uploaded_filename,
                            source_base_name=resolved_base_name,
                            source_video_path=downloaded_video_source.get("path") if downloaded_video_source else None,
                            source_video_filename=downloaded_video_source.get("filename") if downloaded_video_source else None,
                        )
                        chunk_session_manifests.append(_session_to_manifest(fallback_session))
                        chunk_entries.append(_serialize_chunk_session(fallback_session))

                    if persist_tasks:
                        await asyncio.gather(*persist_tasks)

                    if not chunk_entries:
                        raise TranslateWorkflowHttpError(
                            400,
                            {
                                "status": "error",
                                "message": "Unable to derive valid chunks from the provided audio.",
                            },
                        )

                    response_payload = {
                        "status": "ok",
                        "message": f"Prepared {len(chunk_entries)} chunk(s) for advanced processing.",
                        "chunk_count": len(chunk_entries),
                        "duration_ms": total_duration_ms,
                        "duration_label": _format_ms_to_timestamp(total_duration_ms),
                        "chunks": chunk_entries,
                        "chunk_batch_id": chunk_batch_id,
                        "strategy": {
                            "min_chunk_ms": min_chunk_ms,
                            "max_chunk_ms": max_chunk_ms,
                            "min_minutes": min_minutes,
                            "max_minutes": max_minutes,
                            "min_silence_ms": min_silence_ms_value,
                            "silence_threshold_db": silence_threshold_value,
                            "silence_count": silence_stats.get("silence_count", 0),
                            "silence_metrics": silence_stats,
                        },
                        "clearvoice": {
                            "enhancement": True,
                            "super_resolution": apply_super_resolution,
                            "backing_available": backing_available,
                            "backing_source": backing_track_source,
                        },
                        "session_ttl_seconds": ADVANCED_TRANSLATE_SESSION_TTL_SECONDS,
                        "source_audio_filename": uploaded_filename,
                        "base_output_name": resolved_base_name,
                    }

                    await emit("complete", **response_payload)

                except TranslateWorkflowHttpError as http_error:
                    await emit(
                        "error",
                        status_code=http_error.status_code,
                        message=http_error.content.get("message") or "Chunk splitting failed.",
                        details=http_error.content,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    await emit(
                        "error",
                        status_code=500,
                        message=f"Failed to split audio: {str(exc)}",
                    )
                finally:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except asyncio.CancelledError:
                        pass
                    split_profiler.summary()
                    await queue.put(None)

            asyncio.create_task(run_pipeline())
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _json_event_bytes(event)

        return _json_stream_response(split_stream())

    except TranslateWorkflowHttpError as http_error:
        return JSONResponse(status_code=http_error.status_code, content=http_error.content)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to split audio: {str(exc)}"},
        )


@app.post("/api/translate_merge_chunks")
async def api_translate_merge_chunks(payload: MergeChunksRequest):
    """API: Merge multiple chunk sessions back into a single audio file."""
    try:
        merge_profiler = PerfLogger("merge:pending")
        session_ids = _normalize_session_ids(payload.chunk_session_ids)
        chunk_batch_id = (payload.chunk_batch_id or "").strip()

        sessions: List[TranslateSessionData] = []
        resolve_start = time.perf_counter()
        if session_ids:
            sessions, session_error = await _resolve_chunk_sessions_by_ids(session_ids)
            if session_error is not None:
                return session_error
        elif chunk_batch_id:
            sessions = await _list_chunk_sessions(chunk_batch_id)
            if not sessions:
                return _status_error("No chunk sessions found for this batch.", status_code=404)
            sessions.sort(key=lambda s: (s.chunk_index or 0, s.chunk_start_ms or 0))
        else:
            return _status_error("Provide chunk_session_ids (ordered list) or chunk_batch_id to merge.")
        merge_profiler.mark("resolve_chunk_sessions", resolve_start, extra=f"sessions={len(sessions)}")
        if not chunk_batch_id and sessions:
            chunk_batch_id = sessions[0].chunk_parent_id or ""
        merge_profiler.rename(f"merge:{chunk_batch_id or 'adhoc'}")

        merge_backing_request = _coerce_to_bool(payload.merge_backing_track)
        response_format_value = _normalize_translate_output_format(payload.response_format)
        bitrate_value = payload.bitrate or TRANSLATE_DEFAULT_BITRATE
        use_ffmpeg_merge = TRANSLATE_USE_FFMPEG_SPLIT_MERGE and _ffmpeg_available()
        merged_audio_path: Optional[str] = None
        merged_audio_segment: Optional[AudioSegment] = None
        temp_paths_to_cleanup: List[str] = []
        merged_segments: List[Dict[str, Any]] = []
        chunk_results: List[Dict[str, Any]] = []
        timeline_offset_ms = 0
        chunk_materials: List[Dict[str, Any]] = []
        materials_start = time.perf_counter()
        for session in sessions:
            chunk_audio_path = _resolve_session_chunk_audio_path(session)
            chunk_duration_ms = _resolve_chunk_duration_ms(session)
            chunk_results.append(
                {
                    "session_id": session.session_id,
                    "chunk_index": session.chunk_index,
                    "generated": bool(session.chunk_generated),
                    "duration_ms": chunk_duration_ms,
                    "source": "generated" if session.chunk_generated else "original",
                    "audio_url": _chunk_output_url(session) if session.chunk_generated else None,
                }
            )
            merged_segments.extend(
                _offset_segments_for_merge(
                    getattr(session, "base_segments", None),
                    timeline_offset_ms,
                    max_duration_ms=chunk_duration_ms,
                )
            )
            timeline_offset_ms += chunk_duration_ms
            chunk_materials.append(
                {
                    "session": session,
                    "audio_path": chunk_audio_path,
                    "duration_ms": chunk_duration_ms,
                }
            )
        merge_profiler.mark(
            "prepare_chunk_materials",
            materials_start,
            extra=f"chunks={len(chunk_materials)}, timeline={timeline_offset_ms/1000:.1f}s",
        )

        if not chunk_materials:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No audio chunks were provided to merge."},
            )

        ffmpeg_concat_paths = [material["audio_path"] for material in chunk_materials]
        if use_ffmpeg_merge and all(ffmpeg_concat_paths):
            concat_start = time.perf_counter()
            try:
                merged_audio_path = await _run_blocking(
                    _ffmpeg_concat_to_tempfile,
                    ffmpeg_concat_paths,
                    output_format=response_format_value,
                    bitrate=bitrate_value,
                )
                temp_paths_to_cleanup.append(merged_audio_path)
                merge_profiler.mark(
                    "ffmpeg_concat",
                    concat_start,
                    extra=f"segments={len(ffmpeg_concat_paths)}",
                )
            except Exception as exc:
                merge_profiler.mark(
                    "ffmpeg_concat_failed",
                    concat_start,
                    extra=str(exc),
                )
                print(f"⚠️ FFmpeg concat failed, falling back to Python merge: {exc}")
                merged_audio_path = None

        if merged_audio_path is None:
            python_concat_start = time.perf_counter()
            for material in chunk_materials:
                segment_audio = await _run_blocking(_load_chunk_audio_for_merge, material["session"])
                if segment_audio is None:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "status": "error",
                            "message": f"Chunk session '{material['session'].session_id}' has no audio available.",
                        },
                    )
                merged_audio_segment = segment_audio if merged_audio_segment is None else merged_audio_segment + segment_audio
            merge_profiler.mark(
                "python_concat",
                python_concat_start,
                extra=f"segments={len(chunk_materials)}",
            )

        if merged_audio_segment is None and merged_audio_path is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Unable to merge chunk audio."},
            )

        final_audio_path = merged_audio_path
        final_audio_segment = merged_audio_segment
        if merge_backing_request:
            batch_id = chunk_batch_id or (sessions[0].chunk_parent_id if sessions else None)
            if batch_id:
                backing_path = _chunk_batch_media_path(batch_id, "backing_full")
                if os.path.exists(backing_path):
                    ffmpeg_bitrate = (
                        bitrate_value if response_format_value in {"mp3", "ogg", "opus", "aac", "webm"} else None
                    )
                    if final_audio_path and use_ffmpeg_merge:
                        overlay_start = time.perf_counter()
                        PERF_LOGGER.info(
                            "⏳ [%s] ffmpeg overlay starting | backing=%s, format=%s",
                            merge_profiler.label,
                            os.path.basename(backing_path),
                            response_format_value,
                        )
                        overlay_task = asyncio.create_task(
                            _run_blocking(
                                _ffmpeg_overlay_to_tempfile,
                                final_audio_path,
                                backing_path,
                                output_format=response_format_value,
                                bitrate=ffmpeg_bitrate,
                                vocals_volume=1.0,
                                backing_volume=1.0,
                            )
                        )
                        progress_task = asyncio.create_task(_log_overlay_progress(merge_profiler.label, overlay_start))
                        try:
                            overlay_temp = await overlay_task
                            temp_paths_to_cleanup.append(final_audio_path)
                            final_audio_path = overlay_temp
                            temp_paths_to_cleanup.append(overlay_temp)
                            final_audio_segment = None
                            merge_profiler.mark(
                                "ffmpeg_overlay",
                                overlay_start,
                                extra=os.path.basename(backing_path),
                            )
                        except Exception as exc:
                            merge_profiler.mark(
                                "ffmpeg_overlay_failed",
                                overlay_start,
                                extra=str(exc),
                            )
                            print(f"⚠️ FFmpeg overlay failed, falling back to Python mix: {exc}")
                            # FFmpeg failed - need to fall back to Python overlay
                            if final_audio_segment is None:
                                if final_audio_path:
                                    final_audio_segment = await _load_audio_segment_from_path(final_audio_path)
                                else:
                                    final_audio_segment = merged_audio_segment
                        finally:
                            progress_task.cancel()
                            try:
                                await progress_task
                            except asyncio.CancelledError:
                                pass
                    # Only run Python overlay if FFmpeg wasn't used (final_audio_segment is still set)
                    if final_audio_segment is not None:
                        overlay_python_start = time.perf_counter()
                        vocal_len = len(final_audio_segment)
                        print(f"[merge] Running Python overlay for {vocal_len / 1000:.1f}s audio...")
                        PERF_LOGGER.info(
                            "⏳ [%s] python overlay starting | vocals=%.1fs, backing=%s",
                            merge_profiler.label,
                            vocal_len / 1000,
                            os.path.basename(backing_path),
                        )
                        
                        # Run overlay in thread pool to avoid blocking event loop
                        def _do_python_overlay():
                            backing_ext = os.path.splitext(backing_path)[1].lstrip(".") or None
                            full_backing_audio = AudioSegment.from_file(backing_path, format=backing_ext)
                            backing_len = len(full_backing_audio)
                            if backing_len >= vocal_len:
                                return full_backing_audio.overlay(final_audio_segment), backing_len
                            else:
                                print(
                                    f"⚠️ Backing track ({backing_len} ms) shorter than merged vocals ({vocal_len} ms); using vocal timeline as base."
                                )
                                return final_audio_segment.overlay(full_backing_audio), backing_len
                        
                        final_audio_segment, backing_len = await _run_blocking(_do_python_overlay)
                        final_audio_path = None
                        overlay_elapsed = (time.perf_counter() - overlay_python_start) * 1000
                        print(f"[merge] Python overlay complete in {overlay_elapsed:.1f}ms")
                        merge_profiler.mark(
                            "python_overlay",
                            overlay_python_start,
                            extra=f"backing_len={backing_len}ms,vocals_len={vocal_len}ms",
                        )
                else:
                    print(f"⚠️ Full backing track not found at {backing_path}; exporting vocals only.")
        if final_audio_segment is not None:
            export_start = time.perf_counter()
            PERF_LOGGER.info(
                "⏳ [%s] exporting merged audio segment | duration=%.1fs, format=%s",
                merge_profiler.label,
                len(final_audio_segment) / 1000,
                response_format_value,
            )
            merged_bytes = await _export_audio_segment_bytes(
                final_audio_segment,
                fmt=response_format_value,
                bitrate=bitrate_value if response_format_value in {"mp3", "ogg", "opus", "aac", "webm"} else None,
            )
            merge_profiler.mark(
                "export_audio_segment",
                export_start,
                extra=f"format={response_format_value}",
            )
        elif final_audio_path:
            export_start = time.perf_counter()
            PERF_LOGGER.info(
                "⏳ [%s] loading merged audio file | path=%s",
                merge_profiler.label,
                os.path.basename(final_audio_path),
            )
            merged_bytes = await async_read_file(final_audio_path)
            merge_profiler.mark(
                "read_final_audio_file",
                export_start,
                extra=os.path.basename(final_audio_path),
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to render merged audio."},
            )

        template_session = sessions[0]
        resolved_base_name = _normalize_base_filename(
            template_session.source_base_name,
            fallback=template_session.source_audio_filename,
        )
        language_code = _language_code_from_label(template_session.dest_language)
        base_stem = _compose_output_stem(resolved_base_name)
        audio_stem = _compose_output_stem(resolved_base_name, extra=language_code)
        output_filename = f"{audio_stem}.{response_format_value}"
        output_path = os.path.join(TRANSLATE_OUTPUT_DIR, output_filename)
        PERF_LOGGER.info(
            "⏳ [%s] writing merged audio to %s",
            merge_profiler.label,
            output_path,
        )
        with open(output_path, "wb") as outfile:
            outfile.write(merged_bytes)
        for temp_path in temp_paths_to_cleanup:
            await async_remove_file(temp_path)

        audio_url = f"/api/translate_outputs/{output_filename}"
        subtitle_export_start = time.perf_counter()
        PERF_LOGGER.info(
            "⏳ [%s] exporting subtitles | segments=%d",
            merge_profiler.label,
            len(merged_segments),
        )
        subtitle_translated = _export_srt_from_segments(
            merged_segments,
            base_name=base_stem,
            suffix=language_code,
            text_kind="translated",
            empty_note="No merged speech segments were available for subtitle export.",
        )
        subtitle_original = _export_srt_from_segments(
            merged_segments,
            base_name=base_stem,
            suffix="original",
            text_kind="source",
            empty_note="No merged speech segments were available for subtitle export.",
        )
        merge_profiler.mark(
            "export_subtitles",
            subtitle_export_start,
            extra=f"segments={len(merged_segments)}",
        )
        subtitle_url = subtitle_translated["url"] if subtitle_translated else None
        original_subtitle_url = subtitle_original["url"] if subtitle_original else None
        source_video = _source_video_metadata_from_session(template_session)

        merge_profiler.summary()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": f"Merged {len(chunk_results)} chunk(s).",
                "audio_url": audio_url,
                "file_name": output_filename,
                "audio_file_name": output_filename,
                "response_format": response_format_value,
                "chunk_results": chunk_results,
                "chunk_batch_id": chunk_batch_id or sessions[0].chunk_parent_id,
                "subtitle_url": subtitle_url,
                "subtitle_file_name": subtitle_translated["filename"] if subtitle_translated else None,
                "subtitle": subtitle_translated,
                "subtitle_translated": subtitle_translated,
                "original_subtitle_url": original_subtitle_url,
                "original_subtitle_file_name": subtitle_original["filename"] if subtitle_original else None,
                "subtitle_original": subtitle_original,
                "base_output_name": resolved_base_name,
                "language_code": language_code,
                "source_video": source_video,
                "source_video_filename": source_video.get("filename") if source_video else None,
            },
        )
    except Exception as exc:
        traceback.print_exc()
        if "merge_profiler" in locals():
            merge_profiler.summary()
        return _status_error(f"Failed to merge chunks: {str(exc)}", status_code=500)


@app.post("/api/translate_generate_chunks")
async def api_translate_generate_chunks(payload: ChunkBatchGenerateRequest):
    """API: Generate translated audio for multiple chunk sessions in parallel."""
    try:
        session_ids = _normalize_session_ids(payload.chunk_session_ids)
        if not session_ids:
            return _status_error("Select at least one chunk session to generate.")

        sessions, session_error = await _resolve_chunk_sessions_by_ids(session_ids)
        if session_error is not None:
            return session_error

        config_template = sessions[0]
        dest_language_value = (payload.dest_language or config_template.dest_language or "").strip()
        if not dest_language_value:
            return _status_error("Destination language (dest_language) is required.")

        response_format_value = _normalize_translate_output_format(
            payload.response_format or config_template.response_format
        )
        bitrate_value = payload.bitrate or config_template.bitrate or TRANSLATE_DEFAULT_BITRATE
        gemini_model_value = _normalize_gemini_model_name(payload.gemini_model or config_template.gemini_model)
        gemini_api_key_value = (payload.gemini_api_key or config_template.gemini_api_key or "").strip()
        translation_llm_model_value = _normalize_translation_llm_model(
            payload.translation_llm_model or config_template.translation_llm_model
        )
        transcription_pipeline_value = (
            payload.transcription_pipeline
            or getattr(config_template, "transcription_pipeline", None)
            or "qwen_omnivad"
        )
        transcription_pipeline_value = _normalize_transcription_pipeline(
            transcription_pipeline_value
        )
        whisperx_proxy_refiner_flag = (
            _coerce_to_bool(
                payload.whisperx_proxy_refiner
                if payload.whisperx_proxy_refiner is not None
                else getattr(config_template, "whisperx_proxy_refiner", False)
            )
            and transcription_pipeline_value == "whisperx"
        )
        (
            qwen_omnivad_enable_diarization_flag,
            qwen_omnivad_diarization_min_seconds_value,
        ) = _resolve_qwen_omnivad_diarization_options(
            transcription_pipeline_value,
            (
                payload.qwen_omnivad_enable_diarization
                if payload.qwen_omnivad_enable_diarization is not None
                else getattr(config_template, "qwen_omnivad_enable_diarization", True)
            ),
            (
                payload.qwen_omnivad_diarization_min_seconds
                if payload.qwen_omnivad_diarization_min_seconds is not None
                else getattr(config_template, "qwen_omnivad_diarization_min_seconds", 0.0)
            ),
        )
        qwen_omnivad_diarization_backend_value = _resolve_qwen_omnivad_diarization_backend_option(
            transcription_pipeline_value,
            (
                payload.qwen_omnivad_diarization_backend
                if payload.qwen_omnivad_diarization_backend is not None
                else getattr(config_template, "qwen_omnivad_diarization_backend", "auto")
            ),
        )
        qwen_omnivad_enable_forced_aligner_flag = _resolve_qwen_omnivad_forced_aligner_option(
            transcription_pipeline_value,
            (
                payload.qwen_omnivad_enable_forced_aligner
                if payload.qwen_omnivad_enable_forced_aligner is not None
                else getattr(config_template, "qwen_omnivad_enable_forced_aligner", True)
            ),
        )
        qwen_omnivad_merge_gap_seconds_value = _resolve_qwen_omnivad_merge_gap_seconds_option(
            transcription_pipeline_value,
            (
                payload.qwen_omnivad_merge_gap_seconds
                if payload.qwen_omnivad_merge_gap_seconds is not None
                else getattr(
                    config_template,
                    "qwen_omnivad_merge_gap_seconds",
                    DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS,
                )
            ),
        )
        ignore_non_speech_flag = _coerce_to_bool(
            payload.ignore_non_speech if payload.ignore_non_speech is not None else config_template.ignore_non_speech
        )
        preserve_silence_flag = _coerce_to_bool(
            payload.preserve_silence_audio
            if payload.preserve_silence_audio is not None
            else config_template.preserve_silence_audio
        )
        volume_options = _resolve_volume_options(
            payload.generated_volume_percent,
            payload.backing_volume_percent,
            payload.silence_volume_percent
            if payload.silence_volume_percent is not None
            else config_template.silence_volume_percent,
            generated_default=config_template.generated_volume_percent,
            backing_default=config_template.backing_volume_percent,
            silence_default=config_template.silence_volume_percent,
        )
        generated_volume_percent_value = volume_options.generated
        backing_volume_percent_value = volume_options.backing
        silence_volume_percent_value = volume_options.silence
        merge_backing_requested = _coerce_to_bool(
            payload.merge_backing_track
            if payload.merge_backing_track is not None
            else config_template.merge_with_backing
        )
        force_gemini_regenerate_flag = _coerce_to_bool(payload.force_gemini_regenerate or False)
        default_speaker_value = (payload.default_speaker_preset or "").strip()
        if not default_speaker_value:
            default_speaker_value = (getattr(config_template, "default_speaker_preset", None) or "").strip()
        default_emotion_weight_value = (
            payload.default_emotion_weight
            if payload.default_emotion_weight is not None
            else getattr(config_template, "default_emotion_weight", None)
        )
        default_emotion_weight_value = _coerce_emotion_weight(
            default_emotion_weight_value,
            DEFAULT_EMOTION_WEIGHT,
        )

        for session in sessions:
            if default_speaker_value:
                session.default_speaker_preset = default_speaker_value
            session.default_emotion_weight = default_emotion_weight_value

        summary_payload = {
            "chunks": len(sessions),
            "dest_language": dest_language_value,
            "response_format": response_format_value,
            "bitrate": bitrate_value,
            "gemini_model": gemini_model_value,
            "translation_llm_model": translation_llm_model_value,
            "transcription_pipeline": transcription_pipeline_value,
            "whisperx_proxy_refiner": whisperx_proxy_refiner_flag,
            "qwen_omnivad_enable_diarization": qwen_omnivad_enable_diarization_flag,
            "qwen_omnivad_diarization_backend": qwen_omnivad_diarization_backend_value,
            "qwen_omnivad_diarization_min_seconds": qwen_omnivad_diarization_min_seconds_value,
            "qwen_omnivad_enable_forced_aligner": qwen_omnivad_enable_forced_aligner_flag,
            "qwen_omnivad_merge_gap_seconds": qwen_omnivad_merge_gap_seconds_value,
            "merge_backing": merge_backing_requested,
            "force_gemini_regenerate": force_gemini_regenerate_flag,
        }

        async def chunk_generate_stream():
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

            async def emit(event_type: str, **payload_data: Any) -> None:
                event = {
                    "event": event_type,
                    "timestamp": time.time(),
                    **payload_data,
                }
                await queue.put(event)

            async def heartbeat_task():
                """Send periodic heartbeat to keep connection alive during long chunk generation."""
                try:
                    while True:
                        await asyncio.sleep(15)
                        await emit("heartbeat", message="Still generating chunks...")
                except asyncio.CancelledError:
                    pass

            async def run_batch():
                heartbeat = asyncio.create_task(heartbeat_task())
                success_count = 0
                failure_count = 0
                counter_lock = asyncio.Lock()

                async def handle_chunk(session: TranslateSessionData, order_index: int):
                    nonlocal success_count, failure_count
                    delay_seconds = order_index * CHUNK_BATCH_GENERATE_DELAY_SECONDS
                    if delay_seconds > 0:
                        await emit(
                            "chunk_waiting",
                            chunk_session_id=session.session_id,
                            chunk_index=session.chunk_index,
                            delay_seconds=delay_seconds,
                        )
                        await asyncio.sleep(delay_seconds)

                    await emit(
                        "chunk_start",
                        chunk_session_id=session.session_id,
                        chunk_index=session.chunk_index,
                    )

                    try:
                        audio_url, file_name, metadata = await _generate_chunk_audio_from_session(
                            session,
                            dest_language=dest_language_value,
                            response_format=response_format_value,
                            bitrate=bitrate_value,
                            gemini_model=gemini_model_value,
                            gemini_api_key=gemini_api_key_value,
                            translation_llm_model=translation_llm_model_value,
                            ignore_non_speech=ignore_non_speech_flag,
                            preserve_silence_audio=preserve_silence_flag,
                            generated_volume_percent=generated_volume_percent_value,
                            backing_volume_percent=backing_volume_percent_value,
                            merge_backing_track=merge_backing_requested,
                            silence_volume_percent=silence_volume_percent_value,
                            force_gemini_regenerate=force_gemini_regenerate_flag,
                            transcription_pipeline=transcription_pipeline_value,
                            whisperx_proxy_refiner=whisperx_proxy_refiner_flag,
                            default_speaker_preset=session.default_speaker_preset,
                            default_emotion_weight=session.default_emotion_weight,
                            qwen_omnivad_enable_diarization=qwen_omnivad_enable_diarization_flag,
                            qwen_omnivad_diarization_backend=qwen_omnivad_diarization_backend_value,
                            qwen_omnivad_diarization_min_seconds=qwen_omnivad_diarization_min_seconds_value,
                            qwen_omnivad_enable_forced_aligner=qwen_omnivad_enable_forced_aligner_flag,
                            qwen_omnivad_merge_gap_seconds=qwen_omnivad_merge_gap_seconds_value,
                        )
                        async with counter_lock:
                            success_count += 1
                        await emit(
                            "chunk_complete",
                            chunk_session_id=session.session_id,
                            chunk_index=session.chunk_index,
                            audio_url=audio_url,
                            file_name=file_name,
                            metadata=metadata,
                        )
                    except TranslateWorkflowHttpError as http_error:
                        async with counter_lock:
                            failure_count += 1
                        await emit(
                            "chunk_error",
                            chunk_session_id=session.session_id,
                            chunk_index=session.chunk_index,
                            status_code=http_error.status_code,
                            message=http_error.content.get("message") or "Chunk generation failed.",
                            details=http_error.content,
                        )
                    except Exception as exc:
                        async with counter_lock:
                            failure_count += 1
                        traceback.print_exc()
                        await emit(
                            "chunk_error",
                            chunk_session_id=session.session_id,
                            chunk_index=session.chunk_index,
                            status_code=500,
                            message=f"Chunk generation failed: {str(exc)}",
                        )

                try:
                    await emit(
                        "status",
                        stage="start",
                        message=f"Queued {len(sessions)} chunk(s) for generation.",
                        summary=summary_payload,
                    )

                    tasks = [
                        asyncio.create_task(handle_chunk(session, idx))
                        for idx, session in enumerate(sessions)
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

                    await emit(
                        "complete",
                        message="Chunk batch generation finished.",
                        total_chunks=len(sessions),
                        completed_chunks=success_count,
                        failed_chunks=failure_count,
                    )
                finally:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except asyncio.CancelledError:
                        pass
                await queue.put(None)

            asyncio.create_task(run_batch())

            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _json_event_bytes(event)

        return _json_stream_response(chunk_generate_stream())
    except TranslateWorkflowHttpError as http_error:
        return JSONResponse(status_code=http_error.status_code, content=http_error.content)
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to generate chunk audios: {str(exc)}", status_code=500)


@app.post("/api/translate_segments")
async def api_translate_segments(
    request: Request,
    dest_language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    bitrate: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    downloaded_video_id: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    base_filename: Optional[str] = Form(None),
    custom_backing_audio: Optional[str] = Form(None),
    custom_backing_audio_mime_type: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    translate_text: Optional[bool] = Form(True),
    gemini_model: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    translation_llm_model: Optional[str] = Form(None, description="Translation model for WhisperX/Qwen local pipelines."),
    enhance_voice: Optional[bool] = Form(False),
    enhancement_model: Optional[str] = Form(None, description="ClearVoice enhancement model: 'MossFormerGAN_SE_16K' (default), 'FRCRN_SE_16K', or 'MossFormer2_SE_48K'"),
    super_resolution_voice: Optional[bool] = Form(False),
    audio_separator_enabled: Optional[bool] = Form(False, description="Enable audio-separator for vocal/instrumental separation"),
    audio_separator_model: Optional[str] = Form(None, description="Audio-separator model: 'fast', 'balance' (default), or 'quality'"),
    audio_separator_use_soundfile: Optional[bool] = Form(None, description="Use audio-separator soundfile writer for stems (slower fallback)."),
    clearvoice_parallel_enabled: Optional[bool] = Form(False),
    clearvoice_parallel_chunk_seconds: Optional[int] = Form(None),
    clearvoice_parallel_max_workers: Optional[int] = Form(None),
    merge_backing_track: Optional[bool] = Form(False),
    min_speech_ms: Optional[int] = Form(None),
    max_merge_ms: Optional[int] = Form(None),
    segments_json: Optional[str] = Form(None),
    ignore_non_speech: Optional[bool] = Form(False),
    preserve_silence_audio: Optional[bool] = Form(False),
    generated_volume_percent: Optional[float] = Form(None),
    backing_volume_percent: Optional[float] = Form(None),
    silence_volume_percent: Optional[float] = Form(None),
    reuse_session_id: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    custom_backing_audio_file: Optional[UploadFile] = File(None),
    force_gemini_regenerate: Optional[bool] = Form(False),
    default_speaker_preset: Optional[str] = Form(None),
    default_emotion_weight: Optional[float] = Form(None),
    original_srt_file: Optional[UploadFile] = File(None, description="Original language SRT subtitle file"),
    translated_srt_file: Optional[UploadFile] = File(None, description="Translated SRT subtitle file"),
    transcription_pipeline: Optional[str] = Form("qwen_omnivad", description="Transcription pipeline: 'gemini' (default), 'whisperx' (local), 'qwen_omnivad' (Qwen3-ASR + OmniVAD), or 'parakeet' (NVIDIA Parakeet)"),
    whisperx_proxy_refiner: Optional[bool] = Form(False, description="Enable the experimental WhisperX speaker-aware proxy segment refiner."),
    qwen_omnivad_enable_diarization: Optional[bool] = Form(True, description="Enable diarization for Qwen OmniVAD pipeline."),
    qwen_omnivad_diarization_backend: Optional[str] = Form("auto", description="Qwen OmniVAD diarization backend: auto, pyannote, or sortformer."),
    qwen_omnivad_enable_forced_aligner: Optional[bool] = Form(True, description="Enable Qwen3 ForcedAligner timestamps for Qwen OmniVAD pipeline."),
    qwen_omnivad_diarization_min_seconds: Optional[float] = Form(0.0, description="Minimum span duration to split by diarization."),
    qwen_omnivad_merge_gap_seconds: Optional[float] = Form(DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS, description="Merge adjacent OmniVAD spans separated by this many seconds or less."),
):
    """API: Prepare translation segments for advanced translate/edit workflow."""
    reuse_session_id_value: Optional[str] = reuse_session_id
    try:
        payload: Optional[Dict[str, Any]] = None
        dest_language_value = dest_language
        response_format_value = response_format
        bitrate_value = bitrate
        audio_reference = audio
        downloaded_video_id_value = downloaded_video_id
        audio_mime_type_value = audio_mime_type
        base_filename_value = base_filename
        custom_backing_audio_value = custom_backing_audio
        custom_backing_audio_mime_type_value = custom_backing_audio_mime_type
        prompt_override = prompt
        translate_flag_value = translate_text
        gemini_model_value = gemini_model
        gemini_api_key_value = gemini_api_key
        translation_llm_model_value = translation_llm_model
        enhance_voice_value = enhance_voice
        enhancement_model_value = enhancement_model
        super_resolution_voice_value = super_resolution_voice
        audio_separator_enabled_value = audio_separator_enabled
        audio_separator_model_value = audio_separator_model
        audio_separator_use_soundfile_value = audio_separator_use_soundfile
        clearvoice_parallel_enabled_value = clearvoice_parallel_enabled
        clearvoice_parallel_chunk_seconds_value = clearvoice_parallel_chunk_seconds
        clearvoice_parallel_max_workers_value = clearvoice_parallel_max_workers
        merge_backing_track_value = merge_backing_track
        min_speech_duration_value = min_speech_ms
        max_merge_interval_value = max_merge_ms
        segments_override_value = segments_json
        ignore_non_speech_value = ignore_non_speech
        preserve_silence_audio_value = preserve_silence_audio
        generated_volume_percent_value = generated_volume_percent
        backing_volume_percent_value = backing_volume_percent
        silence_volume_percent_value = silence_volume_percent
        uploaded_filename = audio_file.filename if audio_file else None
        force_gemini_regen_value = force_gemini_regenerate
        default_speaker_value = default_speaker_preset
        default_emotion_weight_value = default_emotion_weight
        transcription_pipeline_value = transcription_pipeline
        whisperx_proxy_refiner_value = whisperx_proxy_refiner
        qwen_omnivad_diarization_backend_value = qwen_omnivad_diarization_backend
        qwen_omnivad_enable_forced_aligner_value = qwen_omnivad_enable_forced_aligner
        qwen_omnivad_merge_gap_seconds_value = qwen_omnivad_merge_gap_seconds

        payload, payload_error = await _read_optional_json_payload(
            request,
            dest_language is None
            and not audio_reference
            and not downloaded_video_id_value
            and audio_file is None
            and _request_has_json_body(request),
        )
        if payload_error is not None:
            return payload_error

        if payload is not None:
            dest_language_value = payload.get("dest_language", dest_language_value)
            response_format_value = payload.get("response_format", response_format_value)
            bitrate_value = payload.get("bitrate", bitrate_value)
            audio_reference = payload.get("audio", audio_reference)
            downloaded_video_id_value = payload.get("downloaded_video_id", downloaded_video_id_value)
            audio_mime_type_value = payload.get("audio_mime_type", audio_mime_type_value)
            base_filename_value = payload.get("base_filename", base_filename_value)
            custom_backing_audio_value = payload.get("custom_backing_audio", custom_backing_audio_value)
            custom_backing_audio_mime_type_value = payload.get(
                "custom_backing_audio_mime_type",
                custom_backing_audio_mime_type_value,
            )
            prompt_override = payload.get("prompt", prompt_override)
            translate_flag_value = payload.get("translate", translate_flag_value)
            gemini_model_value = payload.get("gemini_model", gemini_model_value)
            gemini_api_key_value = payload.get("gemini_api_key", gemini_api_key_value)
            translation_llm_model_value = payload.get("translation_llm_model", translation_llm_model_value)
            enhance_voice_value = payload.get("enhance_voice", enhance_voice_value)
            enhancement_model_value = payload.get("enhancement_model", enhancement_model_value)
            super_resolution_voice_value = payload.get("super_resolution_voice", super_resolution_voice_value)
            audio_separator_enabled_value = payload.get("audio_separator_enabled", audio_separator_enabled_value)
            audio_separator_model_value = payload.get("audio_separator_model", audio_separator_model_value)
            audio_separator_use_soundfile_value = payload.get(
                "audio_separator_use_soundfile", audio_separator_use_soundfile_value
            )
            clearvoice_parallel_enabled_value = payload.get(
                "clearvoice_parallel_enabled", clearvoice_parallel_enabled_value
            )
            clearvoice_parallel_chunk_seconds_value = payload.get(
                "clearvoice_parallel_chunk_seconds", clearvoice_parallel_chunk_seconds_value
            )
            clearvoice_parallel_max_workers_value = payload.get(
                "clearvoice_parallel_max_workers", clearvoice_parallel_max_workers_value
            )
            merge_backing_track_value = payload.get("merge_backing_track", merge_backing_track_value)
            min_speech_duration_value = payload.get("min_speech_ms", min_speech_duration_value)
            max_merge_interval_value = payload.get("max_merge_ms", max_merge_interval_value)
            segments_override_value = payload.get("segments_json", segments_override_value)
            ignore_non_speech_value = payload.get("ignore_non_speech", ignore_non_speech_value)
            preserve_silence_audio_value = payload.get("preserve_silence_audio", preserve_silence_audio_value)
            generated_volume_percent_value = payload.get(
                "generated_volume_percent",
                generated_volume_percent_value,
            )
            backing_volume_percent_value = payload.get(
                "backing_volume_percent",
                backing_volume_percent_value,
            )
            silence_volume_percent_value = payload.get(
                "silence_volume_percent",
                silence_volume_percent_value,
            )
            reuse_session_id_value = payload.get("reuse_session_id", reuse_session_id_value)
            force_gemini_regen_value = payload.get("force_gemini_regenerate", force_gemini_regen_value)
            if payload.get("default_speaker_preset"):
                default_speaker_value = payload.get("default_speaker_preset")
            if "default_emotion_weight" in payload:
                default_emotion_weight_value = payload.get("default_emotion_weight", default_emotion_weight_value)
            if payload.get("transcription_pipeline"):
                transcription_pipeline_value = payload.get("transcription_pipeline")
            if "whisperx_proxy_refiner" in payload:
                whisperx_proxy_refiner_value = payload.get("whisperx_proxy_refiner")
            if "qwen_omnivad_enable_diarization" in payload:
                qwen_omnivad_enable_diarization = payload.get("qwen_omnivad_enable_diarization")
            if "qwen_omnivad_diarization_backend" in payload:
                qwen_omnivad_diarization_backend_value = payload.get("qwen_omnivad_diarization_backend")
            if "qwen_omnivad_enable_forced_aligner" in payload:
                qwen_omnivad_enable_forced_aligner_value = payload.get("qwen_omnivad_enable_forced_aligner")
            if "qwen_omnivad_diarization_min_seconds" in payload:
                qwen_omnivad_diarization_min_seconds = payload.get("qwen_omnivad_diarization_min_seconds")
            if "qwen_omnivad_merge_gap_seconds" in payload:
                qwen_omnivad_merge_gap_seconds_value = payload.get("qwen_omnivad_merge_gap_seconds")

        srt_segments_from_upload, srt_upload_error = await _load_srt_segments_override(
            original_srt_file,
            translated_srt_file,
            log_prefix="translate_segments",
        )
        if srt_upload_error is not None:
            return srt_upload_error
        if srt_segments_from_upload:
            segments_override_value = srt_segments_from_upload

        dest_language_value = (dest_language_value or "").strip()
        if not dest_language_value:
            return _status_error("Destination language (dest_language) is required.")

        response_format_value = _normalize_translate_output_format(response_format_value)
        bitrate_value = bitrate_value or TRANSLATE_DEFAULT_BITRATE
        translate_enabled = _coerce_to_bool(translate_flag_value if translate_flag_value is not None else True)
        ignore_non_speech_flag = _coerce_to_bool(ignore_non_speech_value)
        preserve_silence_audio_flag = _coerce_to_bool(preserve_silence_audio_value)
        preprocess_options = _resolve_audio_preprocess_options(
            enhance_voice=enhance_voice_value,
            super_resolution_voice=super_resolution_voice_value,
            enhancement_model=enhancement_model_value,
            audio_separator_enabled=audio_separator_enabled_value,
            audio_separator_model=audio_separator_model_value,
            audio_separator_use_soundfile=audio_separator_use_soundfile_value,
            clearvoice_parallel_enabled=clearvoice_parallel_enabled_value,
            clearvoice_parallel_chunk_seconds=clearvoice_parallel_chunk_seconds_value,
            clearvoice_parallel_max_workers=clearvoice_parallel_max_workers_value,
            force_enhancement_for_super_resolution=True,
        )
        apply_enhancement = preprocess_options.apply_enhancement
        apply_super_resolution = preprocess_options.apply_super_resolution
        parallel_config = preprocess_options.parallel_config
        custom_backing_present = bool(custom_backing_audio_file) or bool((custom_backing_audio_value or "").strip())
        merge_backing_requested_raw = _coerce_to_bool(merge_backing_track_value)
        min_speech_duration = _coerce_positive_int(
            min_speech_duration_value,
            MIN_SPEECH_DURATION_MS,
            min_value=500,
        )
        max_merge_interval = _coerce_positive_int(
            max_merge_interval_value,
            MAX_MERGE_INTERVAL_MS,
            min_value=0,
        )
        volume_options = _resolve_volume_options(
            generated_volume_percent_value,
            backing_volume_percent_value,
            silence_volume_percent_value,
        )
        generated_volume_percent_value = volume_options.generated
        backing_volume_percent_value = volume_options.backing
        silence_volume_percent_value = volume_options.silence
        force_gemini_regenerate_flag = _coerce_to_bool(force_gemini_regen_value or False)
        default_speaker_value = (default_speaker_value or "").strip()
        reuse_session_id_value, reuse_source_session, reuse_session_error = await _resolve_reuse_translate_session(
            reuse_session_id_value
        )
        if reuse_session_error is not None:
            return reuse_session_error
        reuse_backing_available = bool(reuse_source_session and _session_has_backing_audio(reuse_source_session))
        audio_separator_enabled_flag = preprocess_options.audio_separator_enabled
        audio_separator_model_key = preprocess_options.audio_separator_model
        requested_merge_backing = _coerce_merge_backing_flag(
            merge_backing_requested_raw,
            apply_enhancement,
            custom_backing_present or reuse_backing_available,
            audio_separator_enabled=audio_separator_enabled_flag,
        )
        if reuse_source_session is not None and reuse_source_session.chunk_parent_id:
            requested_merge_backing = False
        enhancement_model_name_value = preprocess_options.enhancement_model_name
        clearvoice_settings = preprocess_options.to_clearvoice_settings()
        resolved_gemini_model = _normalize_gemini_model_name(gemini_model_value)
        resolved_translation_llm_model = _normalize_translation_llm_model(translation_llm_model_value)
        gemini_api_key_value = (gemini_api_key_value or "").strip()
        if not default_speaker_value and reuse_source_session:
            default_speaker_value = (reuse_source_session.default_speaker_preset or "").strip()
        if not default_speaker_value:
            default_speaker_value = None
        if default_emotion_weight_value is None and reuse_source_session is not None:
            default_emotion_weight_value = reuse_source_session.default_emotion_weight
        default_emotion_weight_value = _coerce_emotion_weight(
            default_emotion_weight_value,
            DEFAULT_EMOTION_WEIGHT,
        )
        resolved_transcription_pipeline = _normalize_transcription_pipeline(
            transcription_pipeline_value
        )
        whisperx_proxy_refiner_flag = (
            _coerce_to_bool(whisperx_proxy_refiner_value)
            and resolved_transcription_pipeline == "whisperx"
        )
        (
            qwen_omnivad_enable_diarization_flag,
            qwen_omnivad_diarization_min_seconds_value,
        ) = _resolve_qwen_omnivad_diarization_options(
            resolved_transcription_pipeline,
            qwen_omnivad_enable_diarization,
            qwen_omnivad_diarization_min_seconds,
        )
        qwen_omnivad_diarization_backend_value = _resolve_qwen_omnivad_diarization_backend_option(
            resolved_transcription_pipeline,
            qwen_omnivad_diarization_backend_value,
        )
        qwen_omnivad_enable_forced_aligner_flag = _resolve_qwen_omnivad_forced_aligner_option(
            resolved_transcription_pipeline,
            qwen_omnivad_enable_forced_aligner_value,
        )
        qwen_omnivad_merge_gap_seconds_value = _resolve_qwen_omnivad_merge_gap_seconds_option(
            resolved_transcription_pipeline,
            qwen_omnivad_merge_gap_seconds_value,
        )

        downloaded_video_source: Optional[Dict[str, Any]] = None
        downloaded_audio_bytes: Optional[bytes] = None
        downloaded_audio_filename: Optional[str] = None
        if reuse_source_session is None and (downloaded_video_id_value or "").strip():
            (
                downloaded_video_source,
                downloaded_audio_bytes,
                downloaded_audio_filename,
                downloaded_error,
            ) = await _prepare_downloaded_video_audio_request(downloaded_video_id_value)
            if downloaded_error is not None:
                return downloaded_error
            audio_reference = None
            audio_file = None
            audio_mime_type_value = "audio/mpeg"
            uploaded_filename = downloaded_audio_filename or uploaded_filename

        reuse_session_for_segments = reuse_source_session
        session_source_filename = uploaded_filename or (
            reuse_session_for_segments.source_audio_filename if reuse_session_for_segments else None
        )
        resolved_base_name = _determine_output_base_name(
            user_base=base_filename_value,
            upload_filename=session_source_filename,
            reuse_session=reuse_session_for_segments,
        )

        request_summary = {
            "dest_language": dest_language_value,
            "response_format": response_format_value,
            "bitrate": bitrate_value,
            "enhancement": apply_enhancement,
            "super_resolution": apply_super_resolution,
            "audio_separator_enabled": audio_separator_enabled_flag,
            "audio_separator_model": audio_separator_model_key if audio_separator_enabled_flag else None,
            "audio_separator_use_soundfile": (
                preprocess_options.audio_separator_use_soundfile if audio_separator_enabled_flag else None
            ),
            "merge_backing": requested_merge_backing,
            "custom_backing": custom_backing_present,
            "reuse_backing": reuse_backing_available,
            "ignore_non_speech": ignore_non_speech_flag,
            "preserve_silence_audio": preserve_silence_audio_flag,
            "generated_volume_percent": generated_volume_percent_value,
            "backing_volume_percent": backing_volume_percent_value,
            "silence_volume_percent": silence_volume_percent_value,
            "translate_enabled": translate_enabled,
            "base_output_name": resolved_base_name,
            "downloaded_video": _public_downloaded_video_source(downloaded_video_source),
            "force_gemini_regenerate": force_gemini_regenerate_flag,
            "translation_llm_model": resolved_translation_llm_model,
            "transcription_pipeline": resolved_transcription_pipeline,
            "whisperx_proxy_refiner": whisperx_proxy_refiner_flag,
            "qwen_omnivad_enable_diarization": qwen_omnivad_enable_diarization_flag,
            "qwen_omnivad_diarization_backend": qwen_omnivad_diarization_backend_value,
            "qwen_omnivad_diarization_min_seconds": qwen_omnivad_diarization_min_seconds_value,
            "qwen_omnivad_enable_forced_aligner": qwen_omnivad_enable_forced_aligner_flag,
            "qwen_omnivad_merge_gap_seconds": qwen_omnivad_merge_gap_seconds_value,
        }
        print(
            "[translate_segments] dest=%s reuse_session=%s chunk_index=%s upload=%s format=%s backing=%s merge_backing=%s"
            % (
                dest_language_value,
                bool(reuse_session_for_segments),
                getattr(reuse_session_for_segments, "chunk_index", None),
                bool(audio_file and not reuse_session_for_segments),
                response_format_value,
                custom_backing_present
                or bool(reuse_session_for_segments and _session_has_backing_audio(reuse_session_for_segments)),
                requested_merge_backing,
            )
        )

        async def translate_stream():
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

            async def emit(event_type: str, **payload: Any) -> None:
                event = {
                    "event": event_type,
                    "timestamp": time.time(),
                    **payload,
                }
                await queue.put(event)

            async def emit_status(**payload: Any) -> None:
                await emit("status", **payload)

            async def heartbeat_task():
                try:
                    while True:
                        await asyncio.sleep(10)
                        await emit_status(message="Still preparing translation segments...", stage="heartbeat")
                except asyncio.CancelledError:
                    pass

            async def run_pipeline():
                heartbeat = asyncio.create_task(heartbeat_task())
                try:
                    await emit_status(stage="start", message="Segment request accepted.", summary=request_summary)
                    print(
                        "[translate_segments] pipeline start reuse_session=%s chunk_id=%s"
                        % (
                            reuse_session_for_segments.session_id if reuse_session_for_segments else None,
                            getattr(reuse_session_for_segments, "chunk_index", None),
                        )
                    )

                    (
                        original_audio,
                        input_mime_type_value,
                        processed_audio_bytes,
                        gemini_mime_type,
                        backing_track_audio,
                        merge_with_backing,
                        backing_track_source,
                        cached_vocals_path,
                        cached_backing_path,
                    ) = await _prepare_audio_assets(
                        reuse_source_session=reuse_session_for_segments,
                        audio_file=audio_file,
                        audio_reference=audio_reference,
                        preloaded_audio_bytes=downloaded_audio_bytes,
                        source_audio_filename=session_source_filename,
                        audio_mime_type_value=audio_mime_type_value,
                        apply_enhancement=apply_enhancement,
                        apply_super_resolution=apply_super_resolution,
                        requested_merge_backing=requested_merge_backing,
                        custom_backing_audio_file=custom_backing_audio_file,
                        custom_backing_audio_reference=custom_backing_audio_value,
                        custom_backing_mime_type_value=custom_backing_audio_mime_type_value,
                        emit_status=emit_status,
                        clearvoice_parallel_config=parallel_config if parallel_config.enabled else None,
                        enhancement_model_name=enhancement_model_name_value,
                        audio_separator_enabled=audio_separator_enabled_flag,
                        audio_separator_model=audio_separator_model_key,
                        audio_separator_use_soundfile=preprocess_options.audio_separator_use_soundfile,
                    )

                    final_prompt = _resolve_final_prompt(
                        prompt_override,
                        dest_language_value,
                        translate_enabled,
                        ignore_non_speech_flag,
                    )

                    segment_result = await _build_translation_segments(
                        original_audio=original_audio,
                        processed_audio_bytes=processed_audio_bytes,
                        gemini_mime_type=gemini_mime_type,
                        dest_language=dest_language_value,
                        final_prompt=final_prompt,
                        translate_enabled=translate_enabled,
                        response_format_value=response_format_value,
                        bitrate_value=bitrate_value,
                        input_mime_type=input_mime_type_value,
                        apply_enhancement=apply_enhancement,
                        apply_super_resolution=apply_super_resolution,
                        ignore_non_speech_flag=ignore_non_speech_flag,
                        preserve_silence_audio_flag=preserve_silence_audio_flag,
                        generated_volume_percent_value=generated_volume_percent_value,
                        backing_volume_percent_value=backing_volume_percent_value,
                        silence_volume_percent_value=silence_volume_percent_value,
                        backing_track_audio=backing_track_audio,
                        backing_track_source=backing_track_source,
                        merge_with_backing=merge_with_backing,
                        segments_override_value=segments_override_value,
                        min_speech_duration=min_speech_duration,
                        max_merge_interval=max_merge_interval,
                        resolved_gemini_model=resolved_gemini_model,
                        gemini_api_key_value=gemini_api_key_value or None,
                        translation_llm_model=resolved_translation_llm_model,
                        emit_status=emit_status,
                        source_chunk_session=reuse_session_for_segments,
                        source_audio_filename=session_source_filename,
                        source_base_name=resolved_base_name,
                        source_video_path=downloaded_video_source.get("path") if downloaded_video_source else None,
                        source_video_filename=downloaded_video_source.get("filename") if downloaded_video_source else None,
                        force_gemini_regenerate=force_gemini_regenerate_flag,
                        initial_speaker_overrides=getattr(reuse_session_for_segments, "speaker_overrides", None),
                        default_speaker_preset=default_speaker_value,
                        default_emotion_weight=default_emotion_weight_value,
                        clearvoice_settings=clearvoice_settings,
                        # Pass cached paths for fast session storage
                        original_audio_source_path=cached_vocals_path,
                        backing_track_source_path=cached_backing_path,
                        transcription_pipeline=resolved_transcription_pipeline,
                        whisperx_proxy_refiner=whisperx_proxy_refiner_flag,
                        qwen_omnivad_enable_diarization=qwen_omnivad_enable_diarization_flag,
                        qwen_omnivad_diarization_backend=qwen_omnivad_diarization_backend_value,
                        qwen_omnivad_diarization_min_seconds=qwen_omnivad_diarization_min_seconds_value,
                        qwen_omnivad_enable_forced_aligner=qwen_omnivad_enable_forced_aligner_flag,
                        qwen_omnivad_merge_gap_seconds=qwen_omnivad_merge_gap_seconds_value,
                    )

                    metadata = segment_result.metadata
                    metadata["output_base_name"] = resolved_base_name
                    if session_source_filename:
                        metadata["source_audio_filename"] = session_source_filename
                    if segment_result.gemini_raw_text is not None:
                        metadata["gemini_raw_text"] = segment_result.gemini_raw_text
                    if reuse_session_for_segments is not None:
                        metadata["reuse_source_session_id"] = reuse_session_id_value

                    await emit(
                        "complete",
                        message="Segments ready.",
                        session_id=segment_result.session.session_id,
                        segments=segment_result.ui_segments,
                        metadata=metadata,
                        gemini_raw_segments=segment_result.gemini_chunks,
                        separation=metadata.get("separation"),
                    )
                    print(
                        "[translate_segments] complete session_id=%s chunk_id=%s segments=%s"
                        % (
                            segment_result.session.session_id,
                            getattr(reuse_session_for_segments, "chunk_index", None),
                            len(segment_result.ui_segments),
                        )
                    )

                except TranslateWorkflowHttpError as http_error:
                    await emit(
                        "error",
                        status_code=http_error.status_code,
                        message=http_error.content.get("message") or "Translation failed.",
                        details=http_error.content,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    await emit(
                        "error",
                        status_code=500,
                        message=f"Translation failed: {str(exc)}",
                    )
                finally:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except asyncio.CancelledError:
                        pass
                    await queue.put(None)
            asyncio.create_task(run_pipeline())
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _json_event_bytes(item)

        return _json_stream_response(translate_stream())
    except RuntimeError as runtime_error:
        return _status_error(str(runtime_error), status_code=500)
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to prepare translation segments: {str(exc)}", status_code=500)


@app.post("/api/translate_generate_segments")
async def api_translate_generate_segments(payload: TranslateGenerateRequest):
    """API: Generate translated audio from edited segments."""
    try:
        session = await _get_translate_session(payload.session_id)
        if session is None:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Translate session not found or expired."},
            )
        resolved_base_name = _normalize_base_filename(
            session.source_base_name,
            fallback=session.source_audio_filename,
        )

        merge_preference = session.merge_with_backing
        if payload.merge_backing_track is not None:
            merge_preference = _coerce_to_bool(payload.merge_backing_track)
            session.merge_with_backing = merge_preference
        merge_with_backing = merge_preference and _session_has_backing_audio(session)
        if merge_preference and not _session_has_backing_audio(session):
            print("⚠️ Merge-back preference is enabled but no backing track is stored for this session.")

        response_format_value = _normalize_translate_output_format()

        bitrate_value = payload.bitrate or session.bitrate or TRANSLATE_DEFAULT_BITRATE
        max_duration = _session_audio_duration_ms(session)
        base_segment_map = {seg.get("index"): seg for seg in session.base_segments}

        volume_percent = session.generated_volume_percent or DEFAULT_GENERATED_VOLUME_PERCENT
        backing_volume_percent = session.backing_volume_percent or DEFAULT_GENERATED_VOLUME_PERCENT
        silence_volume_percent = session.silence_volume_percent or DEFAULT_SILENCE_VOLUME_PERCENT
        if payload.generated_volume_percent is not None:
            volume_percent = _coerce_volume_percent(
                payload.generated_volume_percent,
                volume_percent,
            )
        session.generated_volume_percent = volume_percent
        if payload.backing_volume_percent is not None:
            backing_volume_percent = _coerce_volume_percent(
                payload.backing_volume_percent,
                backing_volume_percent,
            )
            session.backing_volume_percent = backing_volume_percent
        if payload.silence_volume_percent is not None:
            silence_volume_percent = _coerce_volume_percent(
                payload.silence_volume_percent,
                silence_volume_percent,
            )
            session.silence_volume_percent = silence_volume_percent

        chunk_source_session: Optional[TranslateSessionData] = None
        if session.chunk_source_session_id:
            chunk_source_session = await _get_translate_session(session.chunk_source_session_id)

        async def generate_stream():
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

            async def emit(event_type: str, **payload: Any) -> None:
                event = {
                    "event": event_type,
                    "timestamp": time.time(),
                    **payload,
                }
                await queue.put(event)

            async def emit_status(**payload: Any) -> None:
                await emit("status", **payload)

            async def heartbeat_task():
                try:
                    while True:
                        await asyncio.sleep(10)
                        await emit_status(stage="heartbeat", message="Still synthesizing segments...", session_id=session.session_id)
                except asyncio.CancelledError:
                    pass

            chunk_session_for_generation = chunk_source_session

            async def run_pipeline():
                heartbeat = asyncio.create_task(heartbeat_task())
                try:
                    await emit_status(stage="start", message="Segment synthesis request accepted.", session_id=session.session_id)

                    final_segments: List[Dict[str, Any]] = []
                    sanitized_segments: List[Dict[str, Any]] = []

                    if payload.speaker_overrides is not None:
                        raw_override_input = {
                            key: value.dict(exclude_none=True)
                            for key, value in payload.speaker_overrides.items()
                        }
                        normalized_overrides = _normalize_speaker_overrides(
                            raw_override_input,
                            session.speaker_profiles,
                        )
                        session.speaker_overrides = normalized_overrides
                        await _update_translate_session_speaker_overrides(
                            session.session_id,
                            normalized_overrides,
                        )

                    for seg_input in payload.segments:
                        start_ms = max(0, int(seg_input.start_ms))
                        end_ms = max(0, int(seg_input.end_ms))
                        if end_ms > max_duration:
                            end_ms = max_duration
                        if end_ms <= start_ms:
                            raise TranslateWorkflowHttpError(
                                400,
                                {
                                    "status": "error",
                                    "message": f"Segment index {seg_input.index} has invalid timing (end <= start).",
                                },
                            )
                        duration_ms = end_ms - start_ms
                        start_label = _format_ms_to_timestamp(start_ms)
                        end_label = _format_ms_to_timestamp(end_ms)
                        translated_text = (seg_input.translated_text or "").strip()
                        source_text = (seg_input.source_text or "").strip()

                        is_speech = seg_input.type == "speech"
                        generate_flag = bool(seg_input.generate) if is_speech else False
                        keep_original = not generate_flag if is_speech else True

                        segment_payload: Dict[str, Any] = {
                            "index": seg_input.index,
                            "type": seg_input.type,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "duration_ms": duration_ms,
                            "start": start_label,
                            "end": end_label,
                            "source_text": source_text,
                            "translated_text": translated_text,
                            "generate": generate_flag if is_speech else False,
                            "keep_original": keep_original,
                        }
                        if is_speech and seg_input.volume_percent is not None:
                            segment_payload["volume_percent"] = _coerce_volume_percent(
                                seg_input.volume_percent,
                                DEFAULT_GENERATED_VOLUME_PERCENT,
                            )
                        if is_speech and seg_input.emotion_weight is not None:
                            segment_payload["emotion_weight"] = _coerce_emotion_weight(
                                seg_input.emotion_weight,
                                DEFAULT_EMOTION_WEIGHT,
                            )

                        base_info = base_segment_map.get(seg_input.index)
                        speaker_label = None
                        if base_info and base_info.get("speaker"):
                            speaker_label = base_info["speaker"]
                        elif seg_input.speaker:
                            speaker_label = seg_input.speaker
                        if speaker_label:
                            segment_payload["speaker"] = speaker_label
                        if base_info and base_info.get("text_keys"):
                            segment_payload["text_keys"] = base_info["text_keys"]

                        final_segments.append(segment_payload)

                        sanitized_segment: Dict[str, Any] = {
                            "index": seg_input.index,
                            "type": seg_input.type,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "duration_ms": duration_ms,
                            "start": start_label,
                            "end": end_label,
                            "source_text": source_text,
                            "translated_text": translated_text,
                        }
                        if is_speech and "volume_percent" in segment_payload:
                            sanitized_segment["volume_percent"] = segment_payload["volume_percent"]
                        if is_speech and "emotion_weight" in segment_payload:
                            sanitized_segment["emotion_weight"] = segment_payload["emotion_weight"]
                        if base_info and base_info.get("text_keys"):
                            sanitized_segment["text_keys"] = base_info["text_keys"]
                        if speaker_label:
                            sanitized_segment["speaker"] = speaker_label
                        sanitized_segments.append(sanitized_segment)

                    final_segments.sort(key=lambda seg: (int(seg.get("start_ms", 0)), int(seg.get("index", 0))))
                    sanitized_segments.sort(key=lambda seg: (int(seg.get("start_ms", 0)), int(seg.get("index", 0))))

                    speech_count = sum(1 for s in final_segments if s.get("type") == "speech")
                    await emit_status(
                        stage="synthesis",
                        message=f"Synthesizing translated speech ({speech_count} speech segments)...",
                        session_id=session.session_id,
                    )

                    audio_payload, media_type, metadata = await _synthesize_translated_audio(
                        _get_session_original_audio(session),
                        final_segments,
                        session.dest_language,
                        response_format=response_format_value,
                        bitrate=bitrate_value,
                        input_mime_type=session.input_mime_type,
                        clearvoice_settings=session.clearvoice_settings,
                        backing_track_audio=_get_session_backing_audio(session),
                        backing_track_source=session.backing_track_source,
                        merge_with_backing=merge_with_backing,
                        preserve_silence_audio=session.preserve_silence_audio,
                        generated_volume_percent=volume_percent,
                        silence_volume_percent=session.silence_volume_percent,
                        speaker_overrides=session.speaker_overrides,
                        backing_volume_percent=backing_volume_percent,
                        default_speaker_preset=session.default_speaker_preset,
                        default_emotion_weight=session.default_emotion_weight,
                        emit_status=emit_status,
                        return_unmixed_audio=merge_with_backing and _session_has_backing_audio(session),
                    )

                    backing_meta = metadata.setdefault("backing_track", {})
                    backing_meta["requested"] = session.merge_with_backing
                    backing_meta["volume_percent"] = backing_volume_percent
                    backing_meta.setdefault("source", session.backing_track_source or "none")
                    metadata["ignore_non_speech"] = session.ignore_non_speech
                    metadata["preserve_silence_audio"] = session.preserve_silence_audio
                    metadata["speaker_overrides"] = copy.deepcopy(session.speaker_overrides)
                    metadata["backing_volume_percent"] = backing_volume_percent
                    metadata["default_speaker_preset"] = session.default_speaker_preset
                    metadata["default_emotion_weight"] = session.default_emotion_weight
                    if session.gemini_raw_text is not None:
                        metadata["gemini_raw_text"] = session.gemini_raw_text
                    generated_count = sum(
                        1 for seg in final_segments if seg.get("type") == "speech" and not seg.get("keep_original", False)
                    )
                    preserved_count = sum(
                        1 for seg in final_segments if seg.get("type") == "speech" and seg.get("keep_original", False)
                    )

                    metadata["selected_generated_count"] = generated_count
                    metadata["selected_preserved_count"] = preserved_count
                    metadata["session_id"] = session.session_id
                    metadata["separation"] = {
                        "vocals_available": True,
                        "vocals_url": f"/api/translate_vocals/{session.session_id}",
                        "backing_available": _session_has_backing_audio(session),
                        "backing_url": (
                            f"/api/translate_backing_track/{session.session_id}"
                            if _session_has_backing_audio(session)
                            else None
                        ),
                        "session_id": session.session_id,
                    }
                    metadata["gemini_model"] = session.gemini_model or _get_gemini_model_name()
                    metadata["output_base_name"] = resolved_base_name

                    await _update_translate_session_segments(session.session_id, sanitized_segments)
                    await _update_translate_session_metadata(
                        session.session_id,
                        response_format=response_format_value,
                        bitrate=bitrate_value,
                        gemini_model=session.gemini_model,
                        generated_volume_percent=volume_percent,
                        backing_volume_percent=backing_volume_percent,
                    )

                    headers = {
                        "Content-Disposition": f"attachment; filename=translated_speech.{response_format_value}",
                        "X-Translation-Model": session.gemini_model or _get_gemini_model_name(),
                        "X-Translation-LLM": session.translation_llm_model,
                        "X-Translation-Segments": str(len(final_segments)),
                        "X-Translation-Generated": str(generated_count),
                        "X-Translation-Preserved": str(preserved_count),
                        "X-Translation-Input-Mime": session.input_mime_type or "",
                        "X-Translate-Session": session.session_id,
                        "X-Translation-Volume-Percent": f"{volume_percent:.2f}",
                    }
                    clearvoice_settings = session.clearvoice_settings or {}
                    if clearvoice_settings:
                        headers["X-Translation-ClearVoice"] = (
                            f"enhancement={str(clearvoice_settings.get('enhancement', False)).lower()};"
                            f"super_resolution={str(clearvoice_settings.get('super_resolution', False)).lower()}"
                        )
                    headers["X-Translation-Backing"] = (
                        f"available={str(_session_has_backing_audio(session)).lower()};merged={str(merge_with_backing).lower()}"
                    )

                    chunk_index_for_name = (
                        chunk_session_for_generation.chunk_index
                        if chunk_session_for_generation and chunk_session_for_generation.chunk_parent_id
                        else None
                    )
                    language_code = _language_code_from_label(session.dest_language)
                    base_stem = _compose_output_stem(resolved_base_name, chunk_index=chunk_index_for_name)
                    audio_stem = _compose_output_stem(
                        resolved_base_name,
                        chunk_index=chunk_index_for_name,
                        extra=language_code,
                    )
                    output_filename = f"{audio_stem}.{response_format_value}"
                    output_path = os.path.join(TRANSLATE_OUTPUT_DIR, output_filename)
                    with open(output_path, "wb") as outfile:
                        outfile.write(audio_payload)
                    audio_url = f"/api/translate_outputs/{output_filename}"
                    metadata["audio_file_name"] = output_filename
                    unmixed_audio_bytes = metadata.pop("_unmixed_audio_bytes", None)
                    if unmixed_audio_bytes:
                        unmixed_filename = f"{audio_stem}_vocals.{response_format_value}"
                        unmixed_path = os.path.join(TRANSLATE_OUTPUT_DIR, unmixed_filename)
                        with open(unmixed_path, "wb") as outfile:
                            outfile.write(unmixed_audio_bytes)
                        metadata["translated_vocals_url"] = f"/api/translate_outputs/{unmixed_filename}"
                        metadata["translated_vocals_file_name"] = unmixed_filename
                    subtitle_translated = _export_srt_from_segments(
                        final_segments,
                        base_name=base_stem,
                        suffix=language_code,
                        text_kind="translated",
                        empty_note="No translated speech segments were selected for subtitle export.",
                    )
                    subtitle_original = _export_srt_from_segments(
                        final_segments,
                        base_name=base_stem,
                        suffix="original",
                        text_kind="source",
                        empty_note="No translated speech segments were selected for subtitle export.",
                    )
                    subtitle_url = subtitle_translated["url"] if subtitle_translated else None
                    original_subtitle_url = subtitle_original["url"] if subtitle_original else None
                    metadata["subtitle"] = subtitle_translated
                    metadata["subtitle_translated"] = subtitle_translated
                    metadata["subtitle_original"] = subtitle_original
                    metadata["language_code"] = language_code
                    _apply_source_video_metadata(metadata, session)

                    if chunk_session_for_generation and chunk_session_for_generation.chunk_parent_id:
                        _mark_chunk_generated(
                            chunk_session_for_generation,
                            output_path,
                            output_filename,
                            response_format_value,
                        )
                        chunk_meta = metadata.get("chunk") or {}
                        chunk_meta.update(
                            {
                                "session_id": chunk_session_for_generation.session_id,
                                "chunk_index": chunk_session_for_generation.chunk_index,
                                "start_ms": chunk_session_for_generation.chunk_start_ms,
                                "end_ms": chunk_session_for_generation.chunk_end_ms,
                                "start_label": _format_ms_to_timestamp(chunk_session_for_generation.chunk_start_ms or 0),
                                "end_label": _format_ms_to_timestamp(chunk_session_for_generation.chunk_end_ms or 0),
                                "duration_label": _format_ms_to_timestamp(
                                    max(
                                        0,
                                        (chunk_session_for_generation.chunk_end_ms or 0)
                                        - (chunk_session_for_generation.chunk_start_ms or 0),
                                    )
                                ),
                                "batch_id": chunk_session_for_generation.chunk_parent_id,
                                "cut_reason": chunk_session_for_generation.chunk_cut_reason,
                                "silence_midpoint_ms": chunk_session_for_generation.chunk_silence_midpoint_ms,
                                "backing_available": bool(chunk_session_for_generation.backing_track_audio),
                                "backing_source": chunk_session_for_generation.backing_track_source or "none",
                                "vocals_url": f"/api/translate_vocals/{chunk_session_for_generation.session_id}",
                                "backing_url": (
                                    f"/api/translate_backing_track/{chunk_session_for_generation.session_id}"
                                    if chunk_session_for_generation.backing_track_audio is not None
                                    else None
                                ),
                                "audio_url": audio_url,
                                "output_format": response_format_value,
                                "output_filename": output_filename,
                            }
                        )
                        metadata["chunk"] = chunk_meta

                    await emit(
                        "complete",
                        message="Segment synthesis complete.",
                        audio_url=audio_url,
                        media_type=media_type,
                        headers=headers,
                        metadata=metadata,
                        file_name=output_filename,
                        subtitle_url=subtitle_url,
                        subtitle_file_name=subtitle_translated["filename"] if subtitle_translated else None,
                        original_subtitle_url=original_subtitle_url,
                        original_subtitle_file_name=subtitle_original["filename"] if subtitle_original else None,
                    )

                except TranslateWorkflowHttpError as http_error:
                    await emit(
                        "error",
                        status_code=http_error.status_code,
                        message=http_error.content.get("message") or "Failed to synthesize translation.",
                        details=http_error.content,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    await emit(
                        "error",
                        status_code=500,
                        message=f"Failed to synthesize translation: {str(exc)}",
                    )
                finally:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except asyncio.CancelledError:
                        pass
                    await queue.put(None)

            asyncio.create_task(run_pipeline())

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _json_event_bytes(item)

        return _json_stream_response(generate_stream())
    except RuntimeError as runtime_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(runtime_error)},
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to synthesize translation: {str(exc)}"},
        )


@app.post("/api/translate_segment_preview")
async def api_translate_segment_preview(payload: SegmentPreviewRequest):
    """API: Generate a quick inline preview for a single translated segment."""
    try:
        session = await _get_translate_session(payload.session_id)
        if session is None:
            return _status_error("Translate session not found or expired.", status_code=404)

        seg_input = payload.segment
        if seg_input.type != "speech":
            return _status_error("Only speech segments can be previewed.")

        max_duration = _session_audio_duration_ms(session)
        start_ms = max(0, int(seg_input.start_ms))
        end_ms = max(0, int(seg_input.end_ms))
        if end_ms > max_duration:
            end_ms = max_duration
        if end_ms <= start_ms:
            raise TranslateWorkflowHttpError(
                400,
                {
                    "status": "error",
                    "message": f"Segment index {seg_input.index} has invalid timing (end <= start).",
                },
            )
        duration_ms = end_ms - start_ms
        start_label = _format_ms_to_timestamp(start_ms)
        end_label = _format_ms_to_timestamp(end_ms)
        translated_text = (seg_input.translated_text or "").strip()
        source_text = (seg_input.source_text or "").strip()

        segment_payload: Dict[str, Any] = {
            "index": seg_input.index,
            "type": "speech",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "start": start_label,
            "end": end_label,
            "source_text": source_text,
            "translated_text": translated_text,
            "generate": True,
            "keep_original": False,
        }
        if seg_input.volume_percent is not None:
            segment_payload["volume_percent"] = _coerce_volume_percent(
                seg_input.volume_percent,
                DEFAULT_GENERATED_VOLUME_PERCENT,
            )
        if seg_input.emotion_weight is not None:
            segment_payload["emotion_weight"] = _coerce_emotion_weight(
                seg_input.emotion_weight,
                DEFAULT_EMOTION_WEIGHT,
            )

        base_segment_map = {seg.get("index"): seg for seg in session.base_segments}
        base_info = base_segment_map.get(seg_input.index)
        speaker_label = None
        if base_info and base_info.get("speaker"):
            speaker_label = base_info["speaker"]
        elif seg_input.speaker:
            speaker_label = seg_input.speaker
        if speaker_label:
            segment_payload["speaker"] = speaker_label
        if base_info and base_info.get("text_keys"):
            segment_payload["text_keys"] = base_info["text_keys"]

        override_payload: Dict[str, Dict[str, Any]] = session.speaker_overrides or {}
        if payload.speaker_overrides is not None:
            raw_override_input = {
                key: value.dict(exclude_none=True)
                for key, value in payload.speaker_overrides.items()
            }
            override_payload = _normalize_speaker_overrides(
                raw_override_input,
                session.speaker_profiles,
            )

        volume_percent = session.generated_volume_percent or DEFAULT_GENERATED_VOLUME_PERCENT
        if payload.generated_volume_percent is not None:
            volume_percent = _coerce_volume_percent(payload.generated_volume_percent, volume_percent)
        backing_volume = session.backing_volume_percent or DEFAULT_GENERATED_VOLUME_PERCENT
        if payload.backing_volume_percent is not None:
            backing_volume = _coerce_volume_percent(payload.backing_volume_percent, backing_volume)

        audio_bytes, media_type, metadata = await _synthesize_translated_audio(
            _get_session_original_audio(session),
            [segment_payload],
            session.dest_language,
            response_format=session.response_format or TRANSLATE_DEFAULT_OUTPUT_FORMAT,
            bitrate=session.bitrate or TRANSLATE_DEFAULT_BITRATE,
            input_mime_type=session.input_mime_type,
            clearvoice_settings=session.clearvoice_settings,
            backing_track_audio=None,
            merge_with_backing=False,
            preserve_silence_audio=session.preserve_silence_audio,
            generated_volume_percent=volume_percent,
             silence_volume_percent=session.silence_volume_percent,
            speaker_overrides=override_payload,
            backing_volume_percent=backing_volume,
            pad_to_original=False,
            default_speaker_preset=session.default_speaker_preset,
            default_emotion_weight=session.default_emotion_weight,
        )

        metadata["preview"] = True
        metadata["preview_segment_index"] = seg_input.index
        metadata["default_speaker_preset"] = session.default_speaker_preset
        metadata["default_emotion_weight"] = session.default_emotion_weight

        # Save preview to file and return URL instead of base64
        preview_filename = f"{session.session_id}_segment_{seg_input.index}_preview.mp3"
        preview_path = os.path.join(TRANSLATE_SESSION_MEDIA_DIR, preview_filename)
        
        try:
            with open(preview_path, "wb") as f:
                f.write(audio_bytes)
            cache_bust = int(time.time() * 1000)
            audio_preview_url = (
                f"/api/segment_preview/{session.session_id}/{seg_input.index}"
                f"?variant=preview&ts={cache_bust}"
            )
        except Exception as exc:
            print(f"⚠️ Failed to save preview to file, falling back to base64: {exc}")
            audio_preview_url = f"data:{media_type};base64,{base64.b64encode(audio_bytes).decode('ascii')}"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "segment_index": seg_input.index,
                "audio_preview": audio_preview_url,
                "media_type": media_type,
                "metadata": metadata,
            },
        )
    except TranslateWorkflowHttpError as http_error:
        return JSONResponse(status_code=http_error.status_code, content=http_error.content)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to preview segment: {str(exc)}"},
        )


@app.get("/api/translate_backing_track/{session_id}")
async def api_translate_backing_track(session_id: str):
    """API: Stream the stored instrumental backing track for an advanced translate session."""
    session = await _get_translate_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Translate session not found or expired."},
        )
    # Fast path: serve the persisted MP3 directly (separation output is stored as MP3)
    backing_path = getattr(session, "backing_track_path", None)
    if backing_path and os.path.exists(backing_path) and backing_path.lower().endswith(".mp3"):
        return FileResponse(
            backing_path,
            media_type="audio/mpeg",
            filename=f"translate_backing_{session_id}.mp3",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Fallback to in-memory export if for some reason the path is missing
    backing_audio = _get_session_backing_audio(session)
    if backing_audio is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No backing track is available for this session."},
        )

    try:
        buffer = BytesIO()
        backing_audio.export(buffer, format="mp3", bitrate="128k")
        audio_bytes = buffer.getvalue()
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to export backing track: {str(exc)}"},
        )

    headers = {
        "Content-Disposition": f'inline; filename="translate_backing_{session_id}.mp3"',
        "Cache-Control": "no-store, no-cache",
    }
    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)


@app.get("/api/translate_vocals/{session_id}")
async def api_translate_vocals(session_id: str):
    """API: Stream the separated vocal track for an advanced translate session."""
    session = await _get_translate_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Translate session not found or expired."},
        )
    # Fast path: serve the persisted MP3 directly (separation output is stored as MP3)
    vocals_path = getattr(session, "original_audio_path", None)
    if vocals_path and os.path.exists(vocals_path) and vocals_path.lower().endswith(".mp3"):
        return FileResponse(
            vocals_path,
            media_type="audio/mpeg",
            filename=f"translate_vocals_{session_id}.mp3",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Fallback to in-memory export if for some reason the path is missing
    try:
        original_audio = _get_session_original_audio(session)
    except RuntimeError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No separated vocals are available for this session."},
        )

    try:
        buffer = BytesIO()
        original_audio.export(buffer, format="mp3", bitrate="128k")
        audio_bytes = buffer.getvalue()
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to export vocals: {str(exc)}"},
        )

    headers = {
        "Content-Disposition": f'inline; filename="translate_vocals_{session_id}.mp3"',
        "Cache-Control": "no-store, no-cache",
    }
    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)


@app.get("/api/translate_download_chunks/{batch_id}")
async def api_translate_download_chunks(batch_id: str):
    """API: Download all chunk vocals as a ZIP file for manual transcription."""
    if not batch_id or not batch_id.strip():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "batch_id is required."},
        )

    sessions = await _list_chunk_sessions(batch_id.strip())
    if not sessions:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"No chunks found for batch '{batch_id}'."},
        )

    # Sort sessions by chunk index
    sessions.sort(key=lambda s: s.chunk_index or 0)

    # Create ZIP file in memory
    zip_buffer = BytesIO()
    files_added = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for session in sessions:
            vocals_path = getattr(session, "original_audio_path", None)
            if not vocals_path or not os.path.exists(vocals_path):
                # Try to export from in-memory audio if path is missing
                try:
                    original_audio = _get_session_original_audio(session)
                    audio_buffer = BytesIO()
                    original_audio.export(audio_buffer, format="mp3", bitrate="128k")
                    chunk_name = f"chunk{session.chunk_index or files_added + 1:03d}.mp3"
                    zf.writestr(chunk_name, audio_buffer.getvalue())
                    files_added += 1
                except Exception:
                    continue
            else:
                chunk_name = f"chunk{session.chunk_index or files_added + 1:03d}.mp3"
                zf.write(vocals_path, chunk_name)
                files_added += 1

    if files_added == 0:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No audio files available to download."},
        )

    zip_buffer.seek(0)
    filename = f"chunks_{batch_id}.zip"

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/translate_upload_transcriptions/{batch_id}")
async def api_translate_upload_transcriptions(
    batch_id: str,
    transcriptions_zip: UploadFile = File(...),
    dest_language: str = Form(...),
    gemini_model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    translate_enabled: Optional[bool] = Form(True),
    ignore_non_speech: Optional[bool] = Form(False),
):
    """
    API: Upload a ZIP file with transcription/translation .json files to create Gemini cache entries.
    
    The ZIP should contain files named chunk001.json, chunk002.json, etc.
    Each .json file should contain the raw JSON response from Gemini (or compatible JSON format).
    This creates cache entries so subsequent generation will skip Gemini API calls.
    
    IMPORTANT: The cache key depends on dest_language, gemini_model, translate_enabled, ignore_non_speech,
    and custom prompt. These must match the settings used when generating translated audio.
    """
    if not batch_id or not batch_id.strip():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "batch_id is required."},
        )

    if not dest_language or not dest_language.strip():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "dest_language is required."},
        )

    sessions = await _list_chunk_sessions(batch_id.strip())
    if not sessions:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"No chunks found for batch '{batch_id}'."},
        )

    # Sort sessions by chunk index
    sessions.sort(key=lambda s: s.chunk_index or 0)
    sessions_by_index: Dict[int, TranslateSessionData] = {
        s.chunk_index: s for s in sessions if s.chunk_index is not None
    }

    # Read the ZIP file
    try:
        zip_bytes = await transcriptions_zip.read()
        zip_buffer = BytesIO(zip_bytes)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Failed to read uploaded file: {str(exc)}"},
        )

    # Normalize model name
    model_name = _normalize_gemini_model_name(gemini_model) if gemini_model else _get_gemini_model_name()
    
    # Resolve the prompt the same way it's done when analyzing segments
    # This ensures the cache key matches when analyzing segments later
    translate_enabled_flag = translate_enabled if translate_enabled is not None else True
    ignore_non_speech_flag = ignore_non_speech if ignore_non_speech is not None else False
    prompt_text = _resolve_final_prompt(
        prompt_override=(prompt or "").strip(),
        dest_language=dest_language.strip(),
        translate_enabled=translate_enabled_flag,
        ignore_non_speech_flag=ignore_non_speech_flag,
    )

    created_caches = []
    errors = []

    try:
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            for name in zf.namelist():
                # Skip directories and non-json files
                if name.endswith("/") or not name.lower().endswith(".json"):
                    continue

                # Extract chunk index from filename (chunk001.txt, chunk1.txt, etc.)
                base_name = os.path.splitext(os.path.basename(name))[0].lower()
                match = re.search(r"chunk[_-]?(\d+)", base_name)
                if not match:
                    # Try just extracting any number
                    match = re.search(r"(\d+)", base_name)

                if not match:
                    errors.append(f"Could not determine chunk index from filename: {name}")
                    continue

                chunk_idx = int(match.group(1))
                session = sessions_by_index.get(chunk_idx)

                if session is None:
                    errors.append(f"No session found for chunk index {chunk_idx} (file: {name})")
                    continue

                # Read the transcription text
                try:
                    raw_text = zf.read(name).decode("utf-8")
                except Exception as exc:
                    errors.append(f"Failed to read {name}: {str(exc)}")
                    continue

                if not raw_text.strip():
                    errors.append(f"Empty transcription file: {name}")
                    continue

                # Parse the Gemini JSON to validate and extract segments/speaker profiles
                try:
                    segments, speaker_profiles = _parse_gemini_json(raw_text)
                except Exception as exc:
                    errors.append(f"Failed to parse Gemini JSON in {name}: {str(exc)}")
                    continue

                # Get the audio bytes for this chunk to compute the cache key
                vocals_path = getattr(session, "original_audio_path", None)
                audio_bytes: Optional[bytes] = None

                if vocals_path and os.path.exists(vocals_path):
                    audio_bytes = _read_file_bytes(vocals_path)
                else:
                    try:
                        original_audio = _get_session_original_audio(session)
                        audio_buffer = BytesIO()
                        original_audio.export(audio_buffer, format="mp3", bitrate="128k")
                        audio_bytes = audio_buffer.getvalue()
                    except Exception:
                        pass

                if audio_bytes is None:
                    errors.append(f"Could not get audio bytes for chunk {chunk_idx} to compute cache key")
                    continue

                # Compute the cache key matching how Gemini cache would be created
                audio_hash, cache_key = _gemini_cache_key(
                    audio_bytes,
                    dest_language=dest_language.strip(),
                    prompt_text=prompt_text,
                    model_name=model_name,
                )
                print(f"📦 Chunk {chunk_idx}: audio_md5={audio_hash}, cache_key={cache_key}, model={model_name}")

                # Create the cache record
                cache_record = {
                    "version": GEMINI_CACHE_VERSION,
                    "created_at": time.time(),
                    "audio_md5": audio_hash,
                    "dest_language": dest_language.strip(),
                    "model": model_name,
                    "prompt_hash": hashlib.md5(prompt_text.encode("utf-8")).hexdigest(),
                    "segments": segments,
                    "speaker_profiles": speaker_profiles,
                    "raw_text": raw_text,
                    "imported_from": name,
                }

                # Check if cache already exists (will be overwritten)
                existing_cache = _load_gemini_cache_entry(cache_key)
                was_overwritten = existing_cache is not None

                cache_path = _write_gemini_cache_entry(cache_key, cache_record)
                if cache_path:
                    created_caches.append({
                        "chunk_index": chunk_idx,
                        "session_id": session.session_id,
                        "cache_file": os.path.basename(cache_path),
                        "segments_count": len(segments),
                        "overwritten": was_overwritten,
                    })
                    action = "Updated" if was_overwritten else "Imported"
                    print(f"💾 {action} Gemini cache for chunk {chunk_idx}: {os.path.basename(cache_path)}")
                else:
                    errors.append(f"Failed to write cache entry for chunk {chunk_idx}")

    except zipfile.BadZipFile:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid ZIP file."},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to process ZIP file: {str(exc)}"},
        )

    # Count new vs updated caches
    new_count = sum(1 for c in created_caches if not c.get("overwritten", False))
    updated_count = sum(1 for c in created_caches if c.get("overwritten", False))
    
    if updated_count > 0 and new_count > 0:
        message = f"Imported {new_count} new + updated {updated_count} existing cache entries."
    elif updated_count > 0:
        message = f"Updated {updated_count} existing cache entries."
    else:
        message = f"Created {new_count} cache entries from uploaded transcriptions."

    return JSONResponse(
        content={
            "status": "ok",
            "message": message,
            "created_caches": created_caches,
            "new_count": new_count,
            "updated_count": updated_count,
            "errors": errors,
            "batch_id": batch_id,
            "dest_language": dest_language.strip(),
            "gemini_model": model_name,
            "translate_enabled": translate_enabled_flag,
            "ignore_non_speech": ignore_non_speech_flag,
            "prompt_preview": prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
        },
    )


@app.get("/api/translate_outputs/{filename}")
async def api_translate_outputs(filename: str):
    """Serve generated translate audio files by filename."""
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid filename."},
        )

    file_path = os.path.join(TRANSLATE_OUTPUT_DIR, safe_name)
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Requested audio not found."},
        )

    media_type = _guess_media_type_from_extension(safe_name)
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=safe_name,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/translate_outputs/{filename}/snapshot")
async def api_translate_output_snapshot(filename: str):
    """Serve a generated snapshot for rendered translate output videos."""
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid filename."},
        )
    if os.path.splitext(safe_name)[1].lstrip(".").lower() not in VIDEO_DOWNLOAD_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Snapshots are only available for video outputs."},
        )

    file_path = os.path.join(TRANSLATE_OUTPUT_DIR, safe_name)
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Requested video output not found."},
        )
    try:
        snapshot_path = await _run_blocking(_generate_video_snapshot_sync, file_path)
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Failed to generate video snapshot: {str(exc)}", status_code=500)
    return FileResponse(
        snapshot_path,
        media_type="image/jpeg",
        filename=os.path.basename(snapshot_path),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/translate_audio")
async def api_translate_audio(
    request: Request,
    dest_language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    bitrate: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    downloaded_video_id: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    base_filename: Optional[str] = Form(None),
    custom_backing_audio: Optional[str] = Form(None),
    custom_backing_audio_mime_type: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    translation_llm_model: Optional[str] = Form(None, description="Translation model for WhisperX/Qwen local pipelines."),
    enhance_voice: Optional[bool] = Form(False),
    enhancement_model: Optional[str] = Form(None, description="ClearVoice enhancement model: 'MossFormerGAN_SE_16K' (default), 'FRCRN_SE_16K', or 'MossFormer2_SE_48K'"),
    super_resolution_voice: Optional[bool] = Form(False),
    audio_separator_enabled: Optional[bool] = Form(False, description="Enable audio-separator for vocal/instrumental separation"),
    audio_separator_model: Optional[str] = Form(None, description="Audio-separator model: 'fast', 'balance' (default), or 'quality'"),
    audio_separator_use_soundfile: Optional[bool] = Form(None, description="Use audio-separator soundfile writer for stems (slower fallback)."),
    clearvoice_parallel_enabled: Optional[bool] = Form(False),
    clearvoice_parallel_chunk_seconds: Optional[int] = Form(None),
    clearvoice_parallel_max_workers: Optional[int] = Form(None),
    merge_backing_track: Optional[bool] = Form(False),
    min_speech_ms: Optional[int] = Form(None),
    max_merge_ms: Optional[int] = Form(None),
    segments_json: Optional[str] = Form(None),
    ignore_non_speech: Optional[bool] = Form(False),
    preserve_silence_audio: Optional[bool] = Form(False),
    generated_volume_percent: Optional[float] = Form(None),
    backing_volume_percent: Optional[float] = Form(None),
    silence_volume_percent: Optional[float] = Form(None),
    reuse_session_id: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    custom_backing_audio_file: Optional[UploadFile] = File(None),
    force_gemini_regenerate: Optional[bool] = Form(False),
    default_speaker_preset: Optional[str] = Form(None),
    default_emotion_weight: Optional[float] = Form(None),
    original_srt_file: Optional[UploadFile] = File(None, description="Original language SRT subtitle file"),
    translated_srt_file: Optional[UploadFile] = File(None, description="Translated SRT subtitle file"),
    transcription_pipeline: Optional[str] = Form("qwen_omnivad", description="Transcription pipeline: 'gemini' (default), 'whisperx' (local), 'qwen_omnivad' (Qwen3-ASR + OmniVAD), or 'parakeet' (NVIDIA Parakeet)"),
    whisperx_proxy_refiner: Optional[bool] = Form(False, description="Enable the experimental WhisperX speaker-aware proxy segment refiner."),
    qwen_omnivad_enable_diarization: Optional[bool] = Form(True, description="Enable diarization for Qwen OmniVAD pipeline."),
    qwen_omnivad_diarization_backend: Optional[str] = Form("auto", description="Qwen OmniVAD diarization backend: auto, pyannote, or sortformer."),
    qwen_omnivad_enable_forced_aligner: Optional[bool] = Form(True, description="Enable Qwen3 ForcedAligner timestamps for Qwen OmniVAD pipeline."),
    qwen_omnivad_diarization_min_seconds: Optional[float] = Form(0.0, description="Minimum span duration to split by diarization."),
    qwen_omnivad_merge_gap_seconds: Optional[float] = Form(DEFAULT_QWEN_OMNIVAD_MERGE_GAP_SECONDS, description="Merge adjacent OmniVAD spans separated by this many seconds or less."),
):
    """API: Translate speech audio to a target language and return synthesized audio."""
    reuse_session_id_value: Optional[str] = reuse_session_id
    try:
        payload: Optional[Dict[str, Any]] = None
        dest_language_value = dest_language
        audio_reference = audio
        downloaded_video_id_value = downloaded_video_id
        audio_mime_type_value = audio_mime_type
        base_filename_value = base_filename
        custom_backing_audio_value = custom_backing_audio
        custom_backing_audio_mime_type_value = custom_backing_audio_mime_type
        prompt_override = prompt
        response_format_value = response_format
        bitrate_value = bitrate
        gemini_model_value = gemini_model
        gemini_api_key_value = gemini_api_key
        translation_llm_model_value = translation_llm_model
        enhance_voice_value = enhance_voice
        enhancement_model_value = enhancement_model
        super_resolution_voice_value = super_resolution_voice
        audio_separator_enabled_value = audio_separator_enabled
        audio_separator_model_value = audio_separator_model
        audio_separator_use_soundfile_value = audio_separator_use_soundfile
        clearvoice_parallel_enabled_value = clearvoice_parallel_enabled
        clearvoice_parallel_chunk_seconds_value = clearvoice_parallel_chunk_seconds
        clearvoice_parallel_max_workers_value = clearvoice_parallel_max_workers
        merge_backing_track_value = merge_backing_track
        min_speech_duration_value = min_speech_ms
        max_merge_interval_value = max_merge_ms
        segments_override_value = segments_json
        ignore_non_speech_value = ignore_non_speech
        preserve_silence_audio_value = preserve_silence_audio
        generated_volume_percent_value = generated_volume_percent
        backing_volume_percent_value = backing_volume_percent
        silence_volume_percent_value = silence_volume_percent
        force_gemini_regen_value = force_gemini_regenerate
        default_speaker_value = default_speaker_preset
        default_emotion_weight_value = default_emotion_weight
        transcription_pipeline_value = transcription_pipeline
        whisperx_proxy_refiner_value = whisperx_proxy_refiner
        qwen_omnivad_diarization_backend_value = qwen_omnivad_diarization_backend
        qwen_omnivad_enable_forced_aligner_value = qwen_omnivad_enable_forced_aligner
        qwen_omnivad_merge_gap_seconds_value = qwen_omnivad_merge_gap_seconds

        payload, payload_error = await _read_optional_json_payload(
            request,
            dest_language is None
            and not audio_reference
            and not downloaded_video_id_value
            and audio_file is None
            and _request_has_json_body(request),
        )
        if payload_error is not None:
            return payload_error

        if payload is not None:
            try:
                translate_req = TranslateRequest(**payload)
            except Exception as exc:
                return _status_error(f"Invalid translate request: {str(exc)}")
            dest_language_value = translate_req.dest_language
            audio_reference = translate_req.audio or audio_reference
            downloaded_video_id_value = translate_req.downloaded_video_id or downloaded_video_id_value
            audio_mime_type_value = translate_req.audio_mime_type or audio_mime_type_value
            base_filename_value = translate_req.base_filename or base_filename_value
            custom_backing_audio_value = translate_req.custom_backing_audio or custom_backing_audio_value
            custom_backing_audio_mime_type_value = (
                translate_req.custom_backing_audio_mime_type or custom_backing_audio_mime_type_value
            )
            prompt_override = translate_req.prompt or prompt_override
            response_format_value = translate_req.response_format or response_format_value
            bitrate_value = translate_req.bitrate or bitrate_value
            gemini_model_value = translate_req.gemini_model or gemini_model_value
            gemini_api_key_value = translate_req.gemini_api_key or gemini_api_key_value
            translation_llm_model_value = translate_req.translation_llm_model or translation_llm_model_value
            enhance_voice_value = translate_req.enhance_voice if translate_req.enhance_voice is not None else enhance_voice_value
            super_resolution_voice_value = (
                translate_req.super_resolution_voice if translate_req.super_resolution_voice is not None else super_resolution_voice_value
            )
            audio_separator_enabled_value = (
                translate_req.audio_separator_enabled
                if translate_req.audio_separator_enabled is not None
                else audio_separator_enabled_value
            )
            audio_separator_model_value = translate_req.audio_separator_model or audio_separator_model_value
            if translate_req.audio_separator_use_soundfile is not None:
                audio_separator_use_soundfile_value = translate_req.audio_separator_use_soundfile
            if translate_req.clearvoice_parallel_enabled is not None:
                clearvoice_parallel_enabled_value = translate_req.clearvoice_parallel_enabled
            if translate_req.clearvoice_parallel_chunk_seconds is not None:
                clearvoice_parallel_chunk_seconds_value = translate_req.clearvoice_parallel_chunk_seconds
            if translate_req.clearvoice_parallel_max_workers is not None:
                clearvoice_parallel_max_workers_value = translate_req.clearvoice_parallel_max_workers
            merge_backing_track_value = (
                translate_req.merge_backing_track if translate_req.merge_backing_track is not None else merge_backing_track_value
            )
            min_speech_duration_value = (
                translate_req.min_speech_ms if translate_req.min_speech_ms is not None else min_speech_duration_value
            )
            max_merge_interval_value = (
                translate_req.max_merge_ms if translate_req.max_merge_ms is not None else max_merge_interval_value
            )
            segments_override_value = (
                translate_req.segments_json if translate_req.segments_json is not None else segments_override_value
            )
            ignore_non_speech_value = (
                translate_req.ignore_non_speech if translate_req.ignore_non_speech is not None else ignore_non_speech_value
            )
            preserve_silence_audio_value = (
                translate_req.preserve_silence_audio
                if translate_req.preserve_silence_audio is not None
                else preserve_silence_audio_value
            )
            generated_volume_percent_value = (
                translate_req.generated_volume_percent
                if translate_req.generated_volume_percent is not None
                else generated_volume_percent_value
            )
            backing_volume_percent_value = (
                translate_req.backing_volume_percent
                if translate_req.backing_volume_percent is not None
                else backing_volume_percent_value
            )
            silence_volume_percent_value = (
                translate_req.silence_volume_percent
                if translate_req.silence_volume_percent is not None
                else silence_volume_percent_value
            )
            reuse_session_id_value = translate_req.reuse_session_id or reuse_session_id_value
            force_gemini_regen_value = (
                translate_req.force_gemini_regenerate
                if translate_req.force_gemini_regenerate is not None
                else force_gemini_regen_value
            )
            if translate_req.default_speaker_preset:
                default_speaker_value = translate_req.default_speaker_preset
            if translate_req.default_emotion_weight is not None:
                default_emotion_weight_value = translate_req.default_emotion_weight
            if translate_req.transcription_pipeline:
                transcription_pipeline_value = translate_req.transcription_pipeline
            if translate_req.whisperx_proxy_refiner is not None:
                whisperx_proxy_refiner_value = translate_req.whisperx_proxy_refiner
            if translate_req.qwen_omnivad_enable_diarization is not None:
                qwen_omnivad_enable_diarization = translate_req.qwen_omnivad_enable_diarization
            if translate_req.qwen_omnivad_diarization_backend is not None:
                qwen_omnivad_diarization_backend_value = translate_req.qwen_omnivad_diarization_backend
            if translate_req.qwen_omnivad_enable_forced_aligner is not None:
                qwen_omnivad_enable_forced_aligner_value = translate_req.qwen_omnivad_enable_forced_aligner
            if translate_req.qwen_omnivad_diarization_min_seconds is not None:
                qwen_omnivad_diarization_min_seconds = translate_req.qwen_omnivad_diarization_min_seconds
            if translate_req.qwen_omnivad_merge_gap_seconds is not None:
                qwen_omnivad_merge_gap_seconds_value = translate_req.qwen_omnivad_merge_gap_seconds

        default_speaker_value = (default_speaker_value or "").strip()
        if not default_speaker_value:
            default_speaker_value = None
        default_emotion_weight_value = _coerce_emotion_weight(
            default_emotion_weight_value,
            DEFAULT_EMOTION_WEIGHT,
        )
        resolved_transcription_pipeline = _normalize_transcription_pipeline(
            transcription_pipeline_value
        )
        whisperx_proxy_refiner_flag = (
            _coerce_to_bool(whisperx_proxy_refiner_value)
            and resolved_transcription_pipeline == "whisperx"
        )
        (
            qwen_omnivad_enable_diarization_flag,
            qwen_omnivad_diarization_min_seconds_value,
        ) = _resolve_qwen_omnivad_diarization_options(
            resolved_transcription_pipeline,
            qwen_omnivad_enable_diarization,
            qwen_omnivad_diarization_min_seconds,
        )
        qwen_omnivad_diarization_backend_value = _resolve_qwen_omnivad_diarization_backend_option(
            resolved_transcription_pipeline,
            qwen_omnivad_diarization_backend_value,
        )
        qwen_omnivad_enable_forced_aligner_flag = _resolve_qwen_omnivad_forced_aligner_option(
            resolved_transcription_pipeline,
            qwen_omnivad_enable_forced_aligner_value,
        )
        qwen_omnivad_merge_gap_seconds_value = _resolve_qwen_omnivad_merge_gap_seconds_option(
            resolved_transcription_pipeline,
            qwen_omnivad_merge_gap_seconds_value,
        )

        srt_segments_from_upload, srt_upload_error = await _load_srt_segments_override(
            original_srt_file,
            translated_srt_file,
            log_prefix="translate_audio",
        )
        if srt_upload_error is not None:
            return srt_upload_error
        if srt_segments_from_upload:
            segments_override_value = srt_segments_from_upload

        reuse_session_id_value, reuse_source_session, reuse_session_error = await _resolve_reuse_translate_session(
            reuse_session_id_value
        )
        if reuse_session_error is not None:
            return reuse_session_error

        dest_language_value = (dest_language_value or "").strip()
        if not dest_language_value:
            return _status_error("Destination language (dest_language) is required.")

        response_format_value = _normalize_translate_output_format(response_format_value)
        bitrate_value = bitrate_value or TRANSLATE_DEFAULT_BITRATE
        ignore_non_speech_flag = _coerce_to_bool(ignore_non_speech_value)
        preserve_silence_audio_flag = _coerce_to_bool(preserve_silence_audio_value)
        preprocess_options = _resolve_audio_preprocess_options(
            enhance_voice=enhance_voice_value,
            super_resolution_voice=super_resolution_voice_value,
            enhancement_model=enhancement_model_value,
            audio_separator_enabled=audio_separator_enabled_value,
            audio_separator_model=audio_separator_model_value,
            audio_separator_use_soundfile=audio_separator_use_soundfile_value,
            clearvoice_parallel_enabled=clearvoice_parallel_enabled_value,
            clearvoice_parallel_chunk_seconds=clearvoice_parallel_chunk_seconds_value,
            clearvoice_parallel_max_workers=clearvoice_parallel_max_workers_value,
        )
        apply_enhancement = preprocess_options.apply_enhancement
        apply_super_resolution = preprocess_options.apply_super_resolution
        parallel_config = preprocess_options.parallel_config
        custom_backing_present = bool(custom_backing_audio_file) or bool((custom_backing_audio_value or "").strip())
        merge_backing_requested_raw = _coerce_to_bool(merge_backing_track_value)
        reuse_backing_available = bool(reuse_source_session and _session_has_backing_audio(reuse_source_session))
        audio_separator_enabled_flag = preprocess_options.audio_separator_enabled
        audio_separator_model_key = preprocess_options.audio_separator_model
        requested_merge_backing = _coerce_merge_backing_flag(
            merge_backing_requested_raw,
            apply_enhancement,
            custom_backing_present or reuse_backing_available,
            audio_separator_enabled=audio_separator_enabled_flag,
        )
        if reuse_source_session is not None and reuse_source_session.chunk_parent_id:
            requested_merge_backing = False
        enhancement_model_name_value = preprocess_options.enhancement_model_name
        clearvoice_settings = preprocess_options.to_clearvoice_settings()
        min_speech_duration = _coerce_positive_int(
            min_speech_duration_value,
            MIN_SPEECH_DURATION_MS,
            min_value=500,
        )
        max_merge_interval = _coerce_positive_int(
            max_merge_interval_value,
            MAX_MERGE_INTERVAL_MS,
            min_value=0,
        )
        volume_options = _resolve_volume_options(
            generated_volume_percent_value,
            backing_volume_percent_value,
            silence_volume_percent_value,
        )
        generated_volume_percent_value = volume_options.generated
        backing_volume_percent_value = volume_options.backing
        silence_volume_percent_value = volume_options.silence
        resolved_gemini_model = _normalize_gemini_model_name(gemini_model_value)
        resolved_translation_llm_model = _normalize_translation_llm_model(translation_llm_model_value)
        gemini_api_key_value = (gemini_api_key_value or "").strip()
        force_gemini_regenerate_flag = _coerce_to_bool(force_gemini_regen_value or False)


        uploaded_filename = audio_file.filename if audio_file else None
        audio_bytes: Optional[bytes] = None
        downloaded_video_source: Optional[Dict[str, Any]] = None
        if reuse_source_session is None:
            if (downloaded_video_id_value or "").strip():
                (
                    downloaded_video_source,
                    audio_bytes,
                    downloaded_audio_filename,
                    downloaded_error,
                ) = await _prepare_downloaded_video_audio_request(downloaded_video_id_value)
                if downloaded_error is not None:
                    return downloaded_error
                audio_file = None
                audio_reference = None
                audio_mime_type_value = "audio/mpeg"
                uploaded_filename = downloaded_audio_filename or uploaded_filename
            else:
                audio_bytes, audio_error = await _read_audio_request_bytes(
                    audio_file,
                    audio_reference,
                    missing_message="No audio provided for translation.",
                    empty_message="Provided audio data is empty.",
                )
                if audio_error is not None:
                    return audio_error

        input_mime_type = audio_mime_type_value or (audio_file.content_type if audio_file else None) or "audio/wav"

        reuse_session_for_translate = reuse_source_session
        session_source_filename = uploaded_filename or (
            reuse_session_for_translate.source_audio_filename if reuse_session_for_translate else None
        )
        resolved_base_name = _determine_output_base_name(
            user_base=base_filename_value,
            upload_filename=session_source_filename,
            reuse_session=reuse_session_for_translate,
        )

        request_summary = {
            "dest_language": dest_language_value,
            "response_format": response_format_value,
            "bitrate": bitrate_value,
            "enhancement": apply_enhancement,
            "super_resolution": apply_super_resolution,
            "audio_separator_enabled": audio_separator_enabled_flag,
            "audio_separator_model": audio_separator_model_key if audio_separator_enabled_flag else None,
            "audio_separator_use_soundfile": (
                preprocess_options.audio_separator_use_soundfile if audio_separator_enabled_flag else None
            ),
            "clearvoice_parallel": parallel_config.to_metadata(),
            "merge_backing": requested_merge_backing,
            "custom_backing": custom_backing_present,
            "ignore_non_speech": ignore_non_speech_flag,
            "preserve_silence_audio": preserve_silence_audio_flag,
            "generated_volume_percent": generated_volume_percent_value,
            "backing_volume_percent": backing_volume_percent_value,
            "silence_volume_percent": silence_volume_percent_value,
            "reuse_session": bool(reuse_source_session),
            "base_output_name": resolved_base_name,
            "downloaded_video": _public_downloaded_video_source(downloaded_video_source),
            "force_gemini_regenerate": force_gemini_regenerate_flag,
            "translation_llm_model": resolved_translation_llm_model,
            "transcription_pipeline": resolved_transcription_pipeline,
            "whisperx_proxy_refiner": whisperx_proxy_refiner_flag,
            "qwen_omnivad_enable_diarization": qwen_omnivad_enable_diarization_flag,
            "qwen_omnivad_diarization_backend": qwen_omnivad_diarization_backend_value,
            "qwen_omnivad_diarization_min_seconds": qwen_omnivad_diarization_min_seconds_value,
            "qwen_omnivad_enable_forced_aligner": qwen_omnivad_enable_forced_aligner_flag,
            "qwen_omnivad_merge_gap_seconds": qwen_omnivad_merge_gap_seconds_value,
        }
        print(
            "[translate_audio] dest=%s reuse_session=%s chunk_index=%s upload=%s format=%s backing=%s merge_backing=%s"
            % (
                dest_language_value,
                bool(reuse_session_for_translate),
                getattr(reuse_session_for_translate, "chunk_index", None),
                bool(audio_file and not reuse_session_for_translate),
                response_format_value,
                custom_backing_present
                or bool(reuse_session_for_translate and _session_has_backing_audio(reuse_session_for_translate)),
                requested_merge_backing,
            )
        )

        async def translate_stream():
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

            async def emit(event_type: str, **payload: Any) -> None:
                event = {
                    "event": event_type,
                    "timestamp": time.time(),
                    **payload,
                }
                await queue.put(event)

            async def emit_status(**payload: Any) -> None:
                await emit("status", **payload)

            async def heartbeat_task():
                try:
                    while True:
                        await asyncio.sleep(10)
                        await emit_status(stage="heartbeat", message="Still processing translation...")
                except asyncio.CancelledError:
                    pass

            async def run_pipeline():
                heartbeat = asyncio.create_task(heartbeat_task())
                try:
                    # Detect if using SRT subtitles
                    srt_mode_active = bool(srt_segments_from_upload)
                    start_message = "Translate request accepted."
                    if srt_mode_active:
                        # Count segments from SRT
                        try:
                            srt_parsed = json.loads(srt_segments_from_upload) if srt_segments_from_upload else {}
                            srt_seg_count = len(srt_parsed.get("segments", []))
                            start_message = f"Using SRT subtitles ({srt_seg_count} segments) - skipping Gemini inference."
                        except:
                            start_message = "Using SRT subtitles - skipping Gemini inference."
                    
                    await emit_status(stage="start", message=start_message, summary=request_summary)
                    print(
                        "[translate_audio] pipeline start reuse_session=%s chunk_id=%s srt_mode=%s"
                        % (
                            reuse_session_for_translate.session_id if reuse_session_for_translate else None,
                            getattr(reuse_session_for_translate, "chunk_index", None),
                            srt_mode_active,
                        )
                    )

                    session: Optional[TranslateSessionData] = None
                    (
                        original_audio,
                        input_mime_type_resolved,
                        processed_audio_bytes,
                        gemini_mime_type,
                        backing_track_audio,
                        merge_with_backing,
                        backing_track_source,
                        cached_vocals_path,
                        cached_backing_path,
                    ) = await _prepare_audio_assets(
                        reuse_source_session=reuse_session_for_translate,
                        audio_file=None if reuse_session_for_translate else audio_file,
                        audio_reference=None if reuse_session_for_translate else audio_reference,
                        preloaded_audio_bytes=audio_bytes,
                        source_audio_filename=session_source_filename,
                        audio_mime_type_value=input_mime_type,
                        apply_enhancement=apply_enhancement,
                        apply_super_resolution=apply_super_resolution,
                        requested_merge_backing=requested_merge_backing,
                        custom_backing_audio_file=custom_backing_audio_file,
                        custom_backing_audio_reference=custom_backing_audio_value,
                        custom_backing_mime_type_value=custom_backing_audio_mime_type_value,
                        emit_status=emit_status,
                        clearvoice_parallel_config=parallel_config if parallel_config.enabled else None,
                        enhancement_model_name=enhancement_model_name_value,
                        audio_separator_enabled=audio_separator_enabled_flag,
                        audio_separator_model=audio_separator_model_key,
                        audio_separator_use_soundfile=preprocess_options.audio_separator_use_soundfile,
                    )
                    input_mime_type_local = input_mime_type_resolved or input_mime_type

                    final_prompt = _resolve_final_prompt(
                        prompt_override,
                        dest_language_value,
                        True,
                        ignore_non_speech_flag,
                    )

                    segment_result = await _build_translation_segments(
                        original_audio=original_audio,
                        processed_audio_bytes=processed_audio_bytes,
                        gemini_mime_type=gemini_mime_type,
                        dest_language=dest_language_value,
                        final_prompt=final_prompt,
                        translate_enabled=True,
                        response_format_value=response_format_value,
                        bitrate_value=bitrate_value,
                        input_mime_type=input_mime_type_local,
                        apply_enhancement=apply_enhancement,
                        apply_super_resolution=apply_super_resolution,
                        ignore_non_speech_flag=ignore_non_speech_flag,
                        preserve_silence_audio_flag=preserve_silence_audio_flag,
                        generated_volume_percent_value=generated_volume_percent_value,
                        backing_volume_percent_value=backing_volume_percent_value,
                        silence_volume_percent_value=silence_volume_percent_value,
                        backing_track_audio=backing_track_audio,
                        backing_track_source=backing_track_source,
                        merge_with_backing=merge_with_backing,
                        segments_override_value=segments_override_value,
                        min_speech_duration=min_speech_duration,
                        max_merge_interval=max_merge_interval,
                        resolved_gemini_model=resolved_gemini_model,
                        gemini_api_key_value=gemini_api_key_value or None,
                        translation_llm_model=resolved_translation_llm_model,
                        emit_status=emit_status,
                        source_chunk_session=reuse_session_for_translate,
                        source_audio_filename=session_source_filename,
                        source_base_name=resolved_base_name,
                        source_video_path=downloaded_video_source.get("path") if downloaded_video_source else None,
                        source_video_filename=downloaded_video_source.get("filename") if downloaded_video_source else None,
                        force_gemini_regenerate=force_gemini_regenerate_flag,
                        initial_speaker_overrides=getattr(reuse_session_for_translate, "speaker_overrides", None),
                        default_speaker_preset=default_speaker_value,
                        default_emotion_weight=default_emotion_weight_value,
                        clearvoice_settings=clearvoice_settings,
                        # Pass cached paths for fast session storage
                        original_audio_source_path=cached_vocals_path,
                        backing_track_source_path=cached_backing_path,
                        transcription_pipeline=resolved_transcription_pipeline,
                        whisperx_proxy_refiner=whisperx_proxy_refiner_flag,
                        qwen_omnivad_enable_diarization=qwen_omnivad_enable_diarization_flag,
                        qwen_omnivad_diarization_backend=qwen_omnivad_diarization_backend_value,
                        qwen_omnivad_diarization_min_seconds=qwen_omnivad_diarization_min_seconds_value,
                        qwen_omnivad_enable_forced_aligner=qwen_omnivad_enable_forced_aligner_flag,
                        qwen_omnivad_merge_gap_seconds=qwen_omnivad_merge_gap_seconds_value,
                    )
                    session = segment_result.session
                    segments = segment_result.segments

                    speech_count = sum(1 for s in segments if s.get("type") == "speech")
                    await emit_status(
                        stage="synthesis",
                        message=f"Synthesizing translated speech ({speech_count} speech segments)...",
                    )

                    audio_payload, media_type, synthesis_metadata = await _synthesize_translated_audio(
                        original_audio,
                        segments,
                        dest_language_value,
                        response_format=response_format_value,
                        bitrate=bitrate_value,
                        input_mime_type=input_mime_type_local,
                        clearvoice_settings={
                            "enhancement": apply_enhancement,
                            "super_resolution": apply_super_resolution,
                        },
                        backing_track_audio=backing_track_audio,
                        backing_track_source=backing_track_source,
                        merge_with_backing=merge_with_backing,
                        preserve_silence_audio=preserve_silence_audio_flag,
                        generated_volume_percent=generated_volume_percent_value,
                        silence_volume_percent=silence_volume_percent_value,
                        speaker_overrides=session.speaker_overrides,
                        backing_volume_percent=backing_volume_percent_value,
                        default_speaker_preset=session.default_speaker_preset,
                        default_emotion_weight=session.default_emotion_weight,
                        emit_status=emit_status,
                        return_unmixed_audio=merge_with_backing and backing_track_audio is not None,
                    )

                    metadata = dict(segment_result.metadata)
                    metadata["output_base_name"] = resolved_base_name
                    if session_source_filename:
                        metadata["source_audio_filename"] = session_source_filename
                    metadata.update(synthesis_metadata or {})
                    metadata["ignore_non_speech"] = ignore_non_speech_flag
                    metadata["preserve_silence_audio"] = preserve_silence_audio_flag
                    metadata["generated_volume_percent"] = generated_volume_percent_value
                    metadata["backing_volume_percent"] = backing_volume_percent_value
                    metadata["silence_volume_percent"] = silence_volume_percent_value
                    metadata["gemini_raw_segments"] = segment_result.gemini_chunks
                    if segment_result.gemini_raw_text is not None:
                        metadata["gemini_raw_text"] = segment_result.gemini_raw_text
                    if session is not None:
                        metadata["speaker_overrides"] = copy.deepcopy(session.speaker_overrides)
                    else:
                        metadata["speaker_overrides"] = {}
                    metadata["default_speaker_preset"] = session.default_speaker_preset if session else None
                    metadata["default_emotion_weight"] = session.default_emotion_weight if session else DEFAULT_EMOTION_WEIGHT
                    backing_meta = metadata.setdefault("backing_track", {})
                    backing_meta["requested"] = requested_merge_backing
                    backing_meta["available"] = backing_track_audio is not None
                    backing_meta["merged"] = merge_with_backing
                    backing_meta["volume_percent"] = backing_volume_percent_value
                    backing_meta.setdefault("source", backing_track_source or ("custom" if custom_backing_present else "extracted" if apply_enhancement else "none"))
                    metadata["segment_rules"] = {
                        "min_speech_ms": min_speech_duration,
                        "max_merge_ms": max_merge_interval,
                    }
                    metadata["gemini_model"] = resolved_gemini_model
                    if session is not None:
                        metadata["session_id"] = session.session_id
                        metadata["reuse_session_id"] = session.session_id
                        metadata["separation"] = {
                            "vocals_available": True,
                            "vocals_url": f"/api/translate_vocals/{session.session_id}",
                            "backing_available": backing_track_audio is not None,
                            "backing_url": f"/api/translate_backing_track/{session.session_id}" if backing_track_audio is not None else None,
                            "session_id": session.session_id,
                        }

                    headers = {
                        "X-Translation-Model": resolved_gemini_model,
                        "X-Translation-LLM": resolved_translation_llm_model,
                        "X-Translation-Segments": str(metadata.get("segment_count", len(segments))),
                        "X-Translation-Speech-Segments": str(metadata.get("speech_segment_count", 0)),
                        "X-Translation-Silence-Segments": str(metadata.get("silence_segment_count", 0)),
                        "X-Translation-Input-Mime": input_mime_type_local or "",
                        "X-Translation-ClearVoice": (
                            f"enhancement={str(apply_enhancement).lower()};super_resolution={str(apply_super_resolution).lower()}"
                        ),
                        "X-Translation-Backing": (
                            f"available={str(bool(backing_track_audio)).lower()};merged={str(merge_with_backing).lower()};"
                            f"volume_percent={backing_volume_percent_value:.2f}"
                        ),
                    }

                    chunk_index_for_name = (
                        reuse_session_for_translate.chunk_index
                        if reuse_session_for_translate and reuse_session_for_translate.chunk_parent_id
                        else None
                    )
                    language_code = _language_code_from_label(dest_language_value)
                    base_stem = _compose_output_stem(
                        resolved_base_name,
                        chunk_index=chunk_index_for_name,
                    )
                    audio_stem = _compose_output_stem(
                        resolved_base_name,
                        chunk_index=chunk_index_for_name,
                        extra=language_code,
                    )
                    output_filename = f"{audio_stem}.{response_format_value}"
                    output_path = os.path.join(TRANSLATE_OUTPUT_DIR, output_filename)
                    with open(output_path, "wb") as outfile:
                        outfile.write(audio_payload)
                    audio_url = f"/api/translate_outputs/{output_filename}"
                    metadata["audio_file_name"] = output_filename
                    unmixed_audio_bytes = metadata.pop("_unmixed_audio_bytes", None)
                    if unmixed_audio_bytes:
                        unmixed_filename = f"{audio_stem}_vocals.{response_format_value}"
                        unmixed_path = os.path.join(TRANSLATE_OUTPUT_DIR, unmixed_filename)
                        with open(unmixed_path, "wb") as outfile:
                            outfile.write(unmixed_audio_bytes)
                        metadata["translated_vocals_url"] = f"/api/translate_outputs/{unmixed_filename}"
                        metadata["translated_vocals_file_name"] = unmixed_filename
                    subtitle_translated = _export_srt_from_segments(
                        segments,
                        base_name=base_stem,
                        suffix=language_code,
                        text_kind="translated",
                        empty_note="No speech segments were available for subtitle export.",
                    )
                    subtitle_original = _export_srt_from_segments(
                        segments,
                        base_name=base_stem,
                        suffix="original",
                        text_kind="source",
                        empty_note="No speech segments were available for subtitle export.",
                    )
                    subtitle_url = subtitle_translated["url"] if subtitle_translated else None
                    original_subtitle_url = subtitle_original["url"] if subtitle_original else None
                    if subtitle_translated:
                        metadata["subtitle"] = subtitle_translated
                        metadata["subtitle_translated"] = subtitle_translated
                    if subtitle_original:
                        metadata["subtitle_original"] = subtitle_original
                    metadata["language_code"] = language_code
                    _apply_source_video_metadata(metadata, session)

                    if reuse_session_for_translate and reuse_session_for_translate.chunk_parent_id:
                        _mark_chunk_generated(
                            reuse_session_for_translate,
                            output_path,
                            output_filename,
                            response_format_value,
                        )
                        chunk_meta = metadata.get("chunk") or {}
                        chunk_meta.update(
                            {
                                "session_id": reuse_session_for_translate.session_id,
                                "chunk_index": reuse_session_for_translate.chunk_index,
                                "start_ms": reuse_session_for_translate.chunk_start_ms,
                                "end_ms": reuse_session_for_translate.chunk_end_ms,
                                "start_label": _format_ms_to_timestamp(reuse_session_for_translate.chunk_start_ms or 0),
                                "end_label": _format_ms_to_timestamp(reuse_session_for_translate.chunk_end_ms or 0),
                                "duration_label": _format_ms_to_timestamp(
                                    max(
                                        0,
                                        (reuse_session_for_translate.chunk_end_ms or 0)
                                        - (reuse_session_for_translate.chunk_start_ms or 0),
                                    )
                                ),
                                "batch_id": reuse_session_for_translate.chunk_parent_id,
                                "cut_reason": reuse_session_for_translate.chunk_cut_reason,
                                "silence_midpoint_ms": reuse_session_for_translate.chunk_silence_midpoint_ms,
                                "backing_available": _session_has_backing_audio(reuse_session_for_translate),
                                "backing_source": reuse_session_for_translate.backing_track_source or "none",
                                "vocals_url": f"/api/translate_vocals/{reuse_session_for_translate.session_id}",
                                "backing_url": (
                                    f"/api/translate_backing_track/{reuse_session_for_translate.session_id}"
                                    if _session_has_backing_audio(reuse_session_for_translate)
                                    else None
                                ),
                                "audio_url": audio_url,
                                "output_format": response_format_value,
                                "output_filename": output_filename,
                            }
                        )
                        metadata["chunk"] = chunk_meta

                    await emit(
                        "complete",
                        message="Translation complete.",
                        audio_url=audio_url,
                        media_type=media_type,
                        headers=headers,
                        metadata=metadata,
                        file_name=output_filename,
                        subtitle_url=subtitle_url,
                        subtitle_file_name=subtitle_translated["filename"] if subtitle_translated else None,
                        original_subtitle_url=original_subtitle_url,
                        original_subtitle_file_name=subtitle_original["filename"] if subtitle_original else None,
                    )
                    print(
                        "[translate_audio] complete audio_url=%s chunk_id=%s generated=%s"
                        % (
                            audio_url,
                            getattr(reuse_session_for_translate, "session_id", None)
                            if reuse_session_for_translate
                            else (session.session_id if session else None),
                            bool(reuse_session_for_translate and reuse_session_for_translate.chunk_parent_id),
                        )
                    )

                except TranslateWorkflowHttpError as http_error:
                    await emit(
                        "error",
                        status_code=http_error.status_code,
                        message=http_error.content.get("message") or "Translation failed.",
                        details=http_error.content,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    await emit(
                        "error",
                        status_code=500,
                        message=f"Translation failed: {str(exc)}",
                    )
                finally:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except asyncio.CancelledError:
                        pass
                    await queue.put(None)
            asyncio.create_task(run_pipeline())

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _json_event_bytes(item)

        return _json_stream_response(translate_stream())

    except RuntimeError as runtime_error:
        return _status_error(str(runtime_error), status_code=500)
    except Exception as exc:
        traceback.print_exc()
        return _status_error(f"Translation failed: {str(exc)}", status_code=500)


# Audio input helpers
async def get_audio_bytes_from_url(url: str) -> bytes:
    """Download audio from URL"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Cannot download audio from URL")
            return response.content
    except ImportError:
        # Fallback if httpx is not available - run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def download_sync():
            try:
                with urllib.request.urlopen(url) as response:
                    if response.status != 200:
                        raise Exception("Cannot download audio from URL")
                    return response.read()
            except Exception as e:
                raise Exception(f"Failed to download audio: {str(e)}")
        
        try:
            return await loop.run_in_executor(executor, download_sync)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

async def load_base64_or_url(audio: str) -> BytesIO:
    """Load audio from base64 or URL"""
    if audio.startswith("http://") or audio.startswith("https://"):
        audio_bytes = await get_audio_bytes_from_url(audio)
    else:
        payload = audio.strip()
        if payload.startswith("data:"):
            try:
                _, payload = payload.split(",", 1)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid data URI audio: {str(e)}") from e
        try:
            audio_bytes = base64.b64decode(payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
    
    return BytesIO(audio_bytes)

async def load_audio_bytes_from_request(audio_file, audio):
    """Load audio bytes from file or reference audio string"""
    if audio_file is None:
        if audio is None:
            return None
        return await load_base64_or_url(audio)
    else:
        content = await audio_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Reference audio file is empty")
        return BytesIO(content)


async def _read_audio_request_bytes(
    audio_file: Optional[UploadFile],
    audio_reference: Optional[str],
    *,
    missing_message: str,
    empty_message: str,
) -> Tuple[Optional[bytes], Optional[JSONResponse]]:
    try:
        audio_io = await load_audio_bytes_from_request(audio_file, audio_reference)
    except HTTPException as exc:
        return None, _status_error(str(exc.detail), status_code=exc.status_code)

    if audio_io is None:
        return None, _status_error(missing_message)

    audio_bytes = audio_io.read()
    if not audio_bytes:
        return None, _status_error(empty_message)

    return audio_bytes, None


async def _persist_reference_audio_to_tempfile(
    reference_audio_file: Optional[UploadFile],
    reference_audio: Optional[str],
) -> Tuple[Optional[str], Optional[JSONResponse]]:
    audio_io = await load_audio_bytes_from_request(reference_audio_file, reference_audio)
    if audio_io is None:
        return None, _success_error("No reference audio provided", status_code=400)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    await async_write_file(tmp_path, audio_io.read())
    return tmp_path, None


# API Compatibility Endpoints
@app.post("/add_speaker")
async def add_speaker(
    name: str = Form(..., description="The name of the speaker"),
    audio: Optional[str] = Form(None, description="Reference audio URL or base64"),
    reference_text: Optional[str] = Form(None, description="Optional transcript"),
    audio_file: Optional[UploadFile] = File(None, description="Upload reference audio file"),
    enhance_voice: bool = Form(False, description="Apply ClearVoice speech enhancement"),
    enhancement_model: Optional[str] = Form(None, description="ClearVoice enhancement model: 'MossFormerGAN_SE_16K' (default), 'FRCRN_SE_16K', or 'MossFormer2_SE_48K'"),
    super_resolution_voice: bool = Form(False, description="Apply ClearVoice MossFormer2_SR_48K super-resolution"),
):
    """API: Add a new speaker"""
    try:
        print(f"🎭 API: Adding speaker '{name}'")
        
        if not speaker_api:
            return _success_error("Speaker manager not initialized")
        
        # Load audio from file upload or base64/URL reference.
        try:
            audio_io = await load_audio_bytes_from_request(audio_file, audio)
            if audio_io is None:
                print(f"❌ API: No audio provided for speaker '{name}'")
                return _success_error("No audio provided", status_code=400)
        except Exception as audio_error:
            print(f"❌ API: Audio loading failed for speaker '{name}': {audio_error}")
            return _success_error(f"Audio loading failed: {str(audio_error)}", status_code=400)
        
        # Get audio data and filename
        audio_data = audio_io.read()
        filename = audio_file.filename if audio_file else f"{name}_reference.wav"
        
        apply_enhancement = bool(enhance_voice)
        apply_super_resolution_flag = bool(super_resolution_voice)
        print(f"🎚️ API: ClearVoice options -> enhancement={apply_enhancement}, super_resolution={apply_super_resolution_flag}")
        
        if (apply_enhancement or apply_super_resolution_flag) and ClearVoice is None:
            error_msg = "ClearVoice is required for enhancement or super-resolution. Install the `clearvoice` package to enable these options."
            print(f"❌ API: {error_msg}")
            return _success_error(error_msg)
        
        # Add speaker using SpeakerPresetManager (handles ClearVoice processing internally)
        result = await speaker_api.add_speaker(
            name,
            [audio_data],
            [filename],
            apply_enhancement=apply_enhancement,
            apply_super_resolution=apply_super_resolution_flag,
            enhancement_model_name=enhancement_model,
        )
        
        if result["status"] == "success":
            payload = {"success": True, "role": name}
            if result.get("clearvoice"):
                payload["clearvoice"] = result["clearvoice"]
            return JSONResponse(content=payload)
        else:
            return _success_error(result["message"])
            
    except Exception as e:
        error_msg = f"Failed to add speaker '{name}': {str(e)}"
        print(f"❌ API: {error_msg}")
        return _success_error(error_msg)

@app.post("/delete_speaker")
async def delete_speaker(
    name: str = Form(..., description="The name of the speaker")
):
    """API: Delete a speaker"""
    try:
        print(f"🗑️ API: Deleting speaker '{name}'")
        
        if not speaker_api:
            return _success_error("Speaker manager not initialized")
        
        result = await speaker_api.delete_speaker(name)
        
        if result["status"] == "success":
            return JSONResponse(content={"success": True, "role": name})
        else:
            return _success_error(result["message"])
            
    except Exception as e:
        error_msg = f"Failed to delete speaker '{name}': {str(e)}"
        print(f"❌ API: {error_msg}")
        return _success_error(error_msg)

@app.get("/audio_roles")
async def audio_roles():
    """API: List available speakers"""
    try:
        print("📋 API: Listing audio roles")
        
        if not speaker_api:
            return JSONResponse(content={"success": False, "roles": [], "speakers": {}})
        
        speakers_data = await speaker_api.list_speakers()
        
        if speakers_data["status"] == "success":
            speakers = speakers_data.get("speakers", {})
            roles = list(speakers.keys())
            return JSONResponse(
                content={
                    "success": True,
                    "roles": roles,
                    "speakers": speakers,
                    "total_speakers": speakers_data.get("total_speakers", len(roles)),
                }
            )
        else:
            return JSONResponse(content={"success": False, "roles": [], "speakers": {}})
            
    except Exception as e:
        error_msg = f"Failed to list audio roles: {str(e)}"
        print(f"❌ API: {error_msg}")
        return _success_error(error_msg)


@app.get("/api/speaker_preview/{speaker_name}")
async def api_speaker_preview(speaker_name: str):
    """API: Return stored MP3 preview for a speaker."""
    preview_path = _get_speaker_preview_path(speaker_name)
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Speaker preview not found")
    return FileResponse(
        preview_path,
        media_type="audio/mpeg",
        filename=f"{speaker_name}_preview.mp3",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ============================================================================
# Qwen3-TTS Voice Design API Endpoints
# ============================================================================

def get_voice_design_manager() -> Qwen3VoiceDesignManager:
    """Get or initialize Voice Design manager with preset manager integration."""
    global _voice_design_manager
    if _voice_design_manager is None:
        config = Qwen3TTSConfig(
            voice_design_model_path=os.environ.get(
                "QWEN3_VOICE_DESIGN_MODEL",
                "./checkpoints/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            ),
            device=os.environ.get("QWEN3_TTS_DEVICE", "cuda:0"),
        )
        # Get the speaker manager from TTSManager instance
        tts_instance = TTSManager.get_instance()
        preset_manager = tts_instance.speaker_manager if tts_instance and tts_instance.is_ready() else None
        _voice_design_manager = Qwen3VoiceDesignManager(
            config=config,
            preset_manager=preset_manager
        )
    
    # Check if preset_manager was None during init but is now available
    # This handles the case where Voice Design was used before IndexTTS was fully initialized
    if _voice_design_manager.preset_manager is None:
        tts_instance = TTSManager.get_instance()
        if tts_instance and tts_instance.is_ready() and tts_instance.speaker_manager:
            _voice_design_manager.preset_manager = tts_instance.speaker_manager
            print("[Qwen3-TTS] Updated Voice Design manager with speaker preset manager")
    
    return _voice_design_manager


class VoiceDesignRequest(BaseModel):
    """Request to generate speech with voice design."""
    text: str = Field(..., description="Text to synthesize")
    voice_description: str = Field(..., description="Natural language description of desired voice")
    language: str = Field(default="Auto", description="Target language")
    output_format: Literal["wav", "mp3"] = Field(default="mp3")


class SaveDesignedVoiceRequest(BaseModel):
    """Request to save last generated voice design as a speaker preset."""
    preset_name: str = Field(..., description="Name for the speaker preset")
    description: Optional[str] = Field(None, description="Optional description override")


@app.post("/api/design-voice")
async def design_voice(request: VoiceDesignRequest):
    """
    Generate speech with Qwen3-TTS Voice Design.
    
    Describe the desired voice in natural language and generate speech.
    The result is cached and can be saved to speaker presets.
    """
    try:
        manager = get_voice_design_manager()
        
        # Generate in thread pool to avoid blocking
        result: DesignedVoiceResult = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: manager.generate_voice_design(
                text=request.text,
                voice_description=request.voice_description,
                language=request.language,
            )
        )
        
        # Use existing convert_audio_to_format function for MP3 conversion
        audio_bytes, media_type, _ = await convert_audio_to_format(
            result.audio_waveform,
            result.sample_rate,
            output_format=request.output_format,
            bitrate="128k"
        )
        
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={"X-Voice-Design-Cached": "true"}
        )
    except Exception as e:
        print(f"❌ Voice Design error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/design-voice/save-preset")
async def save_designed_voice_to_preset(request: SaveDesignedVoiceRequest):
    """
    Save the last generated voice design directly to Speaker Presets.
    
    This allows users to save a voice they like without downloading
    and re-uploading the audio file.
    """
    try:
        manager = get_voice_design_manager()
        
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: manager.save_to_preset(
                preset_name=request.preset_name,
                description=request.description,
            )
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Save preset error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/design-voice/languages")
async def get_design_voice_languages():
    """Get list of supported languages for Voice Design."""
    return {"languages": Qwen3VoiceDesignManager.SUPPORTED_LANGUAGES}


@app.get("/api/design-voice/status")
async def get_design_voice_status():
    """Check Voice Design model status."""
    try:
        manager = get_voice_design_manager()
        return {
            "enabled": True,
            "model_loaded": manager.is_model_loaded(),
            "has_cached_result": manager.get_last_generated() is not None,
            "preset_save_available": manager.preset_manager is not None,
        }
    except Exception as e:
        return {
            "enabled": False,
            "error": str(e),
            "preset_save_available": False,
        }


@app.get("/api/segment_preview/{session_id}/{segment_index}")
async def api_segment_preview(
    session_id: str,
    segment_index: int,
    variant: str = Query("auto"),
    _ts: Optional[str] = Query(None),
):
    """API: Return MP3 preview for a translation segment (supports original vs translated variants)."""
    safe_session_id = "".join(c for c in session_id if c.isalnum() or c in {"-", "_"})
    variant_value = (variant or "auto").lower()
    force_preview = variant_value == "preview"
    force_original = variant_value == "original"

    def _candidate_paths(suffix: str) -> List[str]:
        candidates = [os.path.join(TRANSLATE_SESSION_MEDIA_DIR, f"{safe_session_id}_{suffix}")]
        if safe_session_id != session_id:
            candidates.append(os.path.join(TRANSLATE_SESSION_MEDIA_DIR, f"{session_id}_{suffix}"))
        return candidates

    def _resolve_existing_path(candidates: List[str]) -> Optional[str]:
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    preview_candidates = _candidate_paths(f"segment_{segment_index}_preview.mp3")
    original_candidates = _candidate_paths(f"segment_{segment_index}.mp3")

    file_path: Optional[str] = None
    preview_path = _resolve_existing_path(preview_candidates)
    if preview_path and not force_original:
        file_path = preview_path

    if file_path is None:
        if force_preview:
            raise HTTPException(status_code=404, detail="Preview audio not generated yet")
        original_path = _resolve_existing_path(original_candidates)
        if original_path:
            file_path = original_path
        else:
            # Need to generate the original snippet on-demand
            session = await _get_translate_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found or expired")

            segment = None
            for seg in session.base_segments:
                if seg.get("index") == segment_index:
                    segment = seg
                    break

            if not segment:
                raise HTTPException(status_code=404, detail=f"Segment {segment_index} not found in session")

            try:
                start_ms = int(segment.get("start_ms", 0))
                end_ms = int(segment.get("end_ms", start_ms))
                
                print(f"[segment_preview] Generating preview for segment {segment_index} ({start_ms}ms - {end_ms}ms)")
                
                # Try FFmpeg extraction from stored audio file (fast - no full file loading!)
                source_path = getattr(session, "original_audio_path", None)
                preview_result = None
                slice_timeout_seconds = 20
                
                print(f"[segment_preview] Session original_audio_path: {source_path}")
                print(f"[segment_preview] Path exists: {os.path.exists(source_path) if source_path else 'N/A'}")
                
                if source_path and os.path.exists(source_path):
                    print(f"[segment_preview] Using FFmpeg to extract segment {segment_index} from {os.path.basename(source_path)} ({start_ms}ms - {end_ms}ms)")
                    try:
                        preview_result = await asyncio.wait_for(
                            _run_blocking(
                                _save_segment_preview_from_path,
                                session_id,
                                segment_index,
                                source_path,
                                start_ms,
                                end_ms,
                            ),
                            timeout=slice_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        print(f"[segment_preview] FFmpeg extraction timed out for segment {segment_index}")
                        preview_result = None
                    print(f"[segment_preview] FFmpeg extraction result: {preview_result}")
                else:
                    print(f"[segment_preview] Source path not available or doesn't exist")
                
                # No fallback to pydub to avoid loading full audio; fail fast if ffmpeg did not produce a result
                if not preview_result:
                    raise HTTPException(status_code=500, detail="Failed to generate segment preview (ffmpeg)")

                if preview_result:
                    file_path = _resolve_existing_path(original_candidates)
                    if not file_path:
                        file_path = original_candidates[0]
                        if not os.path.exists(file_path):
                            raise HTTPException(status_code=500, detail="Generated preview file is missing")
                else:
                    raise HTTPException(status_code=500, detail="Failed to generate segment preview")
            except Exception as exc:
                print(f"⚠️ Failed to generate segment preview on-demand: {exc}")
                raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(exc)}")
    
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=f"segment_{segment_index}.mp3",
        headers={"Cache-Control": "public, max-age=3600"},
    )


async def _validate_speaker_preset_request(speaker_name: Optional[str]) -> Optional[JSONResponse]:
    if not speaker_name:
        return _success_error("Speaker name is required", status_code=400)
    if speaker_api and not speaker_api.speaker_exists(speaker_name):
        speakers_data = await speaker_api.list_speakers()
        available_roles = list(speakers_data.get("speakers", {}).keys())
        error_msg = f"'{speaker_name}' is not in the list of existing roles: {', '.join(available_roles)}"
        return _success_error(error_msg)
    return None


@app.post("/speak")
async def speak(req: SpeakRequest):
    """API: Generate speech using registered speaker"""
    try:
        print(f"🎭 API: Speaking with '{req.name}' - '{req.text[:50]}...'")
        
        speaker_error = await _validate_speaker_preset_request(req.name)
        if speaker_error is not None:
            return speaker_error
        
        tts = tts_manager.get_tts()
        
        # Generate speech
        output_path = os.path.join("outputs", f"speak_{uuid.uuid4().hex}.wav")
        
        # Check if emotion text is provided and not empty
        use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
        
        result = await tts.infer(
            spk_audio_prompt="",
            text=req.text,
            output_path=output_path,
            speaker_preset=req.name,
            use_emo_text=use_emotion_text,
            emo_text=req.emotion_text if use_emotion_text else None,
            emo_alpha=req.emotion_weight,
            speech_length=req.speech_length,
            diffusion_steps=req.diffusion_steps,
            max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
            verbose=SETTINGS.verbose
        )
        
        response = await _generated_audio_attachment_response(result, req.response_format)
        print(f"✅ API: Generated {len(response.body or b'')} bytes of {req.response_format.upper()} audio")
        return response
        
    except Exception as e:
        import traceback
        error_msg = f"Voice synthesis failed: {str(e)}"
        print(f"❌ API: {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return _success_error(error_msg)

def parse_clone_form(
    text: str = Form(...),
    reference_audio: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(None),
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(None),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    repetition_penalty: float = Form(1.0),
    max_tokens: int = Form(4096),
    length_threshold: int = Form(50),
    window_size: int = Form(50),
    stream: bool = Form(False),
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Form("mp3"),
    emotion_text: Optional[str] = Form(""),
    emotion_weight: float = Form(0.6),
    diffusion_steps: int = Form(10),
    max_text_tokens_per_sentence: int = Form(120),
):
    return CloneRequest(
        text=text, reference_audio=reference_audio, reference_text=reference_text,
        pitch=pitch, speed=speed, temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, max_tokens=max_tokens,
        length_threshold=length_threshold, window_size=window_size,
        stream=stream, response_format=response_format,
        emotion_text=emotion_text, emotion_weight=emotion_weight,
        diffusion_steps=diffusion_steps, max_text_tokens_per_sentence=max_text_tokens_per_sentence
    )

@app.post("/clone_voice")
async def clone_voice(
    req: CloneRequest = Depends(parse_clone_form),
    reference_audio_file: Optional[UploadFile] = File(None),
):
    """API: Clone voice using reference audio"""
    try:
        print(f"🎵 API: Cloning voice - '{req.text[:50]}...'")
        
        tmp_path, reference_error = await _persist_reference_audio_to_tempfile(
            reference_audio_file,
            req.reference_audio,
        )
        if reference_error is not None:
            return reference_error
        assert tmp_path is not None
        
        try:
            tts = tts_manager.get_tts()
            
            # Generate speech using reference audio
            output_path = os.path.join("outputs", f"clone_{uuid.uuid4().hex}.wav")
            
            # Check if emotion text is provided and not empty
            use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
            
            result = await tts.infer(
                spk_audio_prompt=tmp_path,
                text=req.text,
                output_path=output_path,
                use_emo_text=use_emotion_text,
                emo_text=req.emotion_text if use_emotion_text else None,
                emo_alpha=req.emotion_weight,
                speech_length=req.speech_length,
                diffusion_steps=req.diffusion_steps,
                max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
                verbose=SETTINGS.verbose
            )
            
            response = await _generated_audio_attachment_response(result, req.response_format)
            print(f"✅ API: Cloned voice - {len(response.body or b'')} bytes of {req.response_format.upper()}")
            return response
            
        finally:
            # Cleanup
            await async_remove_file(tmp_path)
                
    except Exception as e:
        error_msg = f"Failed to clone voice: {str(e)}"
        print(f"❌ API: {error_msg}")
        return _success_error(error_msg)

@app.get("/server_info")
async def server_info():
    """API: Get server information"""
    try:
        if not speaker_api:
            roles = []
        else:
            speakers_data = await speaker_api.list_speakers()
            roles = list(speakers_data["speakers"].keys()) if speakers_data["status"] == "success" else []
        
        return JSONResponse(content={
            "success": True,
            "info": {
                "model": "IndexTTS-vLLM-v2",
                "roles": roles,
                "sample_rate": 22050,
                "engine": "vLLM v2",
                "chinese_support": True,
                "speaker_presets": True,
                "speaker_manager": "SpeakerPresetManager",
                "whisperx_available": is_whisperx_available(),
                "qwen_omnivad_available": is_qwen_omnivad_available(),
                "parakeet_available": is_parakeet_available(),
                "translation_llm_default": DEFAULT_TRANSLATION_LLM_MODEL,
                "translation_llm_models": list(ALLOWED_TRANSLATION_LLM_MODELS),
                "stable_audio_available": stable_audio3_manager.status().get("available", False),
                "stable_audio_default": STABLE_AUDIO3_DEFAULT_VARIANT,
                "stable_audio_checkpoints": STABLE_AUDIO3_CHECKPOINT_DIR,
            }
        })
        
    except Exception as e:
        return _success_error(f"Failed to get server info: {str(e)}")

@app.post("/speak_stream")
async def speak_stream(req: SpeakRequest):
    """API: Generate speech using registered speaker with streaming"""
    try:
        print(f"🎭 API Streaming: Speaking with '{req.name}' - '{req.text[:50]}...'")
        
        speaker_error = await _validate_speaker_preset_request(req.name)
        if speaker_error is not None:
            return speaker_error
        
        tts = tts_manager.get_tts()
        
        # Check if emotion text is provided and not empty
        use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
        
        async def audio_stream_generator():
            """Generator that yields audio chunks as they are produced"""
            try:
                chunk_count = 0
                async for chunk_idx, wav_cpu, is_last in tts.infer_stream(
                    spk_audio_prompt="",
                    text=req.text,
                    speaker_preset=req.name,
                    use_emo_text=use_emotion_text,
                    emo_text=req.emotion_text if use_emotion_text else None,
                    emo_alpha=req.emotion_weight,
                    speech_length=req.speech_length,
                    diffusion_steps=req.diffusion_steps,
                    max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
                    first_chunk_max_tokens=40,  # Default first chunk size
                    verbose=SETTINGS.verbose
                ):
                    chunk_count += 1
                    print(f"🎵 Streaming chunk {chunk_idx} (is_last={is_last})")
                    audio_bytes = _encode_streaming_audio_chunk(wav_cpu, req.response_format)
                    yield _streaming_audio_frame(chunk_idx, audio_bytes, is_last)
                
                print(f"✅ Streaming complete: {chunk_count} chunks sent")
                
            except Exception as e:
                error_msg = f"ERROR:{str(e)}\n".encode('utf-8')
                print(f"❌ Streaming error: {e}")
                traceback.print_exc()
                yield error_msg
        
        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers=STREAMING_RESPONSE_HEADERS,
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Voice synthesis streaming failed: {str(e)}"
        print(f"❌ API Streaming: {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return _success_error(error_msg)

@app.post("/clone_voice_stream")
async def clone_voice_stream(
    req: CloneRequest = Depends(parse_clone_form),
    reference_audio_file: Optional[UploadFile] = File(None),
):
    """API: Clone voice using reference audio with streaming"""
    tmp_path: Optional[str] = None
    try:
        print(f"🎵 API Streaming: Cloning voice - '{req.text[:50]}...'")
        
        tmp_path, reference_error = await _persist_reference_audio_to_tempfile(
            reference_audio_file,
            req.reference_audio,
        )
        if reference_error is not None:
            return reference_error
        assert tmp_path is not None
        
        tts = tts_manager.get_tts()
        
        # Check if emotion text is provided and not empty
        use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
        
        async def audio_stream_generator():
            """Generator that yields audio chunks as they are produced"""
            try:
                chunk_count = 0
                async for chunk_idx, wav_cpu, is_last in tts.infer_stream(
                    spk_audio_prompt=tmp_path,
                    text=req.text,
                    use_emo_text=use_emotion_text,
                    emo_text=req.emotion_text if use_emotion_text else None,
                    emo_alpha=req.emotion_weight,
                    speech_length=req.speech_length,
                    diffusion_steps=req.diffusion_steps,
                    max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
                    first_chunk_max_tokens=40,  # Default first chunk size
                    verbose=SETTINGS.verbose
                ):
                    chunk_count += 1
                    print(f"🎵 Streaming chunk {chunk_idx} (is_last={is_last})")
                    audio_bytes = _encode_streaming_audio_chunk(wav_cpu, req.response_format)
                    yield _streaming_audio_frame(chunk_idx, audio_bytes, is_last)
                
                print(f"✅ Streaming complete: {chunk_count} chunks sent")
                
            except Exception as e:
                error_msg = f"ERROR:{str(e)}\n".encode('utf-8')
                print(f"❌ Streaming error: {e}")
                traceback.print_exc()
                yield error_msg
            finally:
                # Cleanup temporary file after streaming is complete
                await async_remove_file(tmp_path)
        
        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers=STREAMING_RESPONSE_HEADERS,
        )
                
    except Exception as e:
        if tmp_path:
            await async_remove_file(tmp_path)
        error_msg = f"Failed to clone voice with streaming: {str(e)}"
        print(f"❌ API Streaming: {error_msg}")
        return _success_error(error_msg)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting IndexTTS vLLM v2 FastAPI WebUI...")
    print(f"📁 Model directory: {SETTINGS.model_dir}")
    print(f"🔧 GPU memory utilization: {SETTINGS.gpu_memory_utilization}")
    print(f"🎯 FP16 mode: {SETTINGS.is_fp16}")
    print(f"🌐 Server will start on {SETTINGS.host}:{SETTINGS.port}")
    print(f"🎯 Concurrent capacity: 100 requests")
    print(f"⚡ Single worker process for optimal GPU utilization")
    print(f"💡 Features:")
    print(f"   - IndexTTS vLLM v2 backend for ultra-fast inference")
    print(f"   - Speaker preset management with persistent storage")
    print(f"   - API compatibility for external integrations")
    print(f"   - Modern web interface with Chinese support")
    print(f"   - MP3 output for smaller file sizes")
    print(f"   - High concurrency support (100 concurrent connections)")
    print(f"   - Advanced translate/edit mode with segment editing")
    print(f"   - Gemini model selection (Flash vs Pro) with API key override")
    print(f"   - Per-segment generation control for efficient processing")
    
    uvicorn.run(
        app,
        host=SETTINGS.host,
        port=SETTINGS.port,
        log_level="info",
        workers=1,
        limit_concurrency=100,
        limit_max_requests=None,  # No limit on total requests
        backlog=2048,  # Handle request queue efficiently
        timeout_keep_alive=300,  # Set timeout to 300 seconds
        h11_max_incomplete_event_size=16777216,  # 16MB for large audio uploads
        access_log=True  # Enable access logging for debugging
    )
