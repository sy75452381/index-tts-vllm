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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy
from dataclasses import dataclass, field


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
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]
except ImportError:
    genai = None
    types = None


# FastAPI and web interface
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, validator
from urllib.parse import quote

# IndexTTS v2 and speaker management
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "indextts"))

from indextts.infer_vllm_v2 import IndexTTS2
from speaker_preset_manager import SpeakerPresetManager, initialize_preset_manager

# Configuration
import argparse


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

# Global thread executor for blocking operations
# Use CPU count for parallel audio processing, but cap at 8 to avoid excessive context switching
_executor_workers = min(8, max(4, (os.cpu_count() or 4)))
executor = ThreadPoolExecutor(max_workers=_executor_workers, thread_name_prefix="fastapi_async")

# Global ClearVoice models (initialized lazily and reused)
_enhancement_model: Optional[Any] = None
_super_res_model: Optional[Any] = None


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
DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-pro"
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
    "\"start\" (timestamp mm:ss or mm:ss.xxx), "
    "\"end\" (same format), "
    "\"speaker\" (one of the ids from the speakers list), "
    "\"source_text\" (original-language transcript), "
    "\"translated_text\" (translation in {dest_language}). "
    "Ensure timestamps align with the audio, keep segments coherent, and add a new speaker entry whenever a new voice appears. "
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
    "\"start\" (timestamp mm:ss or mm:ss.xxx), "
    "\"end\" (same), "
    "\"speaker\" (speaker id), "
    "\"source_text\" (transcript in the original language), "
    "\"translated_text\" (use empty string \"\" because no translation is requested). "
    "Make sure timestamps are accurate and segments remain speaker-homogeneous. "
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
TRANSLATION_TTS_CONCURRENCY = 20
MIN_SPEECH_DURATION_MS = 3000
MAX_MERGE_INTERVAL_MS = 0
DEFAULT_GENERATED_VOLUME_PERCENT = 100.0
MIN_GENERATED_VOLUME_PERCENT = 10.0
MAX_GENERATED_VOLUME_PERCENT = 300.0
DEFAULT_SILENCE_VOLUME_PERCENT = DEFAULT_GENERATED_VOLUME_PERCENT
DEFAULT_EMOTION_WEIGHT = 0.6
ALLOWED_GEMINI_MODELS = {"gemini-flash-latest", "gemini-2.5-pro"}
GEMINI_AUDIO_EXPORT_BITRATE = "128k"
GEMINI_CACHE_VERSION = 1
SPEAKER_PREVIEW_DIR = Path("speaker_presets") / "previews"
SPEAKER_PREVIEW_BITRATE = "128k"
CHUNK_SPLIT_DEFAULT_MIN_MINUTES = 10.0
CHUNK_SPLIT_DEFAULT_MAX_MINUTES = 15.0
CHUNK_SPLIT_MIN_MINUTES = 1.0
CHUNK_SPLIT_MAX_MINUTES = 45.0
CHUNK_SPLIT_MIN_SILENCE_MS = 1500
CHUNK_SPLIT_SILENCE_THRESHOLD_DB = -42.0
CHUNK_SPLIT_MAX_CHUNKS = 64
CHUNK_SPLIT_SILENCE_GRACE_MS = 45000
CHUNK_SPLIT_MIN_CHUNK_MS = 1000
CHUNK_BATCH_GENERATE_DELAY_SECONDS = 60
TRANSLATE_USE_FFMPEG_SPLIT_MERGE = _env_flag("TRANSLATE_USE_FFMPEG_SPLIT_MERGE", True)
CHUNK_SPLIT_FFMPEG_CONCURRENCY = _env_int("CHUNK_SPLIT_FFMPEG_CONCURRENCY", 2, min_value=1, max_value=8)

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
class ClearVoiceParallelChunkJob:
    chunk_idx: int
    chunk_path: str
    apply_enhancement: bool
    apply_super_resolution: bool


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


def _coerce_merge_backing_flag(
    requested: bool,
    apply_enhancement: bool,
    alternate_backing_available: bool = False,
) -> bool:
    if not requested:
        return False
    if apply_enhancement or alternate_backing_available:
        return True
    print("⚠️ Merge-back requested without MossFormer2_SE_48K enhancement or custom/reused backing; ignoring request.")
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
    with BytesIO() as buffer:
        export_kwargs: Dict[str, Any] = {"format": fmt}
        if bitrate and fmt in {"mp3", "ogg", "opus", "aac"}:
            export_kwargs["bitrate"] = bitrate
        audio.export(buffer, **export_kwargs)
        return buffer.getvalue()


async def _export_audio_segment_bytes(audio: AudioSegment, fmt: str = "mp3", bitrate: Optional[str] = None) -> bytes:
    return await _run_blocking(_export_audio_segment_bytes_sync, audio, fmt, bitrate)


def _export_audio_segment_to_path_sync(
    audio: AudioSegment,
    path: str,
    fmt: str = "mp3",
    bitrate: Optional[str] = None,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    audio.export(path, format=fmt, bitrate=bitrate)
    return path


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


async def _run_clearvoice_pipeline(
    original_audio: AudioSegment,
    *,
    apply_enhancement: bool,
    apply_super_resolution: bool,
    pre_clearvoice_mix_audio: Optional[AudioSegment],
    emit_status: Optional[Callable[..., Awaitable[None]]],
    source_audio_path: Optional[str] = None,
    clearvoice_parallel_config: Optional[ClearVoiceParallelConfig] = None,
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
    try:
        if emit_status:
            action = "Applying MossFormer2_SE_48K enhancement..."
            if apply_super_resolution and not apply_enhancement:
                action = "Applying MossFormer2_SR_48K super-resolution..."
            elif apply_super_resolution and apply_enhancement:
                action = "Applying MossFormer2_SE_48K enhancement + SR_48K super-resolution..."
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
        cache_dir = os.path.join(CLEARVOICE_CACHE_DIR, cache_hash)
        os.makedirs(cache_dir, exist_ok=True)
        cached_enhanced_path = os.path.join(cache_dir, "mossformer2_se.mp3")
        cached_sr_path = os.path.join(cache_dir, "mossformer2_sr.mp3")
        cached_backing_path = os.path.join(cache_dir, "backing_track.mp3")
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
            print(f"♻️ ClearVoice: Reusing cached MossFormer2_SE_48K output for {cache_hash}.")

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
                    _apply_clearvoice_parallel_sync,
                    temp_input_path,
                    total_duration_ms,
                    apply_enhancement,
                    apply_super_resolution,
                    parallel_config,
                )
            else:
                final_processed_local_path, clearvoice_paths, enhancement_output_local = await apply_clearvoice_processing(
                    temp_input_path,
                    apply_enhancement,
                    apply_super_resolution,
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
                print(f"💾 ClearVoice: Cached MossFormer2_SE_48K output for {cache_hash}.")

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
                backing_track_audio = AudioSegment.from_file(cached_backing_path, format="mp3")
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
                            message=f"Extracted instrumental backing track via MossFormer2_SE_48K{sr_note}.",
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
) -> Tuple[AudioSegment, str, bytes, str, Optional[AudioSegment], bool, str]:
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
        return (
            original_audio,
            input_mime_type,
            processed_audio_bytes,
            "audio/mpeg",
            backing_track_audio,
            merge_with_backing,
            backing_track_source,
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
        normalized_audio_temp_path: Optional[str] = None
        if effective_apply_enhancement:
            upload_temp_path = _persist_audio_upload(audio_bytes, original_filename)
            normalized_audio_temp_path = await _normalize_uploaded_audio(upload_temp_path)
            source_audio_temp_path = normalized_audio_temp_path

        input_mime_type = audio_mime_type_value or (audio_file.content_type if audio_file else None) or "audio/wav"
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

        pre_clearvoice_mix_audio = original_audio if effective_apply_enhancement else None
        processed_audio, backing_track_audio = await _run_clearvoice_pipeline(
            original_audio,
            apply_enhancement=effective_apply_enhancement,
            apply_super_resolution=apply_super_resolution,
            pre_clearvoice_mix_audio=pre_clearvoice_mix_audio,
            emit_status=emit_status,
            source_audio_path=source_audio_temp_path,
            clearvoice_parallel_config=clearvoice_parallel_config,
        )
        if backing_track_audio is not None:
            backing_track_source = "extracted"
        if custom_backing_audio is not None:
            backing_track_audio = custom_backing_audio
            backing_track_source = "custom"

        processed_audio_bytes = await _export_audio_segment_bytes(
            processed_audio,
            fmt="mp3",
            bitrate=GEMINI_AUDIO_EXPORT_BITRATE,
        )
        merge_with_backing = requested_merge_backing and backing_track_audio is not None
        if requested_merge_backing and backing_track_audio is None:
            print("⚠️ Unable to merge with backing track because no instrumental was derived.")

        if emit_status:
            if apply_super_resolution:
                await emit_status(
                    stage="gemini_prep",
                    message="📤 Sending super-resolved audio (MossFormer2_SR_48K) to Gemini for transcription/translation.",
                )
            elif apply_enhancement:
                await emit_status(
                    stage="gemini_prep",
                    message="📤 Sending enhanced audio (MossFormer2_SE_48K) to Gemini for transcription/translation.",
                )
            else:
                await emit_status(
                    stage="gemini_prep",
                    message="📤 Sending original audio to Gemini for transcription/translation.",
                )

        return (
            processed_audio,
            input_mime_type,
            processed_audio_bytes,
            "audio/mpeg",
            backing_track_audio,
            merge_with_backing,
            backing_track_source,
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


def _clearvoice_cache_paths(cache_hash: str) -> Tuple[str, str, str]:
    cache_dir = os.path.join(CLEARVOICE_CACHE_DIR, cache_hash)
    cached_enhanced_path = os.path.join(cache_dir, "mossformer2_se.mp3")
    cached_sr_path = os.path.join(cache_dir, "mossformer2_sr.mp3")
    cached_backing_path = os.path.join(cache_dir, "backing_track.mp3")
    return cache_dir, cached_enhanced_path, cached_sr_path, cached_backing_path


async def _prepare_clearvoice_for_split(
    *,
    audio_file: Optional[UploadFile],
    audio_reference: Optional[str],
    preloaded_audio_bytes: Optional[bytes],
    source_audio_filename: Optional[str],
    apply_super_resolution: bool,
    emit_status: Optional[Callable[..., Awaitable[None]]] = None,
    clearvoice_parallel_config: Optional[ClearVoiceParallelConfig] = None,
) -> SplitClearVoiceAssets:
    """
    Prepare ClearVoice-enhanced audio for chunk splitting without decoding the entire track when cached.
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
        cache_dir, cached_enhanced_path, cached_sr_path, cached_backing_path = _clearvoice_cache_paths(hash_value)
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
        manifest = _load_clearvoice_input_manifest(source_hash)
        if manifest:
            manifest_hash = manifest.get("clearvoice_hash")
            processed_hash = manifest.get("processed_hash")
            if manifest_hash:
                _update_cache_paths(manifest_hash)
                if os.path.exists(cached_enhanced_path) or os.path.exists(cached_sr_path):
                    print(
                        f"♻️ ClearVoice: Reusing cached assets for upload hash {source_hash[:8]} → {cache_hash}."
                    )
                else:
                    cache_hash = None
                    manifest = None
        if cache_hash is None:
            wav_input_path = await _ensure_wav_input_path(audio_bytes)
            _update_cache_paths(_compute_file_md5(wav_input_path))
        final_processed_target = cached_sr_path if apply_super_resolution else cached_enhanced_path

        need_processing = not os.path.exists(final_processed_target)
        if apply_super_resolution and not os.path.exists(cached_enhanced_path):
            need_processing = True
        backing_needed = not os.path.exists(cached_backing_path)

        if need_processing or backing_needed:
            if emit_status:
                await emit_status(
                    stage="enhancement",
                    message="Generating ClearVoice cache for splitting...",
                )
            source_path = normalized_audio_temp_path
            if source_path is None:
                await _ensure_wav_input_path(audio_bytes)
                source_path = normalized_audio_temp_path or wav_input_path
            if source_path is None:
                raise TranslateWorkflowHttpError(
                    500,
                    {"status": "error", "message": "Unable to prepare source audio for ClearVoice."},
                )
            original_audio = await _load_audio_segment_from_path(source_path)
            processed_audio, backing_track_audio = await _run_clearvoice_pipeline(
                original_audio,
                apply_enhancement=True,
                apply_super_resolution=apply_super_resolution,
                pre_clearvoice_mix_audio=original_audio,
                emit_status=emit_status,
                source_audio_path=source_path,
                clearvoice_parallel_config=clearvoice_parallel_config,
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
        if os.path.exists(cached_backing_path):
            try:
                backing_info = _probe_audio_metadata_from_path(cached_backing_path, compute_hash=False)
                backing_source = "cache"
            except Exception as exc:
                print(f"⚠️ Failed to probe cached backing track: {exc}")

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
            backing_track_path=cached_backing_path if os.path.exists(cached_backing_path) else None,
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
    force_gemini_regenerate: bool = False,
    initial_speaker_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    default_speaker_preset: Optional[str] = None,
    default_emotion_weight: Optional[float] = None,
    clearvoice_settings: Optional[Dict[str, Any]] = None,
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

    if emit_status:
        await emit_status(
            stage="gemini",
            message=f"Analyzing audio with Gemini model '{resolved_gemini_model}'...",
        )

    speaker_profiles: List[Dict[str, Any]] = []
    gemini_cache_info: Dict[str, Any] = {}
    if manual_chunk_data is not None:
        gemini_chunks = manual_chunk_data
        speaker_profiles = manual_speaker_profiles or []
    else:
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
            message=f"Received {len(gemini_chunks)} raw segments; preparing timeline...",
        )

    original_duration_ms = len(original_audio)
    segments = _prepare_translation_segments(
        original_duration_ms,
        gemini_chunks,
        dest_language,
        speaker_profiles=speaker_profiles,
        min_speech_duration_ms=min_speech_duration,
        max_merge_interval_ms=max_merge_interval,
    )

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
        default_speaker_preset=normalized_default_speaker,
        default_emotion_weight=normalized_default_emotion,
    )
    if source_chunk_session and source_chunk_session.chunk_parent_id:
        session.chunk_parent_id = source_chunk_session.chunk_parent_id
        session.chunk_index = source_chunk_session.chunk_index
        session.chunk_start_ms = source_chunk_session.chunk_start_ms
        session.chunk_end_ms = source_chunk_session.chunk_end_ms
        session.chunk_cut_reason = source_chunk_session.chunk_cut_reason
        session.chunk_silence_midpoint_ms = source_chunk_session.chunk_silence_midpoint_ms
        session.chunk_source_session_id = source_chunk_session.session_id
    ui_segments = _serialize_segments_for_ui(segments, session.session_id)

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
    source_audio_filename: Optional[str] = None
    source_base_name: Optional[str] = None
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


def _persist_session_audio_segment(
    session_id: str,
    audio: AudioSegment,
    kind: str,
    fmt: str = "mp3",
) -> Tuple[str, AudioAssetInfo]:
    path = _session_media_path(session_id, kind, fmt)
    audio.export(path, format=fmt)
    return path, _audio_segment_metadata(audio)


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

# Create directories
APP_DIR = Path(__file__).resolve().parent
_OUTPUT_SUBDIRS = (
    "translate_results",
    "clearvoice_cache",
    "translate_session_media",
    "split_audio_cache",
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
CLEARVOICE_CACHE_DIR = str((ROOT_OUTPUT_DIR / "clearvoice_cache").resolve())
os.makedirs(CLEARVOICE_CACHE_DIR, exist_ok=True)
GEMINI_CACHE_DIR = str((ROOT_OUTPUT_DIR / "gemini_cache").resolve())
os.makedirs(GEMINI_CACHE_DIR, exist_ok=True)
SPLIT_AUDIO_CACHE_DIR = str((ROOT_OUTPUT_DIR / "split_audio_cache").resolve())
os.makedirs(SPLIT_AUDIO_CACHE_DIR, exist_ok=True)


def _guess_media_type_from_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    mapping = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "opus": "audio/opus",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
        "srt": "application/x-subrip",
    }
    return mapping.get(ext, "application/octet-stream")

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
) -> List[int]:
    """
    Detect silence midpoints within a short audio window and return absolute millisecond offsets.
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
    return [
        offset_ms + int(((start + end) / 2) * 1000 / sample_rate)
        for start, end in silence_intervals
        if end > start
    ]


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
    Returns (sorted_midpoints, stats).
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

    midpoints: List[int] = []
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
        midpoints.extend(
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
                midpoints.extend(future.result())
            except Exception as exc:
                print(f"⚠️ Silence detection window failed: {exc}")

    stats["elapsed_ms"] = (time.perf_counter() - detection_start) * 1000
    return sorted(set(midpoints)), stats


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
    grace_window = min(CHUNK_SPLIT_SILENCE_GRACE_MS, max_chunk_ms // 2 or 15_000)

    range_selection_start = time.perf_counter()
    while current_start < total_duration_ms and len(chunk_ranges) < CHUNK_SPLIT_MAX_CHUNKS:
        desired_min = current_start + min_chunk_ms
        desired_max = min(current_start + max_chunk_ms, total_duration_ms)
        candidate_cut: Optional[int] = None
        candidate_after_window: Optional[int] = None
        cut_from_silence = False

        while pointer < len(silence_midpoints_ms) and silence_midpoints_ms[pointer] <= current_start:
            pointer += 1

        lookahead = pointer
        while lookahead < len(silence_midpoints_ms):
            point = silence_midpoints_ms[lookahead]
            if point < desired_min:
                lookahead += 1
                continue
            if point <= desired_max:
                candidate_cut = point
                cut_from_silence = True
                pointer = lookahead + 1
                break
            candidate_after_window = point
            pointer = lookahead + 1
            break

        if candidate_cut is None and candidate_after_window is not None:
            if candidate_after_window - desired_max <= grace_window:
                candidate_cut = candidate_after_window
                cut_from_silence = True

        if candidate_cut is None:
            candidate_cut = desired_max
            cut_from_silence = False

        if candidate_cut <= current_start:
            candidate_cut = min(total_duration_ms, max(current_start + min_chunk_ms, current_start + 1000))
            cut_from_silence = False

        chunk_ranges.append(
            {
                "start_ms": current_start,
                "end_ms": candidate_cut,
                "cut_reason": "silence_center" if cut_from_silence else "hard_limit",
                "silence_midpoint_ms": candidate_cut if cut_from_silence else None,
            }
        )
        current_start = candidate_cut

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


INVALID_FILENAME_CHARS = set('<>:"/\\|?*')
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
    sanitized_chars: List[str] = []
    for ch in base_name:
        if ch in INVALID_FILENAME_CHARS or ord(ch) < 32:
            sanitized_chars.append("_")
        else:
            sanitized_chars.append(ch)
    sanitized = "".join(sanitized_chars).strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip(" _")
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
        return reuse_session.source_base_name
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
        "source_audio_filename": session.source_audio_filename,
        "source_base_name": session.source_base_name,
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
        manifest.get("response_format", TRANSLATE_DEFAULT_OUTPUT_FORMAT),
        manifest.get("bitrate", TRANSLATE_DEFAULT_BITRATE),
        manifest.get("input_mime_type"),
        manifest.get("clearvoice_settings") or {},
        manifest.get("base_segments") or [],
        manifest.get("gemini_chunks") or [],
        manifest.get("gemini_model") or _get_gemini_model_name(),
        manifest.get("gemini_api_key"),
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
    ignore_non_speech: bool,
    preserve_silence_audio: bool,
    generated_volume_percent: float,
    backing_volume_percent: float,
    merge_backing_track: bool,
    silence_volume_percent: float,
    force_gemini_regenerate: bool = False,
    default_speaker_preset: Optional[str] = None,
    default_emotion_weight: Optional[float] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    clearvoice_settings = chunk_session.clearvoice_settings or {}
    apply_enhancement = bool(clearvoice_settings.get("enhancement"))
    apply_super_resolution = bool(clearvoice_settings.get("super_resolution"))
    if apply_super_resolution and not apply_enhancement:
        apply_enhancement = True
        clearvoice_settings["enhancement"] = True
    parallel_config = _parallel_config_from_settings(clearvoice_settings)
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
        emit_status=None,
        source_chunk_session=chunk_session,
        source_audio_filename=chunk_session.source_audio_filename,
        source_base_name=chunk_session.source_base_name,
        force_gemini_regenerate=force_gemini_regenerate,
        initial_speaker_overrides=chunk_session.speaker_overrides,
        default_speaker_preset=default_speaker_preset,
        default_emotion_weight=default_emotion_weight,
        clearvoice_settings=clearvoice_settings,
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
    )

    metadata = dict(segment_result.metadata)
    metadata.update(synthesis_metadata or {})
    metadata["ignore_non_speech"] = ignore_non_speech
    metadata["preserve_silence_audio"] = preserve_silence_audio
    metadata["generated_volume_percent"] = generated_volume_percent
    metadata["backing_volume_percent"] = backing_volume_percent
    metadata["silence_volume_percent"] = silence_volume_percent
    metadata["gemini_model"] = gemini_model
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
) -> Tuple[str, List[str], Optional[str]]:
    if not config or not config.enabled:
        return _apply_clearvoice_processing_sync(input_path, apply_enhancement, apply_super_resolution)
    if not _ffmpeg_available():
        print("⚠️ ClearVoice parallel requested but ffmpeg is unavailable. Falling back to sequential run.")
        return _apply_clearvoice_processing_sync(input_path, apply_enhancement, apply_super_resolution)

    chunk_ranges = _plan_clearvoice_parallel_chunks(total_duration_ms, config.chunk_ms)
    if len(chunk_ranges) <= 1:
        return _apply_clearvoice_processing_sync(input_path, apply_enhancement, apply_super_resolution)

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
) -> Tuple[str, List[str], Optional[str]]:
    """Run ClearVoice enhancement/super-resolution synchronously."""
    global _enhancement_model, _super_res_model
    
    if ClearVoice is None:
        raise RuntimeError("ClearVoice package is not available in the environment.")
    
    generated_paths: List[str] = []
    enhancement_output_path: Optional[str] = None
    current_input = input_path
    final_path = input_path
    
    try:
        if apply_enhancement:
            print("✨ ClearVoice: Applying MossFormer2_SE_48K enhancement...")
            # Initialize enhancement model if not already created
            if _enhancement_model is None:
                print("🔧 Initializing enhancement model (first use)...")
                _enhancement_model = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
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
) -> Tuple[str, List[str], Optional[str]]:
    """Async wrapper for ClearVoice processing."""
    if not (apply_enhancement or apply_super_resolution):
        return input_path, []
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        _apply_clearvoice_processing_sync,
        input_path,
        apply_enhancement,
        apply_super_resolution,
    )

async def convert_audio_to_format(wav_data, sample_rate, output_format="mp3", bitrate="128k"):
    """Convert audio data to specified format (MP3 or WAV)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _convert_audio_to_format_sync, wav_data, sample_rate, output_format, bitrate)

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
    Accepts either a JSON array or an object with "segments" and optional "speakers".
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


def _segment_audio_data_uri(
    audio: AudioSegment,
    start_ms: int,
    end_ms: int,
    fmt: str = "mp3",
    bitrate: str = "128k",
) -> Optional[str]:
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
    buffer = BytesIO()
    export_kwargs: Dict[str, Any] = {}
    if fmt == "mp3":
        export_kwargs["bitrate"] = bitrate
        mime_type = "audio/mpeg"
    else:
        mime_type = f"audio/{fmt}"
    snippet.export(buffer, format=fmt, **export_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
    media_format: str = "wav",
    source_audio_filename: Optional[str] = None,
    source_base_name: Optional[str] = None,
    default_speaker_preset: Optional[str] = None,
    default_emotion_weight: Optional[float] = None,
    original_audio_path: Optional[str] = None,
    original_audio_info: Optional[AudioAssetInfo] = None,
    session_id: Optional[str] = None,
) -> TranslateSessionData:
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

    audio_path = original_audio_path
    audio_info = original_audio_info
    if not audio_path and original_audio is None:
        raise ValueError("Either original_audio or original_audio_path must be provided.")
    should_persist_audio = persist_media or not audio_path or (audio_path and not os.path.exists(audio_path))
    if should_persist_audio:
        if original_audio is None:
            raise ValueError("Original audio data is required to persist session media.")
        target_format = media_format or "wav"
        audio_path, audio_info = _persist_session_audio_segment(session_id, original_audio, "vocals", fmt=target_format)
    if audio_info is None and audio_path:
        try:
            audio_info = _probe_audio_metadata_from_path(audio_path, compute_hash=False)
        except Exception as exc:
            print(f"⚠️ Failed to probe audio metadata for {audio_path}: {exc}")
            audio_info = None

    backing_path = backing_track_path
    backing_info = backing_audio_info
    if backing_path is None and backing_track_audio is not None:
        backing_path, backing_info = _persist_session_audio_segment(session_id, backing_track_audio, "backing", fmt="mp3")
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
        source_audio_filename=source_audio_filename,
        source_base_name=resolved_base_name,
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
                session.response_format = response_format
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
    return codec_args


def _run_ffmpeg_command(cmd: List[str]) -> None:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        raise RuntimeError(stderr or stdout or "ffmpeg command failed")


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
) -> Tuple[bytes, str, Dict[str, Any]]:
    tts = tts_manager.get_tts()
    frame_rate = int(original_audio.frame_rate or 22050)
    sample_width = int(original_audio.sample_width or 2)
    channels = int(original_audio.channels or 1)

    combined_audio = _create_silence_segment(0, frame_rate, sample_width, channels)
    generation_log: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(TRANSLATION_TTS_CONCURRENCY)
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
                    verbose=cmd_args.verbose,
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

    segment_tasks = [process_segment(idx, segment) for idx, segment in enumerate(segments)]
    results = await asyncio.gather(*segment_tasks)
    results.sort(key=lambda item: item[0])

    for _, audio_segment, log_entry in results:
        combined_audio += audio_segment
        generation_log.append(log_entry)

    original_duration_ms = len(original_audio)
    final_duration_ms = len(combined_audio)
    if pad_to_original and final_duration_ms < original_duration_ms:
        combined_audio += _create_silence_segment(original_duration_ms - final_duration_ms, frame_rate, sample_width, channels)
        final_duration_ms = len(combined_audio)

    backing_applied = False
    if merge_with_backing and backing_track_audio is not None:
        try:
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
                len(combined_audio),
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
            combined_audio = prepared_backing.overlay(combined_audio)
            backing_applied = True
        except Exception as merge_error:
            print(f"⚠️ Failed to merge translated audio with backing track: {merge_error}")

    buffer = BytesIO()
    export_kwargs: Dict[str, Any] = {}
    audio_format = (response_format or TRANSLATE_DEFAULT_OUTPUT_FORMAT).lower()
    if audio_format == "mp3" and bitrate:
        export_kwargs["bitrate"] = bitrate
    combined_audio.export(buffer, format=audio_format, **export_kwargs)
    audio_bytes = buffer.getvalue()

    media_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "opus": "audio/opus",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
    }
    media_type = media_type_map.get(audio_format, f"audio/{audio_format}")

    metadata = {
        "dest_language": dest_language,
        "segment_count": len(segments),
        "speech_segment_count": sum(1 for s in segments if s["type"] == "speech"),
        "silence_segment_count": sum(1 for s in segments if s["type"] == "silence"),
        "original_duration_ms": original_duration_ms,
        "generated_duration_ms": len(combined_audio),
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
                if not os.path.exists(cmd_args.model_dir):
                    raise FileNotFoundError(f"Model directory {cmd_args.model_dir} does not exist")
                
                # Check required files
                required_files = [
                    "bpe.model",
                    "gpt.pth", 
                    "config.yaml",
                    "s2mel.pth",
                    "wav2vec2bert_stats.pt"
                ]
                
                for file in required_files:
                    file_path = os.path.join(cmd_args.model_dir, file)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Required file {file_path} does not exist")
                
                # Initialize IndexTTS2
                self.tts = IndexTTS2(
                    model_dir=cmd_args.model_dir,
                    is_fp16=cmd_args.is_fp16,
                    use_torch_compile=cmd_args.use_torch_compile,
                    gpu_memory_utilization=cmd_args.gpu_memory_utilization
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
                        cv_features.append("MossFormer2_SE_48K enhancement")
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
        description="Apply ClearVoice MossFormer2_SE_48K enhancement before translation.",
    )
    super_resolution_voice: Optional[bool] = Field(
        default=False,
        description="Apply ClearVoice MossFormer2_SR_48K super-resolution before translation.",
    )
    clearvoice_parallel_enabled: Optional[bool] = Field(
        default=False,
        description="Enable experimental ClearVoice parallel chunk processing for faster separation.",
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

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting IndexTTS vLLM v2 FastAPI WebUI...")
    await tts_manager.initialize()
    
    # Only run warmup inference when torch_compile is enabled
    # torch_compile benefits from warmup to compile optimized CUDA graphs
    if cmd_args.use_torch_compile:
        print("🔥 Running warmup (--use_torch_compile enabled)...")
        await warmup_model()
    else:
        print("⏭️ Skipping warmup (--use_torch_compile not enabled)")
    
    yield
    # Shutdown (if needed)
    print("🔄 Shutting down IndexTTS vLLM v2...")
    # Shutdown the thread executor
    executor.shutdown(wait=True)

# Create FastAPI app
app = FastAPI(
    title="IndexTTS vLLM v2 FastAPI WebUI",
    description="Ultra-fast TTS with vLLM backend, speaker presets, and advanced translate/edit mode with Gemini integration",
    lifespan=lifespan
)

# Web Interface - HTML file paths for hot reload support
_HTML_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
_HTML_NEW_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_new.html")

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open(_HTML_NEW_FILE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        # Inject dynamic configuration values
        html_content = html_content.replace("{{CHUNK_SPLIT_MIN_SILENCE_MS}}", str(CHUNK_SPLIT_MIN_SILENCE_MS))
        return html_content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1><p>Please ensure index.html exists in the same directory as fastapi_webui_v2.py</p>",
            status_code=500
        )

@app.get("/old", response_class=HTMLResponse)
async def home_old():
    try:
        with open(_HTML_FILE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        # Inject dynamic configuration values
        html_content = html_content.replace("{{CHUNK_SPLIT_MIN_SILENCE_MS}}", str(CHUNK_SPLIT_MIN_SILENCE_MS))
        return html_content
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index_new.html not found</h1><p>Please ensure index_new.html exists in the same directory as fastapi_webui_v2.py</p>",
            status_code=500
        )

# Health check endpoint
# Health check removed - use /server_info endpoint instead

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


@app.post("/api/translate_split_audio")
async def api_translate_split_audio(
    request: Request,
    dest_language: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    base_filename: Optional[str] = Form(None),
    chunk_min_minutes: Optional[float] = Form(None),
    chunk_max_minutes: Optional[float] = Form(None),
    min_silence_ms: Optional[int] = Form(None),
    silence_threshold_db: Optional[float] = Form(None),
    super_resolution_voice: Optional[bool] = Form(False),
    clearvoice_parallel_enabled: Optional[bool] = Form(False),
    clearvoice_parallel_chunk_seconds: Optional[int] = Form(None),
    clearvoice_parallel_max_workers: Optional[int] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """API: Split long audio into ClearVoice-enhanced chunks for reuse."""
    if ClearVoice is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "ClearVoice package is required for audio chunking. Install the `clearvoice` package to enable this feature.",
            },
        )

    try:
        payload: Optional[Dict[str, Any]] = None
        clearvoice_parallel_enabled_value = clearvoice_parallel_enabled
        clearvoice_parallel_chunk_seconds_value = clearvoice_parallel_chunk_seconds
        clearvoice_parallel_max_workers_value = clearvoice_parallel_max_workers
        content_type = request.headers.get("content-type", "")
        if (
            audio_file is None
            and (audio is None or not audio.strip())
            and "application/json" in content_type.lower()
        ):
            try:
                payload = await request.json()
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid JSON payload: {str(exc)}"},
                )

        if payload is not None:
            dest_language = payload.get("dest_language", dest_language)
            audio = payload.get("audio", audio)
            audio_mime_type = payload.get("audio_mime_type", audio_mime_type)
            base_filename = payload.get("base_filename", base_filename)
            chunk_min_minutes = payload.get("chunk_min_minutes", chunk_min_minutes)
            chunk_max_minutes = payload.get("chunk_max_minutes", chunk_max_minutes)
            min_silence_ms = payload.get("min_silence_ms", min_silence_ms)
            silence_threshold_db = payload.get("silence_threshold_db", silence_threshold_db)
            super_resolution_voice = payload.get("super_resolution_voice", super_resolution_voice)
            clearvoice_parallel_enabled_value = payload.get(
                "clearvoice_parallel_enabled", clearvoice_parallel_enabled_value
            )
            clearvoice_parallel_chunk_seconds_value = payload.get(
                "clearvoice_parallel_chunk_seconds", clearvoice_parallel_chunk_seconds_value
            )
            clearvoice_parallel_max_workers_value = payload.get(
                "clearvoice_parallel_max_workers", clearvoice_parallel_max_workers_value
            )

        audio_reference_value = (audio or "").strip()
        if audio_file is None and not audio_reference_value:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Source audio is required for chunk splitting.",
                },
            )

        dest_language_value = (dest_language or "").strip() or "unspecified"
        apply_super_resolution = _coerce_to_bool(super_resolution_voice)
        parallel_config = _coerce_clearvoice_parallel_config(
            clearvoice_parallel_enabled_value,
            clearvoice_parallel_chunk_seconds_value,
            clearvoice_parallel_max_workers_value,
        )
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

        if audio_file is not None:
            try:
                audio_io = await load_audio_bytes_from_request(audio_file, None)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"status": "error", "message": exc.detail},
                )
            preloaded_audio_bytes = audio_io.read()
            if not preloaded_audio_bytes:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Uploaded audio file is empty."},
                )
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
            "super_resolution": apply_super_resolution,
            "clearvoice_parallel": parallel_config.to_metadata(),
            "base_output_name": resolved_base_name,
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
                heartbeat = asyncio.create_task(heartbeat_task())
                try:
                    await emit_status(stage="start", message="Split request accepted.", summary=split_request_summary)

                    audio_prepare_start = time.perf_counter()
                    clearvoice_assets = await _prepare_clearvoice_for_split(
                        audio_file=audio_file_for_pipeline,
                        audio_reference=audio_reference_value if audio_reference_value else None,
                        preloaded_audio_bytes=preloaded_audio_bytes,
                        source_audio_filename=uploaded_filename,
                        apply_super_resolution=apply_super_resolution,
                        emit_status=emit_status,
                        clearvoice_parallel_config=parallel_config if parallel_config.enabled else None,
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
                        cached_base_name = cached_split_data.get("base_output_name")
                        if cached_base_name:
                            resolved_base_name = cached_base_name
                            print(f"♻️ Using cached base name for artifact discovery: {resolved_base_name}")
                        elif cached_manifests and cached_manifests[0].get("source_base_name"):
                            resolved_base_name = cached_manifests[0]["source_base_name"]
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
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(split_stream(), media_type="application/json")

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
        session_ids: List[str] = [
            sid for sid in (payload.chunk_session_ids or []) if isinstance(sid, str) and sid.strip()
        ]
        chunk_batch_id = (payload.chunk_batch_id or "").strip()

        sessions: List[TranslateSessionData] = []
        resolve_start = time.perf_counter()
        if session_ids:
            for sid in session_ids:
                session = await _get_translate_session(sid)
                if session is None:
                    return JSONResponse(
                        status_code=404,
                        content={"status": "error", "message": f"Chunk session '{sid}' not found or expired."},
                    )
                if session.chunk_parent_id is None:
                    return JSONResponse(
                        status_code=400,
                        content={"status": "error", "message": f"Session '{sid}' is not a chunk session."},
                    )
                sessions.append(session)
        elif chunk_batch_id:
            sessions = await _list_chunk_sessions(chunk_batch_id)
            if not sessions:
                return JSONResponse(
                    status_code=404,
                    content={"status": "error", "message": "No chunk sessions found for this batch."},
                )
            sessions.sort(key=lambda s: (s.chunk_index or 0, s.chunk_start_ms or 0))
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Provide chunk_session_ids (ordered list) or chunk_batch_id to merge.",
                },
            )
        merge_profiler.mark("resolve_chunk_sessions", resolve_start, extra=f"sessions={len(sessions)}")
        if not chunk_batch_id and sessions:
            chunk_batch_id = sessions[0].chunk_parent_id or ""
        merge_profiler.rename(f"merge:{chunk_batch_id or 'adhoc'}")

        merge_backing_request = _coerce_to_bool(payload.merge_backing_track)
        response_format_value = TRANSLATE_DEFAULT_OUTPUT_FORMAT
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
                        try:
                            overlay_temp = await _run_blocking(
                                _ffmpeg_overlay_to_tempfile,
                                final_audio_path,
                                backing_path,
                                output_format=response_format_value,
                                bitrate=ffmpeg_bitrate,
                                vocals_volume=1.0,
                                backing_volume=1.0,
                            )
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
                    if final_audio_segment is None:
                        if final_audio_path:
                            final_audio_segment = await _load_audio_segment_from_path(final_audio_path)
                        else:
                            final_audio_segment = merged_audio_segment
                    if final_audio_segment is not None:
                        overlay_python_start = time.perf_counter()
                        backing_ext = os.path.splitext(backing_path)[1].lstrip(".") or None
                        full_backing_audio = AudioSegment.from_file(backing_path, format=backing_ext)
                        backing_len = len(full_backing_audio)
                        vocal_len = len(final_audio_segment)
                        if backing_len >= vocal_len:
                            final_audio_segment = full_backing_audio.overlay(final_audio_segment)
                        else:
                            print(
                                f"⚠️ Backing track ({backing_len} ms) shorter than merged vocals ({vocal_len} ms); using vocal timeline as base."
                            )
                            final_audio_segment = final_audio_segment.overlay(full_backing_audio)
                        final_audio_path = None
                        merge_profiler.mark(
                            "python_overlay",
                            overlay_python_start,
                            extra=f"backing_len={backing_len}ms,vocals_len={vocal_len}ms",
                        )
                else:
                    print(f"⚠️ Full backing track not found at {backing_path}; exporting vocals only.")
        if final_audio_segment is not None:
            export_start = time.perf_counter()
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
        with open(output_path, "wb") as outfile:
            outfile.write(merged_bytes)
        for temp_path in temp_paths_to_cleanup:
            await async_remove_file(temp_path)

        audio_url = f"/api/translate_outputs/{output_filename}"
        subtitle_export_start = time.perf_counter()
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

        merge_profiler.summary()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": f"Merged {len(chunk_results)} chunk(s).",
                "audio_url": audio_url,
                "file_name": output_filename,
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
            },
        )
    except Exception as exc:
        traceback.print_exc()
        if "merge_profiler" in locals():
            merge_profiler.summary()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to merge chunks: {str(exc)}"},
        )


@app.post("/api/translate_generate_chunks")
async def api_translate_generate_chunks(payload: ChunkBatchGenerateRequest):
    """API: Generate translated audio for multiple chunk sessions in parallel."""
    try:
        session_ids = [sid.strip() for sid in payload.chunk_session_ids if isinstance(sid, str) and sid.strip()]
        if not session_ids:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Select at least one chunk session to generate."},
            )

        sessions: List[TranslateSessionData] = []
        for sid in session_ids:
            session = await _get_translate_session(sid)
            if session is None:
                return JSONResponse(
                    status_code=404,
                    content={"status": "error", "message": f"Chunk session '{sid}' not found or expired."},
                )
            if session.chunk_parent_id is None:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Session '{sid}' is not a chunk session."},
                )
            sessions.append(session)

        config_template = sessions[0]
        dest_language_value = (payload.dest_language or config_template.dest_language or "").strip()
        if not dest_language_value:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Destination language (dest_language) is required."},
            )

        response_format_value = (payload.response_format or config_template.response_format or TRANSLATE_DEFAULT_OUTPUT_FORMAT).lower()
        bitrate_value = payload.bitrate or config_template.bitrate or TRANSLATE_DEFAULT_BITRATE
        gemini_model_value = _normalize_gemini_model_name(payload.gemini_model or config_template.gemini_model)
        gemini_api_key_value = (payload.gemini_api_key or config_template.gemini_api_key or "").strip()
        ignore_non_speech_flag = _coerce_to_bool(
            payload.ignore_non_speech if payload.ignore_non_speech is not None else config_template.ignore_non_speech
        )
        preserve_silence_flag = _coerce_to_bool(
            payload.preserve_silence_audio
            if payload.preserve_silence_audio is not None
            else config_template.preserve_silence_audio
        )
        generated_volume_percent_value = _coerce_volume_percent(
            payload.generated_volume_percent,
            config_template.generated_volume_percent,
        )
        backing_volume_percent_value = _coerce_volume_percent(
            payload.backing_volume_percent,
            config_template.backing_volume_percent,
        )
        silence_volume_percent_value = _coerce_volume_percent(
            payload.silence_volume_percent
            if payload.silence_volume_percent is not None
            else config_template.silence_volume_percent,
            config_template.silence_volume_percent,
        )
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
                            ignore_non_speech=ignore_non_speech_flag,
                            preserve_silence_audio=preserve_silence_flag,
                            generated_volume_percent=generated_volume_percent_value,
                            backing_volume_percent=backing_volume_percent_value,
                            merge_backing_track=merge_backing_requested,
                            silence_volume_percent=silence_volume_percent_value,
                            force_gemini_regenerate=force_gemini_regenerate_flag,
                            default_speaker_preset=session.default_speaker_preset,
                            default_emotion_weight=session.default_emotion_weight,
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
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(chunk_generate_stream(), media_type="application/json")
    except TranslateWorkflowHttpError as http_error:
        return JSONResponse(status_code=http_error.status_code, content=http_error.content)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to generate chunk audios: {str(exc)}"},
        )


@app.post("/api/translate_segments")
async def api_translate_segments(
    request: Request,
    dest_language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    bitrate: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    base_filename: Optional[str] = Form(None),
    custom_backing_audio: Optional[str] = Form(None),
    custom_backing_audio_mime_type: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    translate_text: Optional[bool] = Form(True),
    gemini_model: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    enhance_voice: Optional[bool] = Form(False),
    super_resolution_voice: Optional[bool] = Form(False),
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
):
    """API: Prepare translation segments for advanced translate/edit workflow."""
    reuse_session_id_value: Optional[str] = reuse_session_id
    try:
        payload: Optional[Dict[str, Any]] = None
        dest_language_value = dest_language
        response_format_value = response_format
        bitrate_value = bitrate
        audio_reference = audio
        audio_mime_type_value = audio_mime_type
        base_filename_value = base_filename
        custom_backing_audio_value = custom_backing_audio
        custom_backing_audio_mime_type_value = custom_backing_audio_mime_type
        prompt_override = prompt
        translate_flag_value = translate_text
        gemini_model_value = gemini_model
        gemini_api_key_value = gemini_api_key
        enhance_voice_value = enhance_voice
        super_resolution_voice_value = super_resolution_voice
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

        content_type = request.headers.get("content-type", "")
        if (
            dest_language is None
            and not audio_reference
            and audio_file is None
            and "application/json" in content_type.lower()
        ):
            try:
                payload = await request.json()
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid JSON payload: {str(exc)}"},
                )

        if payload is not None:
            dest_language_value = payload.get("dest_language", dest_language_value)
            response_format_value = payload.get("response_format", response_format_value)
            bitrate_value = payload.get("bitrate", bitrate_value)
            audio_reference = payload.get("audio", audio_reference)
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
            enhance_voice_value = payload.get("enhance_voice", enhance_voice_value)
            super_resolution_voice_value = payload.get("super_resolution_voice", super_resolution_voice_value)
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

        dest_language_value = (dest_language_value or "").strip()
        if not dest_language_value:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Destination language (dest_language) is required."},
            )

        response_format_value = TRANSLATE_DEFAULT_OUTPUT_FORMAT
        bitrate_value = bitrate_value or TRANSLATE_DEFAULT_BITRATE
        translate_enabled = _coerce_to_bool(translate_flag_value if translate_flag_value is not None else True)
        ignore_non_speech_flag = _coerce_to_bool(ignore_non_speech_value)
        preserve_silence_audio_flag = _coerce_to_bool(preserve_silence_audio_value)
        apply_enhancement = _coerce_to_bool(enhance_voice_value)
        apply_super_resolution = _coerce_to_bool(super_resolution_voice_value)
        parallel_config = _coerce_clearvoice_parallel_config(
            clearvoice_parallel_enabled_value,
            clearvoice_parallel_chunk_seconds_value,
            clearvoice_parallel_max_workers_value,
        )
        if not (apply_enhancement or apply_super_resolution):
            parallel_config.enabled = False
        if apply_super_resolution and not apply_enhancement:
            apply_enhancement = True
        if apply_super_resolution and not apply_enhancement:
            apply_enhancement = True
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
        generated_volume_percent_value = _coerce_volume_percent(
            generated_volume_percent_value,
            DEFAULT_GENERATED_VOLUME_PERCENT,
        )
        backing_volume_percent_value = _coerce_volume_percent(
            backing_volume_percent_value,
            DEFAULT_GENERATED_VOLUME_PERCENT,
        )
        silence_volume_percent_value = _coerce_volume_percent(
            silence_volume_percent_value,
            DEFAULT_SILENCE_VOLUME_PERCENT,
        )
        force_gemini_regenerate_flag = _coerce_to_bool(force_gemini_regen_value or False)
        default_speaker_value = (default_speaker_value or "").strip()
        reuse_source_session: Optional[TranslateSessionData] = None
        reuse_session_id_value = (reuse_session_id_value or "").strip()
        if reuse_session_id_value:
            reuse_source_session = await _get_translate_session(reuse_session_id_value)
            if reuse_source_session is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "error",
                        "message": "Reuse session not found or expired. Please re-upload the audio.",
                    },
                )
        reuse_backing_available = bool(reuse_source_session and _session_has_backing_audio(reuse_source_session))
        requested_merge_backing = _coerce_merge_backing_flag(
            merge_backing_requested_raw,
            apply_enhancement,
            custom_backing_present or reuse_backing_available,
        )
        if reuse_source_session is not None and reuse_source_session.chunk_parent_id:
            requested_merge_backing = False
        clearvoice_settings: Dict[str, Any] = {
            "enhancement": apply_enhancement,
            "super_resolution": apply_super_resolution,
        }
        if parallel_config.enabled:
            clearvoice_settings.update(parallel_config.to_metadata())
        resolved_gemini_model = _normalize_gemini_model_name(gemini_model_value)
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
            "force_gemini_regenerate": force_gemini_regenerate_flag,
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
                    ) = await _prepare_audio_assets(
                        reuse_source_session=reuse_session_for_segments,
                        audio_file=audio_file,
                        audio_reference=audio_reference,
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
                        emit_status=emit_status,
                        source_chunk_session=reuse_session_for_segments,
                        source_audio_filename=session_source_filename,
                        source_base_name=resolved_base_name,
                        force_gemini_regenerate=force_gemini_regenerate_flag,
                        initial_speaker_overrides=getattr(reuse_session_for_segments, "speaker_overrides", None),
                        default_speaker_preset=default_speaker_value,
                        default_emotion_weight=default_emotion_weight_value,
                        clearvoice_settings=clearvoice_settings,
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
                yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(translate_stream(), media_type="application/json")
    except RuntimeError as runtime_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(runtime_error)},
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to prepare translation segments: {str(exc)}"},
        )


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

        response_format_value = TRANSLATE_DEFAULT_OUTPUT_FORMAT

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

                    await emit_status(
                        stage="synthesis",
                        message=f"Synthesizing translated speech ({len(final_segments)} selected segments)...",
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
                yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(generate_stream(), media_type="application/json")
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
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Translate session not found or expired."},
            )

        seg_input = payload.segment
        if seg_input.type != "speech":
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Only speech segments can be previewed."},
            )

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


@app.post("/api/translate_audio")
async def api_translate_audio(
    request: Request,
    dest_language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    bitrate: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    base_filename: Optional[str] = Form(None),
    custom_backing_audio: Optional[str] = Form(None),
    custom_backing_audio_mime_type: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    enhance_voice: Optional[bool] = Form(False),
    super_resolution_voice: Optional[bool] = Form(False),
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
):
    """API: Translate speech audio to a target language and return synthesized audio."""
    reuse_session_id_value: Optional[str] = reuse_session_id
    try:
        payload: Optional[Dict[str, Any]] = None
        dest_language_value = dest_language
        audio_reference = audio
        audio_mime_type_value = audio_mime_type
        base_filename_value = base_filename
        custom_backing_audio_value = custom_backing_audio
        custom_backing_audio_mime_type_value = custom_backing_audio_mime_type
        prompt_override = prompt
        response_format_value = response_format
        bitrate_value = bitrate
        gemini_model_value = gemini_model
        gemini_api_key_value = gemini_api_key
        enhance_voice_value = enhance_voice
        super_resolution_voice_value = super_resolution_voice
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

        content_type = request.headers.get("content-type", "")
        if (
            dest_language is None
            and not audio_reference
            and audio_file is None
            and "application/json" in content_type.lower()
        ):
            try:
                payload = await request.json()
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid JSON payload: {str(exc)}"},
                )

        if payload is not None:
            try:
                translate_req = TranslateRequest(**payload)
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid translate request: {str(exc)}"},
                )
            dest_language_value = translate_req.dest_language
            audio_reference = translate_req.audio or audio_reference
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
            enhance_voice_value = translate_req.enhance_voice if translate_req.enhance_voice is not None else enhance_voice_value
            super_resolution_voice_value = (
                translate_req.super_resolution_voice if translate_req.super_resolution_voice is not None else super_resolution_voice_value
            )
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

        default_speaker_value = (default_speaker_value or "").strip()
        if not default_speaker_value:
            default_speaker_value = None
        default_emotion_weight_value = _coerce_emotion_weight(
            default_emotion_weight_value,
            DEFAULT_EMOTION_WEIGHT,
        )

        reuse_source_session: Optional[TranslateSessionData] = None
        reuse_session_id_value = (reuse_session_id_value or "").strip()
        if reuse_session_id_value:
            reuse_source_session = await _get_translate_session(reuse_session_id_value)
            if reuse_source_session is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "error",
                        "message": "Reuse session not found or expired. Please re-upload the audio.",
                    },
                )

        dest_language_value = (dest_language_value or "").strip()
        if not dest_language_value:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Destination language (dest_language) is required."},
            )

        response_format_value = TRANSLATE_DEFAULT_OUTPUT_FORMAT
        bitrate_value = bitrate_value or TRANSLATE_DEFAULT_BITRATE
        ignore_non_speech_flag = _coerce_to_bool(ignore_non_speech_value)
        preserve_silence_audio_flag = _coerce_to_bool(preserve_silence_audio_value)
        apply_enhancement = _coerce_to_bool(enhance_voice_value)
        apply_super_resolution = _coerce_to_bool(super_resolution_voice_value)
        parallel_config = _coerce_clearvoice_parallel_config(
            clearvoice_parallel_enabled_value,
            clearvoice_parallel_chunk_seconds_value,
            clearvoice_parallel_max_workers_value,
        )
        if not (apply_enhancement or apply_super_resolution):
            parallel_config.enabled = False
        custom_backing_present = bool(custom_backing_audio_file) or bool((custom_backing_audio_value or "").strip())
        merge_backing_requested_raw = _coerce_to_bool(merge_backing_track_value)
        reuse_backing_available = bool(reuse_source_session and _session_has_backing_audio(reuse_source_session))
        requested_merge_backing = _coerce_merge_backing_flag(
            merge_backing_requested_raw,
            apply_enhancement,
            custom_backing_present or reuse_backing_available,
        )
        if reuse_source_session is not None and reuse_source_session.chunk_parent_id:
            requested_merge_backing = False
        clearvoice_settings: Dict[str, Any] = {
            "enhancement": apply_enhancement,
            "super_resolution": apply_super_resolution,
        }
        if parallel_config.enabled:
            clearvoice_settings.update(parallel_config.to_metadata())
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
        generated_volume_percent_value = _coerce_volume_percent(
            generated_volume_percent_value,
            DEFAULT_GENERATED_VOLUME_PERCENT,
        )
        backing_volume_percent_value = _coerce_volume_percent(
            backing_volume_percent_value,
            DEFAULT_GENERATED_VOLUME_PERCENT,
        )
        silence_volume_percent_value = _coerce_volume_percent(
            silence_volume_percent_value,
            DEFAULT_SILENCE_VOLUME_PERCENT,
        )
        resolved_gemini_model = _normalize_gemini_model_name(gemini_model_value)
        gemini_api_key_value = (gemini_api_key_value or "").strip()
        force_gemini_regenerate_flag = _coerce_to_bool(force_gemini_regen_value or False)


        uploaded_filename = audio_file.filename if audio_file else None
        audio_bytes: Optional[bytes] = None
        if reuse_source_session is None:
            audio_io = await load_audio_bytes_from_request(audio_file, audio_reference)
            if audio_io is None:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "No audio provided for translation."},
                )

            audio_bytes = audio_io.read()
            if not audio_bytes:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Provided audio data is empty."},
                )

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
            "force_gemini_regenerate": force_gemini_regenerate_flag,
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
                    await emit_status(stage="start", message="Translate request accepted.", summary=request_summary)
                    print(
                        "[translate_audio] pipeline start reuse_session=%s chunk_id=%s"
                        % (
                            reuse_session_for_translate.session_id if reuse_session_for_translate else None,
                            getattr(reuse_session_for_translate, "chunk_index", None),
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
                        emit_status=emit_status,
                        source_chunk_session=reuse_session_for_translate,
                         source_audio_filename=session_source_filename,
                         source_base_name=resolved_base_name,
                        force_gemini_regenerate=force_gemini_regenerate_flag,
                        initial_speaker_overrides=getattr(reuse_session_for_translate, "speaker_overrides", None),
                        default_speaker_preset=default_speaker_value,
                        default_emotion_weight=default_emotion_weight_value,
                        clearvoice_settings=clearvoice_settings,
                    )
                    session = segment_result.session
                    segments = segment_result.segments

                    await emit_status(
                        stage="synthesis",
                        message=f"Synthesizing translated speech ({len(segments)} total segments)...",
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
                yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(translate_stream(), media_type="application/json")

    except RuntimeError as runtime_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(runtime_error)},
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Translation failed: {str(exc)}"},
        )


# API Helper Functions (matching deploy_vllm_indextts.py exactly)
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

# API Compatibility Endpoints
@app.post("/add_speaker")
async def add_speaker(
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker"),
    audio: Optional[str] = Form(None, description="Reference audio URL or base64"),
    reference_text: Optional[str] = Form(None, description="Optional transcript"),
    audio_file: Optional[UploadFile] = File(None, description="Upload reference audio file"),
    enhance_voice: bool = Form(False, description="Apply ClearVoice MossFormer2_SE_48K enhancement"),
    super_resolution_voice: bool = Form(False, description="Apply ClearVoice MossFormer2_SR_48K super-resolution"),
):
    """API: Add a new speaker"""
    try:
        print(f"🎭 API: Adding speaker '{name}'")
        print(f"🔍 Debug: audio_file={audio_file}, audio={audio is not None}, reference_text={reference_text}")
        
        if not speaker_api:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Speaker manager not initialized"}
            )
        
        # Load audio from file or reference string (matching deploy_vllm_indextts.py)
        try:
            audio_io = await load_audio_bytes_from_request(audio_file, audio)
            if audio_io is None:
                print(f"❌ API: No audio provided for speaker '{name}'")
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "No audio provided"}
                )
        except Exception as audio_error:
            print(f"❌ API: Audio loading failed for speaker '{name}': {audio_error}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Audio loading failed: {str(audio_error)}"}
            )
        
        # Get audio data and filename
        audio_data = audio_io.read()
        filename = audio_file.filename if audio_file else f"{name}_reference.wav"
        
        apply_enhancement = bool(enhance_voice)
        apply_super_resolution_flag = bool(super_resolution_voice)
        print(f"🎚️ API: ClearVoice options -> enhancement={apply_enhancement}, super_resolution={apply_super_resolution_flag}")
        
        if (apply_enhancement or apply_super_resolution_flag) and ClearVoice is None:
            error_msg = "ClearVoice is required for enhancement or super-resolution. Install the `clearvoice` package to enable these options."
            print(f"❌ API: {error_msg}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": error_msg}
            )
        
        # Add speaker using SpeakerPresetManager (handles ClearVoice processing internally)
        result = await speaker_api.add_speaker(
            name,
            [audio_data],
            [filename],
            apply_enhancement=apply_enhancement,
            apply_super_resolution=apply_super_resolution_flag,
        )
        
        if result["status"] == "success":
            payload = {"success": True, "role": name}
            if result.get("clearvoice"):
                payload["clearvoice"] = result["clearvoice"]
            return JSONResponse(content=payload)
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result["message"]}
            )
            
    except Exception as e:
        error_msg = f"Failed to add speaker '{name}': {str(e)}"
        print(f"❌ API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.post("/delete_speaker")
async def delete_speaker(
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker")
):
    """API: Delete a speaker"""
    try:
        print(f"🗑️ API: Deleting speaker '{name}'")
        
        if not speaker_api:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Speaker manager not initialized"}
            )
        
        result = await speaker_api.delete_speaker(name)
        
        if result["status"] == "success":
            return JSONResponse(content={"success": True, "role": name})
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result["message"]}
            )
            
    except Exception as e:
        error_msg = f"Failed to delete speaker '{name}': {str(e)}"
        print(f"❌ API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

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
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )


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
                original_audio = _get_session_original_audio(session)
                start_ms = int(segment.get("start_ms", 0))
                end_ms = int(segment.get("end_ms", start_ms))

                preview_result = _save_segment_audio_preview(
                    session_id,
                    segment_index,
                    original_audio,
                    start_ms,
                    end_ms,
                )

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


@app.post("/speak")
async def speak(req: SpeakRequest):
    """API: Generate speech using registered speaker"""
    try:
        print(f"🎭 API: Speaking with '{req.name}' - '{req.text[:50]}...'")
        
        if not req.name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Speaker name is required"}
            )
        
        # Simple speaker validation to prevent failures
        if speaker_api and not speaker_api.speaker_exists(req.name):
            speakers_data = await speaker_api.list_speakers()
            available_roles = list(speakers_data.get("speakers", {}).keys())
            error_msg = f"'{req.name}' is not in the list of existing roles: {', '.join(available_roles)}"
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": error_msg}
            )
        
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
            verbose=cmd_args.verbose
        )
        
        # Convert to requested format and return as bytes (matching deploy_vllm_indextts.py)
        if req.response_format != "wav":
            # Read audio data once
            audio_data, sample_rate = await async_audio_read(result)
            # Convert to requested format and get bytes
            audio_bytes, media_type, _ = await convert_audio_to_format(
                audio_data, sample_rate, req.response_format, "128k"
            )
        else:
            # Read WAV file as bytes
            audio_bytes = await async_read_file(result)
            media_type = "audio/wav"
        
        # Set content type (matching deploy_vllm_indextts.py)
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus", 
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        content_type = content_type_map.get(req.response_format, f"audio/{req.response_format}")
        
        print(f"✅ API: Generated {len(audio_bytes)} bytes of {req.response_format.upper()} audio")
        
        # Cleanup temporary file
        await async_remove_file(result)
        
        # Return Response with bytes (matching deploy_vllm_indextts.py format exactly)
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "Cache-Control": "no-cache",
            }
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Voice synthesis failed: {str(e)}"
        print(f"❌ API: {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

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
        
        # Load reference audio (matching deploy_vllm_indextts.py)
        audio_io = await load_audio_bytes_from_request(reference_audio_file, req.reference_audio)
        if audio_io is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No reference audio provided"}
            )
        
        # Save reference audio to temporary file
        audio_data = audio_io.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        await async_write_file(tmp_path, audio_data)
        
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
                verbose=cmd_args.verbose
            )
            
            # Convert to requested format and return as bytes (matching deploy_vllm_indextts.py)
            if req.response_format != "wav":
                # Read audio data once
                audio_data, sample_rate = await async_audio_read(result)
                # Convert to requested format and get bytes
                audio_bytes, media_type, _ = await convert_audio_to_format(
                    audio_data, sample_rate, req.response_format, "128k"
                )
            else:
                # Read WAV file as bytes
                audio_bytes = await async_read_file(result)
                media_type = "audio/wav"
            
            # Set content type (matching deploy_vllm_indextts.py)
            content_type_map = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac", 
                "flac": "audio/flac",
                "wav": "audio/wav",
                "pcm": "audio/pcm",
            }
            content_type = content_type_map.get(req.response_format, f"audio/{req.response_format}")
            
            print(f"✅ API: Cloned voice - {len(audio_bytes)} bytes of {req.response_format.upper()}")
            
            # Cleanup temporary file
            await async_remove_file(result)
            
            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                    "Cache-Control": "no-cache",
                }
            )
            
        finally:
            # Cleanup
            await async_remove_file(tmp_path)
                
    except Exception as e:
        error_msg = f"Failed to clone voice: {str(e)}"
        print(f"❌ API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

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
                "speaker_manager": "SpeakerPresetManager"
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to get server info: {str(e)}"
            }
        )

@app.post("/speak_stream")
async def speak_stream(req: SpeakRequest):
    """API: Generate speech using registered speaker with streaming"""
    from fastapi.responses import StreamingResponse
    
    try:
        print(f"🎭 API Streaming: Speaking with '{req.name}' - '{req.text[:50]}...'")
        
        if not req.name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Speaker name is required"}
            )
        
        # Simple speaker validation to prevent failures
        if speaker_api and not speaker_api.speaker_exists(req.name):
            speakers_data = await speaker_api.list_speakers()
            available_roles = list(speakers_data.get("speakers", {}).keys())
            error_msg = f"'{req.name}' is not in the list of existing roles: {', '.join(available_roles)}"
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": error_msg}
            )
        
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
                    verbose=cmd_args.verbose
                ):
                    chunk_count += 1
                    print(f"🎵 Streaming chunk {chunk_idx} (is_last={is_last})")
                    
                    # Convert tensor to WAV bytes
                    wav_data = wav_cpu.numpy().astype(np.int16)
                    
                    # Create WAV file in memory
                    with BytesIO() as wav_buffer:
                        sf.write(wav_buffer, wav_data.T, 22050, format='WAV')
                        wav_bytes = wav_buffer.getvalue()
                    
                    # Convert to requested format
                    if req.response_format == "wav":
                        audio_bytes = wav_bytes
                    else:
                        audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
                        with BytesIO() as audio_buffer:
                            audio_segment.export(audio_buffer, format=req.response_format, bitrate="128k" if req.response_format == "mp3" else None)
                            audio_bytes = audio_buffer.getvalue()
                    
                    # Yield chunk with metadata header
                    header = f"CHUNK:{chunk_idx}:{len(audio_bytes)}:{'LAST' if is_last else 'MORE'}\n".encode('utf-8')
                    yield header + audio_bytes
                
                print(f"✅ Streaming complete: {chunk_count} chunks sent")
                
            except Exception as e:
                error_msg = f"ERROR:{str(e)}\n".encode('utf-8')
                print(f"❌ Streaming error: {e}")
                traceback.print_exc()
                yield error_msg
        
        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable proxy buffering
            }
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Voice synthesis streaming failed: {str(e)}"
        print(f"❌ API Streaming: {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.post("/clone_voice_stream")
async def clone_voice_stream(
    req: CloneRequest = Depends(parse_clone_form),
    reference_audio_file: Optional[UploadFile] = File(None),
):
    """API: Clone voice using reference audio with streaming"""
    from fastapi.responses import StreamingResponse
    
    try:
        print(f"🎵 API Streaming: Cloning voice - '{req.text[:50]}...'")
        
        # Load reference audio (matching deploy_vllm_indextts.py)
        audio_io = await load_audio_bytes_from_request(reference_audio_file, req.reference_audio)
        if audio_io is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No reference audio provided"}
            )
        
        # Save reference audio to temporary file
        audio_data = audio_io.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        await async_write_file(tmp_path, audio_data)
        
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
                    verbose=cmd_args.verbose
                ):
                    chunk_count += 1
                    print(f"🎵 Streaming chunk {chunk_idx} (is_last={is_last})")
                    
                    # Convert tensor to WAV bytes
                    wav_data = wav_cpu.numpy().astype(np.int16)
                    
                    # Create WAV file in memory
                    with BytesIO() as wav_buffer:
                        sf.write(wav_buffer, wav_data.T, 22050, format='WAV')
                        wav_bytes = wav_buffer.getvalue()
                    
                    # Convert to requested format
                    if req.response_format == "wav":
                        audio_bytes = wav_bytes
                    else:
                        audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
                        with BytesIO() as audio_buffer:
                            audio_segment.export(audio_buffer, format=req.response_format, bitrate="128k" if req.response_format == "mp3" else None)
                            audio_bytes = audio_buffer.getvalue()
                    
                    # Yield chunk with metadata header
                    header = f"CHUNK:{chunk_idx}:{len(audio_bytes)}:{'LAST' if is_last else 'MORE'}\n".encode('utf-8')
                    yield header + audio_bytes
                
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
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable proxy buffering
            }
        )
                
    except Exception as e:
        error_msg = f"Failed to clone voice with streaming: {str(e)}"
        print(f"❌ API Streaming: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting IndexTTS vLLM v2 FastAPI WebUI...")
    print(f"📁 Model directory: {cmd_args.model_dir}")
    print(f"🔧 GPU memory utilization: {cmd_args.gpu_memory_utilization}")
    print(f"🎯 FP16 mode: {cmd_args.is_fp16}")
    print(f"🌐 Server will start on {cmd_args.host}:{cmd_args.port}")
    print(f"🎯 Concurrent capacity: 100 requests (matching Modal deployment)")
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
        host=cmd_args.host,
        port=cmd_args.port,
        log_level="info",
        workers=1,  # Single worker to match Modal's single container
        limit_concurrency=100,  # Match Modal's max_inputs=100
        limit_max_requests=None,  # No limit on total requests
        backlog=2048,  # Handle request queue efficiently
        timeout_keep_alive=300,  # Set timeout to 300 seconds
        h11_max_incomplete_event_size=16777216,  # 16MB for large audio uploads
        access_log=True  # Enable access logging for debugging
    )
