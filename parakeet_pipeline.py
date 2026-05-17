"""
NVIDIA Parakeet NeMo ASR transcription pipeline.

This is a drop-in local ASR alternative for the audio translation workflow in
``fastapi_webui_v2.py``. It returns the same
``(segments, speaker_profiles, raw_text, cache_info)`` tuple as the Gemini,
WhisperX, and Qwen3-ASR + OmniVAD paths.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    import nemo.collections.asr as nemo_asr  # type: ignore[import]
    _NEMO_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    nemo_asr = None  # type: ignore[assignment]
    _NEMO_IMPORT_ERROR = exc

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None  # type: ignore[assignment]

try:
    from omnivad import OmniVAD
except Exception:
    OmniVAD = None  # type: ignore[assignment,misc]

try:
    from whisperx_pipeline import (
        _can_attempt_translation_backend,
        _make_translation_llm,
        _translate_texts_adaptively,
    )
except Exception:
    _can_attempt_translation_backend = None  # type: ignore[assignment]
    _make_translation_llm = None  # type: ignore[assignment]
    _translate_texts_adaptively = None  # type: ignore[assignment]


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int = 1,
    max_value: Optional[int] = None,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float = 0.0,
    max_value: Optional[float] = None,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = default
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


PARAKEET_ASR_MODEL = (
    os.getenv("PARAKEET_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3").strip()
    or "nvidia/parakeet-tdt-0.6b-v3"
)
PARAKEET_ASR_DEVICE = os.getenv("PARAKEET_ASR_DEVICE", "auto").strip() or "auto"
PARAKEET_ASR_CACHE_DIR = os.getenv(
    "PARAKEET_ASR_CACHE_DIR",
    os.path.join(_SCRIPT_DIR, "parakeet_cache"),
)
PARAKEET_ASR_CACHE_VERSION = 4
PARAKEET_ASR_BATCH_SIZE = _env_int("PARAKEET_ASR_BATCH_SIZE", 1)
PARAKEET_ASR_USE_OMNIVAD = os.getenv(
    "PARAKEET_ASR_USE_OMNIVAD",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO = os.getenv(
    "PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
PARAKEET_ASR_MAX_CLIP_SECONDS = _env_float(
    "PARAKEET_ASR_MAX_CLIP_SECONDS",
    300.0,
    min_value=30.0,
)
PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS = _env_float(
    "PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS",
    180.0,
    min_value=1.0,
)
PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS = _env_float(
    "PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS",
    60.0,
    min_value=1.0,
)
PARAKEET_ASR_MIN_SPLIT_SILENCE_SECONDS = _env_float(
    "PARAKEET_ASR_MIN_SPLIT_SILENCE_SECONDS",
    1.0,
    min_value=0.0,
)
PARAKEET_ASR_OMNIVAD_CHUNK_SECONDS = _env_float(
    "PARAKEET_ASR_OMNIVAD_CHUNK_SECONDS",
    600.0,
    min_value=30.0,
)
PARAKEET_ASR_OMNIVAD_OVERLAP_SECONDS = _env_float(
    "PARAKEET_ASR_OMNIVAD_OVERLAP_SECONDS",
    2.0,
    min_value=0.0,
)
PARAKEET_ASR_OMNIVAD_MERGE_GAP_SECONDS = _env_float(
    "PARAKEET_ASR_OMNIVAD_MERGE_GAP_SECONDS",
    0.001,
    min_value=0.0,
)
PARAKEET_ASR_OMNIVAD_MODEL_PATH = os.getenv(
    "PARAKEET_ASR_OMNIVAD_MODEL_PATH",
    os.getenv(
        "QWEN_OMNIVAD_OMNIVAD_MODEL_PATH",
        os.path.join(
            _SCRIPT_DIR,
            "checkpoints",
            "qwen_omnivad",
            "omnivad",
            "vad.omnivad",
        ),
    ),
).strip()
PARAKEET_ASR_USE_LOCAL_ATTENTION = os.getenv(
    "PARAKEET_ASR_USE_LOCAL_ATTENTION",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
PARAKEET_ASR_ATT_CONTEXT_SIZE = os.getenv(
    "PARAKEET_ASR_ATT_CONTEXT_SIZE",
    "256,256",
).strip()
PARAKEET_ASR_TRANSLATION_LLM = os.getenv(
    "PARAKEET_ASR_TRANSLATION_LLM",
    os.getenv("WHISPERX_TRANSLATION_LLM", "lightning-ai/gemma-4-31B-it"),
)
PARAKEET_ASR_TRANSLATION_BATCH_SIZE = _env_int(
    "PARAKEET_ASR_TRANSLATION_BATCH_SIZE",
    30,
)
PARAKEET_ASR_TRANSLATION_MAX_WORKERS = _env_int(
    "PARAKEET_ASR_TRANSLATION_MAX_WORKERS",
    10,
    max_value=10,
)

PARAKEET_SUPPORTED_LANGUAGES = (
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
)


@dataclass
class TimelineItem:
    start: float
    end: float
    text: str
    speaker: str = "speaker1"


@dataclass
class AudioChunk:
    start: float
    end: float
    split_reason: str = "full"
    silence_start: Optional[float] = None
    silence_end: Optional[float] = None


_PARAKEET_MODEL: Any = None
_PARAKEET_MODEL_DEVICE: Optional[str] = None
_OMNIVAD_MODEL_LOAD_PATH: Optional[str] = None


def is_parakeet_available() -> bool:
    return nemo_asr is not None


def _resolve_device() -> str:
    requested = (PARAKEET_ASR_DEVICE or "auto").strip()
    if requested.lower() != "auto":
        return requested
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parse_att_context_size() -> List[int]:
    raw_parts = [part.strip() for part in PARAKEET_ASR_ATT_CONTEXT_SIZE.split(",")]
    values: List[int] = []
    for part in raw_parts:
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            pass
    return values if len(values) == 2 else [256, 256]


def _load_parakeet_model() -> Any:
    global _PARAKEET_MODEL, _PARAKEET_MODEL_DEVICE
    if _PARAKEET_MODEL is not None:
        return _PARAKEET_MODEL
    if nemo_asr is None:
        detail = f" Import failed: {_NEMO_IMPORT_ERROR}" if _NEMO_IMPORT_ERROR else ""
        raise RuntimeError(
            "NVIDIA NeMo ASR is not installed. Install with "
            "`pip install -U \"nemo_toolkit[asr]\"` to use the Parakeet pipeline."
            + detail
        )

    print(f"Loading NVIDIA Parakeet ASR model: {PARAKEET_ASR_MODEL}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_ASR_MODEL)
    device = _resolve_device()
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    if PARAKEET_ASR_USE_LOCAL_ATTENTION and hasattr(model, "change_attention_model"):
        context_size = _parse_att_context_size()
        print(f"Parakeet: enabling local attention context {context_size}")
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=context_size,
        )
    _PARAKEET_MODEL = model
    _PARAKEET_MODEL_DEVICE = device
    return _PARAKEET_MODEL


def _release_memory() -> None:
    gc.collect()
    if torch is not None and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()


def _suffix_for_mime(mime_type: Optional[str]) -> str:
    if not mime_type:
        return ".audio"
    value = mime_type.split(";", 1)[0].strip().lower()
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/flac": ".flac",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/aac": ".aac",
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/webm": ".webm",
    }
    return mapping.get(value, ".audio")


def _write_temp_wav(audio_bytes: bytes, input_mime_type: Optional[str]) -> Tuple[str, float]:
    if AudioSegment is None:
        raise RuntimeError("pydub is required for the Parakeet pipeline; install pydub and ffmpeg.")

    fd, input_path = tempfile.mkstemp(suffix=_suffix_for_mime(input_mime_type))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(audio_bytes)
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16_000)
        duration_seconds = len(audio) / 1000.0
        out_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(out_fd)
        audio.export(wav_path, format="wav")
        return wav_path, duration_seconds
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


def _resolve_omnivad_model_path() -> Optional[str]:
    if not PARAKEET_ASR_OMNIVAD_MODEL_PATH:
        return None

    target_path = os.path.abspath(os.path.expanduser(PARAKEET_ASR_OMNIVAD_MODEL_PATH))
    if os.path.isfile(target_path):
        return target_path

    try:
        from omnivad.vad import default_model_dir

        source_path = os.path.join(default_model_dir(), "vad.omnivad")
        if os.path.isfile(source_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copyfile(source_path, target_path)
            print(f"Copied OmniVAD model to {target_path}")
            return target_path
    except Exception as exc:
        print(f"Warning: could not stage OmniVAD model for Parakeet: {exc}")

    return None


def _merge_nearby_speech_spans(
    spans: Sequence[Tuple[float, float]],
    *,
    max_gap_seconds: float,
) -> List[Tuple[float, float]]:
    cleaned = sorted(
        (max(0.0, float(start)), max(0.0, float(end)))
        for start, end in spans
        if float(end) > float(start)
    )
    if not cleaned or max_gap_seconds <= 0:
        return cleaned

    merged: List[Tuple[float, float]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _detect_omnivad_speech_spans(
    audio_path: str,
    *,
    duration_seconds: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Run OmniVAD and return merged and raw speech intervals in seconds."""
    global _OMNIVAD_MODEL_LOAD_PATH

    if not PARAKEET_ASR_USE_OMNIVAD or OmniVAD is None:
        return [], []

    try:
        _OMNIVAD_MODEL_LOAD_PATH = _resolve_omnivad_model_path()
        vad = OmniVAD(model_path=_OMNIVAD_MODEL_LOAD_PATH) if _OMNIVAD_MODEL_LOAD_PATH else OmniVAD()
        result = vad.detect(
            audio_path,
            chunk_seconds=PARAKEET_ASR_OMNIVAD_CHUNK_SECONDS,
            overlap_seconds=PARAKEET_ASR_OMNIVAD_OVERLAP_SECONDS,
        )
    except Exception as exc:
        print(f"Warning: OmniVAD detection for Parakeet failed: {exc}")
        return [], []

    timestamps = result.get("timestamps") if isinstance(result, dict) else None
    if not timestamps:
        return [], []

    clipped: List[Tuple[float, float]] = []
    for start, end in timestamps:
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if duration_seconds > 0:
            start_f = max(0.0, min(start_f, duration_seconds))
            end_f = max(0.0, min(end_f, duration_seconds))
        if end_f > start_f:
            clipped.append((start_f, end_f))

    if not clipped:
        return [], []

    raw_spans = sorted(clipped, key=lambda span: span[0])
    merged = _merge_nearby_speech_spans(
        raw_spans,
        max_gap_seconds=PARAKEET_ASR_OMNIVAD_MERGE_GAP_SECONDS,
    )
    preserved_gaps = sum(
        1
        for idx in range(1, len(merged))
        if merged[idx][0] > merged[idx - 1][1] + 0.05
    )
    print(
        f"OmniVAD [Parakeet] -> {len(raw_spans)} speech span(s) after clipping -> "
        f"{len(merged)} merged span(s); {preserved_gaps} silence gap(s) kept."
    )
    return merged, raw_spans


def _silence_gaps_from_speech_spans(
    speech_spans: Sequence[Tuple[float, float]],
    *,
    duration_seconds: float,
) -> List[Tuple[float, float]]:
    duration = max(0.0, float(duration_seconds or 0.0))
    if duration <= 0:
        return []

    clipped = _merge_nearby_speech_spans(
        [
            (
                max(0.0, min(float(start), duration)),
                max(0.0, min(float(end), duration)),
            )
            for start, end in speech_spans
            if float(end) > float(start)
        ],
        max_gap_seconds=PARAKEET_ASR_OMNIVAD_MERGE_GAP_SECONDS,
    )
    gaps: List[Tuple[float, float]] = []
    cursor = 0.0
    for start, end in clipped:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _build_silence_aligned_chunks(
    duration_seconds: float,
    speech_spans: Sequence[Tuple[float, float]],
    *,
    max_clip_seconds: float = PARAKEET_ASR_MAX_CLIP_SECONDS,
    search_start_seconds: float = PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS,
    search_window_seconds: float = PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS,
    min_silence_seconds: float = PARAKEET_ASR_MIN_SPLIT_SILENCE_SECONDS,
) -> List[AudioChunk]:
    duration = max(0.0, float(duration_seconds or 0.0))
    max_clip = max(1.0, float(max_clip_seconds or PARAKEET_ASR_MAX_CLIP_SECONDS))
    search_start = max(1.0, float(search_start_seconds or PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS))
    search_window = max(1.0, float(search_window_seconds or PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS))
    if duration <= 0:
        return []
    if duration <= max_clip + 0.001:
        return [AudioChunk(start=0.0, end=duration, split_reason="full")]

    gaps = _silence_gaps_from_speech_spans(
        speech_spans,
        duration_seconds=duration,
    )
    eps = 0.001
    chunks: List[AudioChunk] = []

    usable_gaps: List[Tuple[float, float, float, float]] = []
    for gap_start, gap_end in gaps:
        silence_len = gap_end - gap_start
        if silence_len < min_silence_seconds:
            continue
        split_point = (gap_start + gap_end) / 2.0
        if split_point <= eps or split_point >= duration - eps:
            continue
        usable_gaps.append((gap_start, gap_end, split_point, silence_len))

    cursor = 0.0
    while duration - cursor > max_clip + eps:
        window_start = cursor + search_start
        selected: Optional[Tuple[float, float, float, float]] = None
        while window_start < duration - eps:
            window_end = min(duration - eps, window_start + search_window)
            candidates = [
                gap
                for gap in usable_gaps
                if cursor + eps < gap[2] and window_start <= gap[2] <= window_end
            ]
            if candidates:
                selected = max(
                    candidates,
                    key=lambda gap: (gap[3], -abs(gap[2] - window_start)),
                )
                break
            window_start = window_end

        if selected is None:
            chunks.append(
                AudioChunk(
                    start=cursor,
                    end=duration,
                    split_reason="unsplit_no_silence",
                )
            )
            cursor = duration
            break

        gap_start, gap_end, split_point, _silence_len = selected
        if split_point <= cursor + eps or split_point >= duration - eps:
            chunks.append(
                AudioChunk(
                    start=cursor,
                    end=duration,
                    split_reason="unsplit_no_silence",
                )
            )
            cursor = duration
            break

        chunks.append(
            AudioChunk(
                start=cursor,
                end=split_point,
                split_reason="silence",
                silence_start=gap_start,
                silence_end=gap_end,
            )
        )
        cursor = split_point

    if cursor < duration - eps:
        chunks.append(
            AudioChunk(
                start=cursor,
                end=duration,
                split_reason="final" if chunks else "full",
            )
        )

    normalized = [
        AudioChunk(
            start=max(0.0, min(chunk.start, duration)),
            end=max(0.0, min(chunk.end, duration)),
            split_reason=chunk.split_reason,
            silence_start=chunk.silence_start,
            silence_end=chunk.silence_end,
        )
        for chunk in sorted(chunks, key=lambda item: item.start)
        if chunk.end > chunk.start + eps
    ]
    return normalized or [AudioChunk(start=0.0, end=duration, split_reason="full")]


def _export_audio_chunks(
    wav_path: str,
    chunks: Sequence[AudioChunk],
) -> List[Tuple[AudioChunk, str]]:
    if AudioSegment is None:
        raise RuntimeError("pydub is required for Parakeet chunking; install pydub and ffmpeg.")

    full = AudioSegment.from_file(wav_path)
    duration_ms = max(1, len(full))
    jobs: List[Tuple[AudioChunk, str]] = []
    for chunk in chunks:
        start_ms = max(0, min(int(round(chunk.start * 1000.0)), duration_ms - 1))
        end_ms = max(start_ms + 1, min(int(round(chunk.end * 1000.0)), duration_ms))
        fd, clip_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        full[start_ms:end_ms].export(clip_path, format="wav")
        actual_chunk = AudioChunk(
            start=start_ms / 1000.0,
            end=end_ms / 1000.0,
            split_reason=chunk.split_reason,
            silence_start=chunk.silence_start,
            silence_end=chunk.silence_end,
        )
        jobs.append((actual_chunk, clip_path))
    return jobs


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;])\s+|(?<=[.!?;])", text)
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return cleaned or [text]


def _join_words(words: Sequence[str]) -> str:
    pieces: List[str] = []
    for raw_word in words:
        word = str(raw_word or "").strip()
        if not word:
            continue
        if not pieces:
            pieces.append(word)
            continue
        if re.match(r"^[,.;:!?%)}\]]", word):
            pieces.append(word)
        else:
            pieces.append(f" {word}")
    return "".join(pieces).strip()


def _stamp_attr(stamp: Any, *names: str) -> Any:
    for name in names:
        if isinstance(stamp, dict) and name in stamp:
            return stamp.get(name)
        value = getattr(stamp, name, None)
        if value is not None:
            return value
    return None


def _stamp_items(stamps: Any) -> List[Any]:
    if stamps is None:
        return []
    if isinstance(stamps, dict):
        embedded = stamps.get("items") or stamps.get("segments") or stamps.get("words")
        if embedded is not None:
            return list(embedded)
        return [stamps]
    try:
        return list(stamps)
    except TypeError:
        return []


def _coerce_seconds(value: Any, duration_seconds: float) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    # Some timestamp backends report sample offsets. Parakeet's public API uses
    # seconds, but this keeps cached/offline variants sensible.
    if duration_seconds > 0 and parsed > duration_seconds * 2 and parsed > 1000:
        parsed = parsed / 16_000.0
    return max(0.0, parsed)


def _proportional_timeline(
    text: str,
    start: float,
    end: float,
    *,
    speaker: str = "speaker1",
) -> List[TimelineItem]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    duration = max(0.1, end - start)
    weights = [max(1, len(sentence)) for sentence in sentences]
    total = float(sum(weights))
    cursor = start
    items: List[TimelineItem] = []
    for idx, sentence in enumerate(sentences):
        if idx == len(sentences) - 1:
            next_end = end
        else:
            next_end = cursor + duration * (weights[idx] / total)
        items.append(
            TimelineItem(
                start=cursor,
                end=max(cursor + 0.1, next_end),
                text=sentence,
                speaker=speaker,
            )
        )
        cursor = next_end
    return items


def _segment_timestamps_to_items(
    stamps: Any,
    *,
    duration_seconds: float,
    speaker: str = "speaker1",
) -> List[TimelineItem]:
    items: List[TimelineItem] = []
    for stamp in _stamp_items(stamps):
        text = str(
            _stamp_attr(stamp, "segment", "text", "word", "token") or ""
        ).strip()
        start = _coerce_seconds(
            _stamp_attr(stamp, "start", "start_time", "start_offset"),
            duration_seconds,
        )
        end = _coerce_seconds(
            _stamp_attr(stamp, "end", "end_time", "end_offset"),
            duration_seconds,
        )
        if not text or start is None or end is None:
            continue
        if duration_seconds > 0:
            start = min(start, duration_seconds)
            end = min(end, duration_seconds)
        if end <= start:
            continue
        items.append(TimelineItem(start=start, end=end, text=text, speaker=speaker))
    return items


def _word_timestamps_to_items(
    stamps: Any,
    fallback_text: str,
    *,
    duration_seconds: float,
    speaker: str = "speaker1",
) -> List[TimelineItem]:
    items: List[TimelineItem] = []
    current_words: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def flush() -> None:
        nonlocal current_words, current_start, current_end
        text = _join_words(current_words)
        if text and current_start is not None and current_end is not None and current_end > current_start:
            items.append(
                TimelineItem(
                    start=current_start,
                    end=current_end,
                    text=text,
                    speaker=speaker,
                )
            )
        current_words = []
        current_start = None
        current_end = None

    for stamp in _stamp_items(stamps):
        word = str(_stamp_attr(stamp, "word", "text", "token") or "").strip()
        start = _coerce_seconds(
            _stamp_attr(stamp, "start", "start_time", "start_offset"),
            duration_seconds,
        )
        end = _coerce_seconds(
            _stamp_attr(stamp, "end", "end_time", "end_offset"),
            duration_seconds,
        )
        if not word or start is None or end is None:
            continue
        if duration_seconds > 0:
            start = min(start, duration_seconds)
            end = min(end, duration_seconds)
        if end <= start:
            continue
        if current_start is None:
            current_start = start
        current_words.append(word)
        current_end = end
        if re.search(r"[.!?;]$", word) or len(_join_words(current_words)) >= 80:
            flush()
    flush()
    if items:
        return items
    return []


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "transcript", "prediction"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr in ("text", "transcript", "prediction"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(result or "").strip()


def _extract_language(result: Any) -> Optional[str]:
    if isinstance(result, dict):
        value = result.get("language") or result.get("lang")
    else:
        value = getattr(result, "language", None) or getattr(result, "lang", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_timestamp_payload(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("timestamp") or result.get("timestamps")
    return getattr(result, "timestamp", None) or getattr(result, "timestamps", None)


def _offset_timeline_items(
    items: Sequence[TimelineItem],
    *,
    start_offset: float,
    end_limit: Optional[float],
) -> List[TimelineItem]:
    offset = max(0.0, float(start_offset or 0.0))
    limit = float(end_limit) if end_limit is not None else None
    adjusted: List[TimelineItem] = []
    for item in items:
        start = offset + max(0.0, float(item.start or 0.0))
        end = offset + max(0.0, float(item.end or 0.0))
        if limit is not None:
            start = min(max(offset, start), limit)
            end = min(max(offset, end), limit)
        if end <= start:
            end = start + 0.05
            if limit is not None:
                end = min(end, limit)
        if end > start:
            adjusted.append(
                TimelineItem(
                    start=start,
                    end=end,
                    text=item.text,
                    speaker=item.speaker,
                )
            )
    return adjusted


def _transcription_to_timeline(
    result: Any,
    *,
    duration_seconds: float,
    start_offset: float = 0.0,
    end_limit: Optional[float] = None,
) -> Tuple[List[TimelineItem], str, Optional[str], str]:
    text = _extract_text(result)
    language = _extract_language(result)
    payload = _extract_timestamp_payload(result)
    timestamp_source = "none"

    if isinstance(payload, dict):
        segment_stamps = payload.get("segment") or payload.get("segments")
        word_stamps = payload.get("word") or payload.get("words")
    else:
        segment_stamps = None
        word_stamps = None

    items = _segment_timestamps_to_items(
        segment_stamps,
        duration_seconds=duration_seconds,
    )
    if items:
        timestamp_source = "parakeet_segment"
    else:
        items = _word_timestamps_to_items(
            word_stamps,
            text,
            duration_seconds=duration_seconds,
        )
        if items:
            timestamp_source = "parakeet_word"

    if not items:
        items = _proportional_timeline(text, 0.0, max(duration_seconds, 0.1))
        timestamp_source = "proportional"
    if start_offset or end_limit is not None:
        items = _offset_timeline_items(
            items,
            start_offset=start_offset,
            end_limit=end_limit,
        )
    return items, text, language, timestamp_source


def _items_to_segments(items: Sequence[TimelineItem]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for item in items:
        text = (item.text or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": _format_timestamp(item.start),
                "end": _format_timestamp(max(item.end, item.start + 0.05)),
                "speaker": item.speaker or "speaker1",
                "source_text": text,
                "translated_text": "",
            }
        )
    return segments


def _speaker_profiles(items: Iterable[TimelineItem]) -> List[Dict[str, Any]]:
    speakers: List[str] = []
    for item in items:
        speaker = (item.speaker or "speaker1").strip() or "speaker1"
        if speaker not in speakers:
            speakers.append(speaker)
    return [
        {
            "id": speaker,
            "description": (
                "Default speaker for NVIDIA Parakeet ASR pipeline."
                if speaker == "speaker1"
                else "Detected by NVIDIA Parakeet ASR pipeline."
            ),
        }
        for speaker in (speakers or ["speaker1"])
    ]


def _cache_key(
    audio_hash: str,
    *,
    dest_language: str,
    enable_translation: bool,
    translation_llm_model: Optional[str],
    input_mime_type: Optional[str],
) -> str:
    raw = "|".join(
        [
            f"v{PARAKEET_ASR_CACHE_VERSION}",
            audio_hash,
            f"mime={input_mime_type or ''}",
            f"dst={dest_language}",
            f"translate={enable_translation}",
            f"model={PARAKEET_ASR_MODEL}",
            f"device={PARAKEET_ASR_DEVICE}",
            f"batch={PARAKEET_ASR_BATCH_SIZE}",
            f"omnivad={PARAKEET_ASR_USE_OMNIVAD}",
            f"omnivad_required={PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO}",
            f"omnivad_model={PARAKEET_ASR_OMNIVAD_MODEL_PATH or 'default'}",
            f"max_clip={PARAKEET_ASR_MAX_CLIP_SECONDS:.3f}",
            f"split_search_start={PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS:.3f}",
            f"split_search_window={PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS:.3f}",
            f"min_split_silence={PARAKEET_ASR_MIN_SPLIT_SILENCE_SECONDS:.3f}",
            f"vad_chunk={PARAKEET_ASR_OMNIVAD_CHUNK_SECONDS:.3f}",
            f"vad_overlap={PARAKEET_ASR_OMNIVAD_OVERLAP_SECONDS:.3f}",
            f"vad_merge_gap={PARAKEET_ASR_OMNIVAD_MERGE_GAP_SECONDS:.4f}",
            f"local_attn={PARAKEET_ASR_USE_LOCAL_ATTENTION}",
            f"att_ctx={PARAKEET_ASR_ATT_CONTEXT_SIZE}",
            f"llm={translation_llm_model or 'default'}",
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_key: str) -> str:
    os.makedirs(PARAKEET_ASR_CACHE_DIR, exist_ok=True)
    return os.path.join(PARAKEET_ASR_CACHE_DIR, f"parakeet_{cache_key}.json")


def _load_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(cache_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_cache(cache_key: str, record: Dict[str, Any]) -> Optional[str]:
    path = _cache_path(cache_key)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        return path
    except Exception as exc:
        print(f"Warning: Parakeet cache write failed: {exc}")
        return None


def _transcribe_with_timestamps(model: Any, wav_path: str) -> Any:
    kwargs = {
        "timestamps": True,
        "batch_size": PARAKEET_ASR_BATCH_SIZE,
    }
    try:
        results = model.transcribe([wav_path], **kwargs)
    except TypeError:
        kwargs.pop("batch_size", None)
        results = model.transcribe([wav_path], **kwargs)
    if isinstance(results, tuple) and results:
        results = results[0]
    if not results:
        raise RuntimeError("Parakeet returned no transcription results.")
    return results[0]


def _transcribe_paths_with_timestamps(model: Any, wav_paths: Sequence[str]) -> List[Any]:
    paths = [path for path in wav_paths if path]
    if not paths:
        return []

    kwargs = {
        "timestamps": True,
        "batch_size": PARAKEET_ASR_BATCH_SIZE,
    }
    try:
        results = model.transcribe(paths, **kwargs)
    except TypeError:
        kwargs.pop("batch_size", None)
        results = model.transcribe(paths, **kwargs)
    if isinstance(results, tuple) and results:
        results = results[0]
    if results is None:
        return []
    if len(paths) == 1 and not isinstance(results, (list, tuple)):
        return [results]
    return list(results)


def _combined_timestamp_source(sources: Sequence[str], *, chunked: bool) -> str:
    clean = [source for source in sources if source]
    if not clean:
        return "none"
    if len(set(clean)) == 1:
        source = clean[0]
    elif any(source != "proportional" for source in clean):
        source = "parakeet_mixed"
    else:
        source = "proportional"
    return f"{source}_chunked" if chunked else source


def _transcribe_chunks_with_timestamps(
    model: Any,
    wav_path: str,
    chunks: Sequence[AudioChunk],
) -> Tuple[List[TimelineItem], str, Optional[str], str, List[Dict[str, Any]]]:
    if not chunks:
        raise RuntimeError("Parakeet chunk planner returned no audio chunks.")

    chunked = len(chunks) > 1
    if not chunked:
        result = _transcribe_with_timestamps(model, wav_path)
        duration = max(0.1, chunks[0].end - chunks[0].start)
        items, text, language, timestamp_source = _transcription_to_timeline(
            result,
            duration_seconds=duration,
        )
        return (
            items,
            text,
            language,
            timestamp_source,
            [
                {
                    "index": 1,
                    "start": chunks[0].start,
                    "end": chunks[0].end,
                    "duration": duration,
                    "split_reason": chunks[0].split_reason,
                }
            ],
        )

    print(
        f"Running NVIDIA Parakeet ASR on {len(chunks)} silence-aligned chunk(s) "
        f"(search starts at {PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS:.1f}s "
        f"in {PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS:.1f}s windows; "
        f"cut at silence midpoint; no fixed mid-speech cuts; "
        f"batch_size={PARAKEET_ASR_BATCH_SIZE})..."
    )
    jobs = _export_audio_chunks(wav_path, chunks)
    results_by_index: List[Tuple[int, List[TimelineItem], str, Optional[str], str]] = []
    chunk_records: List[Dict[str, Any]] = []
    try:
        batch_size = max(1, PARAKEET_ASR_BATCH_SIZE)
        for start_idx in range(0, len(jobs), batch_size):
            batch_jobs = jobs[start_idx : start_idx + batch_size]
            batch_paths = [path for _chunk, path in batch_jobs]
            batch_results = _transcribe_paths_with_timestamps(model, batch_paths)
            if len(batch_results) != len(batch_jobs):
                raise RuntimeError(
                    "Parakeet returned an unexpected number of chunk transcription results "
                    f"({len(batch_results)} for {len(batch_jobs)} inputs)."
                )

            for offset, ((chunk, _path), result) in enumerate(zip(batch_jobs, batch_results)):
                index = start_idx + offset + 1
                chunk_duration = max(0.1, chunk.end - chunk.start)
                items, text, language, source = _transcription_to_timeline(
                    result,
                    duration_seconds=chunk_duration,
                    start_offset=chunk.start,
                    end_limit=chunk.end,
                )
                preview = text.replace("\n", " ").strip()
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                print(
                    f"  Chunk {index}/{len(jobs)}: "
                    f"[{chunk.start:.2f}s -> {chunk.end:.2f}s] {preview}"
                )
                results_by_index.append((index, items, text, language, source))
                chunk_records.append(
                    {
                        "index": index,
                        "start": chunk.start,
                        "end": chunk.end,
                        "duration": chunk_duration,
                        "split_reason": chunk.split_reason,
                        "silence_start": chunk.silence_start,
                        "silence_end": chunk.silence_end,
                        "timestamp_source": source,
                    }
                )
    finally:
        for _chunk, path in jobs:
            try:
                os.remove(path)
            except OSError:
                pass

    results_by_index.sort(key=lambda row: row[0])
    chunk_records.sort(key=lambda row: int(row["index"]))
    all_items: List[TimelineItem] = []
    fragments: List[str] = []
    detected_language: Optional[str] = None
    timestamp_sources: List[str] = []
    for _index, items, text, language, source in results_by_index:
        if items:
            all_items.extend(items)
        if text.strip():
            fragments.append(text.strip())
        if language:
            detected_language = language
        timestamp_sources.append(source)
    return (
        all_items,
        " ".join(fragments).strip(),
        detected_language,
        _combined_timestamp_source(timestamp_sources, chunked=True),
        chunk_records,
    )


def _translate_segments(
    segments: List[Dict[str, Any]],
    *,
    dest_language: str,
    source_language: Optional[str],
    llm_model: str,
    translation_batch_size: int,
    translation_max_workers: int,
) -> None:
    print("=" * 60)
    print(f"PASS C [Parakeet]: LLM translation -> {dest_language}")
    print("=" * 60)

    if (
        _make_translation_llm is None
        or _translate_texts_adaptively is None
        or _can_attempt_translation_backend is None
        or not _can_attempt_translation_backend()
    ):
        print(
            "Warning: translation helpers are unavailable; "
            "leaving translated_text empty."
        )
        return

    translation_jobs: List[Dict[str, Any]] = []
    for start in range(0, len(segments), translation_batch_size):
        chunk = segments[start : start + translation_batch_size]
        non_empty = [seg for seg in chunk if (seg.get("source_text") or "").strip()]
        if not non_empty:
            continue
        translation_jobs.append(
            {
                "label": f"Batch {start // translation_batch_size + 1}",
                "start": start + 1,
                "end": min(start + translation_batch_size, len(segments)),
                "segments": non_empty,
                "source_texts": [seg["source_text"] for seg in non_empty],
            }
        )

    if not translation_jobs:
        return

    worker_count = min(max(1, translation_max_workers), len(translation_jobs))
    print(
        f"  Translating {len(translation_jobs)} batch(es) with "
        f"up to {worker_count} parallel worker(s)."
    )

    def _run_translation_job(job: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        batch_label = str(job["label"])
        print(
            f"\n  Translating {batch_label} "
            f"(Segments {job['start']} to {job['end']})..."
        )
        llm = _make_translation_llm(
            llm_model=llm_model,
            dest_language=dest_language,
            source_language=source_language,
            batch_label=batch_label,
        )
        translated = _translate_texts_adaptively(
            llm=llm,
            source_texts=job["source_texts"],
            dest_language=dest_language,
            batch_label=batch_label,
            source_language=source_language,
        )
        return job, translated

    def _apply_translation_result(job: Dict[str, Any], translated_texts: List[str]) -> None:
        batch_label = str(job["label"])
        source_texts = job["source_texts"]
        if translated_texts and len(translated_texts) == len(source_texts):
            for seg_ref, trans in zip(job["segments"], translated_texts):
                seg_ref["translated_text"] = trans
            blank_count = sum(1 for item in translated_texts if not item.strip())
            if blank_count:
                print(
                    f"  [{batch_label}] Translated with "
                    f"{blank_count} empty fallback item(s)."
                )
            else:
                print(f"  [{batch_label}] Batch translated successfully.")
        else:
            print(f"  [{batch_label}] Batch failed after retries.")

    if worker_count <= 1:
        for job in translation_jobs:
            result_job, translated_texts = _run_translation_job(job)
            _apply_translation_result(result_job, translated_texts)
        return

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="parakeet_translate",
    ) as pool:
        pending_futures: Dict[Any, Dict[str, Any]] = {}
        next_job_index = 0

        def _submit_next_translation_job() -> bool:
            nonlocal next_job_index
            if next_job_index >= len(translation_jobs):
                return False
            job = translation_jobs[next_job_index]
            pending_futures[pool.submit(_run_translation_job, job)] = job
            next_job_index += 1
            return True

        for _ in range(worker_count):
            _submit_next_translation_job()

        while pending_futures:
            done, _ = wait(
                pending_futures,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                job = pending_futures.pop(future)
                try:
                    result_job, translated_texts = future.result()
                except Exception as exc:
                    print(f"  [{job['label']}] Batch failed: {exc}")
                else:
                    _apply_translation_result(result_job, translated_texts)
                _submit_next_translation_job()


def translate_audio(
    audio_bytes: bytes,
    *,
    input_mime_type: Optional[str] = None,
    dest_language: str = "English",
    enable_translation: bool = True,
    translation_llm_model: Optional[str] = None,
    translation_batch_size: int = PARAKEET_ASR_TRANSLATION_BATCH_SIZE,
    translation_max_workers: int = PARAKEET_ASR_TRANSLATION_MAX_WORKERS,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Run NVIDIA Parakeet ASR and optional local LLM translation.

    Parakeet auto-detects supported European languages and English, so no
    source-language prompt is passed to the model.
    """
    if not is_parakeet_available():
        detail = f" Import failed: {_NEMO_IMPORT_ERROR}" if _NEMO_IMPORT_ERROR else ""
        raise RuntimeError(
            "NVIDIA NeMo ASR is not installed. Install with "
            "`pip install -U \"nemo_toolkit[asr]\"` to use the Parakeet pipeline."
            + detail
        )

    translation_batch_size = max(1, int(translation_batch_size or 1))
    try:
        translation_max_workers = int(translation_max_workers or 1)
    except (TypeError, ValueError):
        translation_max_workers = PARAKEET_ASR_TRANSLATION_MAX_WORKERS
    translation_max_workers = max(1, min(10, translation_max_workers))
    llm_model = translation_llm_model or PARAKEET_ASR_TRANSLATION_LLM

    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    cache_key = _cache_key(
        audio_hash,
        dest_language=dest_language,
        enable_translation=enable_translation,
        translation_llm_model=llm_model,
        input_mime_type=input_mime_type,
    )
    cache_info: Dict[str, Any] = {
        "audio_md5": audio_hash,
        "hit": False,
        "force_refresh": bool(force_refresh),
        "pipeline": "parakeet",
        "parakeet_model": PARAKEET_ASR_MODEL,
        "parakeet_device": _PARAKEET_MODEL_DEVICE or _resolve_device(),
        "local_attention": bool(PARAKEET_ASR_USE_LOCAL_ATTENTION),
        "supported_languages": list(PARAKEET_SUPPORTED_LANGUAGES),
        "translation_batch_size": translation_batch_size,
        "translation_max_workers": translation_max_workers,
        "omnivad_enabled": bool(PARAKEET_ASR_USE_OMNIVAD),
        "omnivad_available": bool(OmniVAD is not None),
        "omnivad_required_for_long_audio": bool(PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO),
        "max_clip_seconds": PARAKEET_ASR_MAX_CLIP_SECONDS,
        "split_search_start_seconds": PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS,
        "split_search_window_seconds": PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS,
        "min_split_silence_seconds": PARAKEET_ASR_MIN_SPLIT_SILENCE_SECONDS,
    }

    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached and isinstance(cached.get("segments"), list):
            cache_info["hit"] = True
            cache_info["cache_file"] = os.path.basename(_cache_path(cache_key))
            cache_info["created_at"] = cached.get("created_at")
            return (
                cached["segments"],
                cached.get("speaker_profiles") or [],
                cached.get("raw_text", ""),
                cache_info,
            )

    wav_path, duration = _write_temp_wav(audio_bytes, input_mime_type)
    try:
        print("Running NVIDIA Parakeet ASR transcription...")
        speech_spans: List[Tuple[float, float]] = []
        raw_vad_spans: List[Tuple[float, float]] = []
        silence_gaps: List[Tuple[float, float]] = []
        vad_attempted = False
        if duration > PARAKEET_ASR_MAX_CLIP_SECONDS + 0.001 and PARAKEET_ASR_USE_OMNIVAD:
            if OmniVAD is None:
                if PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO:
                    raise RuntimeError(
                        "OmniVAD is required to split long audio for the Parakeet pipeline, "
                        "but the omnivad package is not importable. Install omnivad or set "
                        "PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO=0 to use fixed-size chunks."
                    )
                print(
                    "Warning: OmniVAD is unavailable; Parakeet will split long audio "
                    "into fixed-size chunks."
                )
            else:
                vad_attempted = True
                print("OmniVAD [Parakeet]: detecting speech spans for silence-based chunking...")
                speech_spans, raw_vad_spans = _detect_omnivad_speech_spans(
                    wav_path,
                    duration_seconds=duration,
                )
                if not speech_spans and PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO:
                    raise RuntimeError(
                        "OmniVAD did not return usable speech timestamps for this long audio, "
                        "so Parakeet cannot build silence-aligned chunks safely. Check the "
                        "OmniVAD model path/logs, or set "
                        "PARAKEET_ASR_REQUIRE_OMNIVAD_FOR_LONG_AUDIO=0 to use fixed-size chunks."
                    )
                silence_gaps = _silence_gaps_from_speech_spans(
                    speech_spans,
                    duration_seconds=duration,
                )

        chunks = _build_silence_aligned_chunks(
            duration,
            speech_spans,
            max_clip_seconds=PARAKEET_ASR_MAX_CLIP_SECONDS,
            search_start_seconds=PARAKEET_ASR_SPLIT_SEARCH_START_SECONDS,
            search_window_seconds=PARAKEET_ASR_SPLIT_SEARCH_WINDOW_SECONDS,
            min_silence_seconds=PARAKEET_ASR_MIN_SPLIT_SILENCE_SECONDS,
        )
        model = _load_parakeet_model()
        items, text, detected_language, timestamp_source, chunk_records = (
            _transcribe_chunks_with_timestamps(
                model,
                wav_path,
                chunks,
            )
        )
        if not text:
            raise RuntimeError("Parakeet returned an empty transcription.")

        segments = _items_to_segments(items)
        if not segments:
            raise RuntimeError("Parakeet returned no usable transcript segments.")
        speaker_profiles = _speaker_profiles(items)
        cache_info["source_language"] = detected_language or "auto"
        cache_info["timestamp_source"] = timestamp_source
        cache_info["duration_seconds"] = duration
        cache_info["omnivad_attempted"] = vad_attempted
        cache_info["omnivad_model_path"] = _OMNIVAD_MODEL_LOAD_PATH
        cache_info["raw_vad_span_count"] = len(raw_vad_spans)
        cache_info["merged_vad_span_count"] = len(speech_spans)
        cache_info["silence_gap_count"] = len(silence_gaps)
        cache_info["parakeet_chunk_count"] = len(chunk_records)

        if enable_translation and dest_language:
            _translate_segments(
                segments,
                dest_language=dest_language,
                source_language=detected_language,
                llm_model=llm_model,
                translation_batch_size=translation_batch_size,
                translation_max_workers=translation_max_workers,
            )
            cache_info["translation_llm_model"] = llm_model
        elif not enable_translation:
            for segment in segments:
                segment["translated_text"] = segment.get("source_text", "")

        raw_output = {
            "pipeline": "parakeet",
            "parakeet_model": PARAKEET_ASR_MODEL,
            "parakeet_device": _PARAKEET_MODEL_DEVICE,
            "source_language": detected_language or "auto",
            "timestamp_source": timestamp_source,
            "duration_seconds": duration,
            "omnivad_model_path": _OMNIVAD_MODEL_LOAD_PATH,
            "raw_vad_spans": raw_vad_spans,
            "merged_vad_spans": speech_spans,
            "silence_gaps": silence_gaps,
            "asr_chunks": chunk_records,
            "text": text,
            "speakers": speaker_profiles,
            "segments": segments,
        }
        raw_text = json.dumps(raw_output, ensure_ascii=False, indent=2)
        cache_record = {
            "created_at": time.time(),
            "segments": segments,
            "speaker_profiles": speaker_profiles,
            "raw_text": raw_text,
            "source_language": detected_language,
            "translation_llm_model": llm_model if enable_translation else None,
            "translation_batch_size": translation_batch_size if enable_translation else None,
            "translation_max_workers": translation_max_workers if enable_translation else None,
            "timestamp_source": timestamp_source,
            "duration_seconds": duration,
            "raw_vad_spans": raw_vad_spans,
            "merged_vad_spans": speech_spans,
            "silence_gaps": silence_gaps,
            "asr_chunks": chunk_records,
        }
        cache_path = _write_cache(cache_key, cache_record)
        if cache_path:
            cache_info["cache_file"] = os.path.basename(cache_path)
        return segments, speaker_profiles, raw_text, cache_info
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        _release_memory()


_run_parakeet_pipeline_sync = translate_audio
