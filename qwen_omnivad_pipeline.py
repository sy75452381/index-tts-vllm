"""
Qwen3-ASR + OmniVAD audio translation pipeline.

This module implements the same return contract as the Gemini and WhisperX
pipelines used by fastapi_webui_v2.py:

    (segments, speaker_profiles, raw_json_text, cache_info)

It keeps the rest of the existing translation workflow unchanged: segment
normalization, editable sessions, subtitle export, and IndexTTS synthesis all
continue to run through the existing code paths.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None  # type: ignore[assignment]

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None  # type: ignore[assignment]

try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None  # type: ignore[assignment,misc]

try:
    from omnivad import OmniVAD
except ImportError:
    OmniVAD = None  # type: ignore[assignment,misc]

try:
    from whisperx_pipeline import (
        LitaiLLM,
        _translate_texts_adaptively,
    )
except Exception:
    LitaiLLM = None  # type: ignore[assignment]
    _translate_texts_adaptively = None  # type: ignore[assignment]


QWEN_OMNIVAD_MODEL_DIR = os.getenv(
    "QWEN_OMNIVAD_MODEL_DIR",
    os.path.join(_SCRIPT_DIR, "checkpoints", "qwen_omnivad"),
)
QWEN_OMNIVAD_HF_CACHE_DIR = os.path.join(QWEN_OMNIVAD_MODEL_DIR, ".cache", "huggingface")
QWEN_ASR_LOCAL_DIR = os.getenv("QWEN_ASR_LOCAL_DIR", "").strip()
QWEN_ASR_FORCED_ALIGNER_LOCAL_DIR = os.getenv("QWEN_ASR_FORCED_ALIGNER_LOCAL_DIR", "").strip()
QWEN_OMNIVAD_MODEL_FORCE_DOWNLOAD = os.getenv("QWEN_OMNIVAD_MODEL_FORCE_DOWNLOAD", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
QWEN_OMNIVAD_MODEL_LOCAL_FILES_ONLY = os.getenv("QWEN_OMNIVAD_MODEL_LOCAL_FILES_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
QWEN_OMNIVAD_MODEL_ALLOW_PATTERNS = os.getenv("QWEN_OMNIVAD_MODEL_ALLOW_PATTERNS", "").strip()
QWEN_OMNIVAD_OMNIVAD_MODEL_PATH = os.getenv(
    "QWEN_OMNIVAD_OMNIVAD_MODEL_PATH",
    os.path.join(QWEN_OMNIVAD_MODEL_DIR, "omnivad", "vad.omnivad"),
).strip()
QWEN_OMNIVAD_CACHE_DIR = os.path.join(_SCRIPT_DIR, "qwen_omnivad_cache")
QWEN_OMNIVAD_CACHE_VERSION = 3

QWEN_ASR_MODEL = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
QWEN_ASR_BACKEND = os.getenv("QWEN_ASR_BACKEND", "transformers").strip().lower()
QWEN_ASR_DEVICE = os.getenv("QWEN_ASR_DEVICE", "cuda:0")
QWEN_ASR_DTYPE = os.getenv("QWEN_ASR_DTYPE", "bfloat16")
QWEN_ASR_MAX_BATCH_SIZE = int(os.getenv("QWEN_ASR_MAX_BATCH_SIZE", "8"))
QWEN_ASR_MAX_NEW_TOKENS = int(os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "4096"))
QWEN_ASR_FORCED_ALIGNER = os.getenv("QWEN_ASR_FORCED_ALIGNER", "").strip()

QWEN_OMNIVAD_TRANSLATION_LLM = os.getenv(
    "QWEN_OMNIVAD_TRANSLATION_LLM",
    os.getenv("WHISPERX_TRANSLATION_LLM", "lightning-ai/gemma-4-31B-it"),
)
QWEN_OMNIVAD_TRANSLATION_BATCH_SIZE = int(os.getenv("QWEN_OMNIVAD_TRANSLATION_BATCH_SIZE", "30"))
QWEN_OMNIVAD_TRANSLATION_MAX_WORKERS = int(os.getenv("QWEN_OMNIVAD_TRANSLATION_MAX_WORKERS", "10"))
QWEN_OMNIVAD_USE_OMNIVAD = os.getenv("QWEN_OMNIVAD_USE_OMNIVAD", "1").strip().lower() not in {"0", "false", "no"}
QWEN_OMNIVAD_REQUIRE_VAD_TIMELINE = os.getenv(
    "QWEN_OMNIVAD_REQUIRE_VAD_TIMELINE",
    "1",
).strip().lower() not in {"0", "false", "no"}
QWEN_OMNIVAD_CHUNK_SECONDS = float(os.getenv("QWEN_OMNIVAD_CHUNK_SECONDS", "600"))
QWEN_OMNIVAD_OVERLAP_SECONDS = float(os.getenv("QWEN_OMNIVAD_OVERLAP_SECONDS", "2"))
QWEN_OMNIVAD_MERGE_GAP_SECONDS = float(os.getenv("QWEN_OMNIVAD_MERGE_GAP_SECONDS", "0.5"))

_LANGUAGE_ALIASES = {
    "auto": None,
    "detect": None,
    "unknown": None,
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "it": "Italian",
    "italian": "Italian",
}


@dataclass
class TimelineItem:
    start: float
    end: float
    text: str
    speaker: str = "speaker1"


_ASR_MODEL: Any = None
_ASR_MODEL_LOAD_PATH: Optional[str] = None
_FORCED_ALIGNER_LOAD_PATH: Optional[str] = None
_OMNIVAD_MODEL_LOAD_PATH: Optional[str] = None


def _safe_model_dir_name(model_ref: str) -> str:
    value = (model_ref or "").strip().strip("/")
    value = value.replace("\\", "/").replace("/", "--")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value) or "model"


def _looks_like_local_path(model_ref: str) -> bool:
    if not model_ref:
        return False
    expanded = os.path.expanduser(model_ref)
    return (
        os.path.isdir(expanded)
        or model_ref.startswith((".", "~", os.sep))
        or (os.altsep is not None and model_ref.startswith(os.altsep))
    )


def _snapshot_allow_patterns() -> Optional[List[str]]:
    if not QWEN_OMNIVAD_MODEL_ALLOW_PATTERNS:
        return None
    return [
        pattern.strip()
        for pattern in QWEN_OMNIVAD_MODEL_ALLOW_PATTERNS.split(",")
        if pattern.strip()
    ] or None


def _resolve_hf_model_path(
    model_ref: str,
    *,
    local_dir_override: str = "",
    label: str,
) -> str:
    if local_dir_override:
        return os.path.abspath(os.path.expanduser(local_dir_override))

    if _looks_like_local_path(model_ref):
        return os.path.abspath(os.path.expanduser(model_ref))

    if snapshot_download is None:
        raise RuntimeError(
            f"huggingface_hub is required to download {label} model '{model_ref}' "
            f"into {QWEN_OMNIVAD_MODEL_DIR}."
        )

    os.makedirs(QWEN_OMNIVAD_MODEL_DIR, exist_ok=True)
    os.makedirs(QWEN_OMNIVAD_HF_CACHE_DIR, exist_ok=True)
    local_dir = os.path.join(QWEN_OMNIVAD_MODEL_DIR, _safe_model_dir_name(model_ref))
    print(f"Downloading/loading {label} model '{model_ref}' in {local_dir}")
    return snapshot_download(
        repo_id=model_ref,
        local_dir=local_dir,
        cache_dir=QWEN_OMNIVAD_HF_CACHE_DIR,
        allow_patterns=_snapshot_allow_patterns(),
        force_download=QWEN_OMNIVAD_MODEL_FORCE_DOWNLOAD,
        local_files_only=QWEN_OMNIVAD_MODEL_LOCAL_FILES_ONLY,
    )


def _resolve_omnivad_model_path() -> Optional[str]:
    if not QWEN_OMNIVAD_OMNIVAD_MODEL_PATH:
        return None
    target_path = os.path.abspath(os.path.expanduser(QWEN_OMNIVAD_OMNIVAD_MODEL_PATH))
    if os.path.isfile(target_path):
        return target_path
    try:
        from omnivad.vad import default_model_dir

        source_path = os.path.join(default_model_dir(), "vad.omnivad")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copyfile(source_path, target_path)
        print(f"Copied OmniVAD model to {target_path}")
        return target_path
    except Exception as exc:
        print(f"Warning: could not stage OmniVAD model in checkpoints: {exc}")
        return None


def is_qwen_omnivad_available() -> bool:
    """Return true when the required Qwen ASR package is importable."""
    return Qwen3ASRModel is not None and torch is not None


def _normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = language.strip()
    if not value:
        return None
    return _LANGUAGE_ALIASES.get(value.lower().replace("_", "-"), value)


def _torch_dtype() -> Any:
    if torch is None:
        return None
    name = QWEN_ASR_DTYPE.strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def _load_qwen_model() -> Any:
    global _ASR_MODEL, _ASR_MODEL_LOAD_PATH, _FORCED_ALIGNER_LOAD_PATH
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    if not is_qwen_omnivad_available():
        raise RuntimeError(
            "Qwen3-ASR is not installed. Current qwen-asr releases conflict with "
            "qwen-tts because they pin different exact transformers versions; install "
            "qwen-asr, omnivad, and litai in a separate environment for this pipeline."
        )

    dtype = _torch_dtype()
    forced_aligner = QWEN_ASR_FORCED_ALIGNER or None
    model_path = _resolve_hf_model_path(
        QWEN_ASR_MODEL,
        local_dir_override=QWEN_ASR_LOCAL_DIR,
        label="Qwen3-ASR",
    )
    _ASR_MODEL_LOAD_PATH = model_path
    forced_kwargs = None
    if forced_aligner:
        forced_aligner = _resolve_hf_model_path(
            forced_aligner,
            local_dir_override=QWEN_ASR_FORCED_ALIGNER_LOCAL_DIR,
            label="Qwen3 forced-aligner",
        )
        _FORCED_ALIGNER_LOAD_PATH = forced_aligner
        forced_kwargs = {"dtype": dtype, "device_map": QWEN_ASR_DEVICE}

    common_kwargs = {
        "forced_aligner": forced_aligner,
        "forced_aligner_kwargs": forced_kwargs,
        "max_inference_batch_size": QWEN_ASR_MAX_BATCH_SIZE,
        "max_new_tokens": QWEN_ASR_MAX_NEW_TOKENS,
    }
    print(f"Loading Qwen3-ASR model: {model_path} ({QWEN_ASR_BACKEND})")
    if QWEN_ASR_BACKEND == "vllm":
        raise RuntimeError(
            "In-process Qwen3-ASR vLLM is disabled for this app. "
            "This repository pins vllm==0.10.2 for IndexTTS2, while Qwen3-ASR "
            "vLLM support currently expects qwen-asr[vllm]/newer vLLM wheels. "
            "Use QWEN_ASR_BACKEND=transformers in this environment, or run "
            "Qwen3-ASR vLLM in a separate service/environment."
        )
    else:
        _ASR_MODEL = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=QWEN_ASR_DEVICE,
            **common_kwargs,
        )
    return _ASR_MODEL


def _release_memory() -> None:
    gc.collect()
    if torch is not None and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()


def _duration_seconds(audio_path: str) -> float:
    if AudioSegment is None:
        return 0.0
    try:
        return len(AudioSegment.from_file(audio_path)) / 1000.0
    except Exception:
        return 0.0


def _suffix_for_mime(mime_type: Optional[str]) -> str:
    value = (mime_type or "").lower()
    if "mpeg" in value or "mp3" in value:
        return ".mp3"
    if "flac" in value:
        return ".flac"
    if "ogg" in value:
        return ".ogg"
    if "webm" in value:
        return ".webm"
    if "mp4" in value or "m4a" in value or "aac" in value:
        return ".m4a"
    return ".wav"


def _write_temp_audio(audio_bytes: bytes, mime_type: Optional[str]) -> str:
    fd, path = tempfile.mkstemp(suffix=_suffix_for_mime(mime_type))
    with os.fdopen(fd, "wb") as handle:
        handle.write(audio_bytes)
    return path


def _write_omnivad_input(audio_path: str) -> Optional[str]:
    """Create a temporary 16kHz mono WAV for OmniVAD."""
    if AudioSegment is None:
        return None

    fd, vad_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        (
            AudioSegment.from_file(audio_path)
            .set_frame_rate(16_000)
            .set_channels(1)
            .set_sample_width(2)
            .export(vad_path, format="wav")
        )
        return vad_path
    except Exception as exc:
        try:
            os.remove(vad_path)
        except OSError:
            pass
        print(f"Warning: failed to prepare 16kHz OmniVAD input: {exc}")
        return None


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def _cache_key(
    audio_hash: str,
    *,
    source_language: Optional[str],
    dest_language: str,
    enable_translation: bool,
    translation_llm_model: Optional[str],
    input_mime_type: Optional[str],
) -> str:
    raw = "|".join(
        [
            f"v{QWEN_OMNIVAD_CACHE_VERSION}",
            audio_hash,
            f"mime={input_mime_type or ''}",
            f"src={source_language or 'auto'}",
            f"dst={dest_language}",
            f"translate={enable_translation}",
            f"model={QWEN_ASR_MODEL}",
            f"backend={QWEN_ASR_BACKEND}",
            f"aligner={QWEN_ASR_FORCED_ALIGNER or 'none'}",
            f"llm={translation_llm_model or 'default'}",
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_key: str) -> str:
    os.makedirs(QWEN_OMNIVAD_CACHE_DIR, exist_ok=True)
    return os.path.join(QWEN_OMNIVAD_CACHE_DIR, f"qla_{cache_key}.json")


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
        print(f"Warning: Qwen/OmniVAD cache write failed: {exc}")
        return None


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？；;])\s+|(?<=[。！？；;])", text)
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return cleaned or [text]


def _time_stamps_to_timeline(time_stamps: Optional[Sequence[Any]], fallback_text: str) -> List[TimelineItem]:
    if not time_stamps:
        return []

    items: List[TimelineItem] = []
    current_words: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def flush() -> None:
        nonlocal current_words, current_start, current_end
        text = "".join(current_words).strip()
        if text and current_start is not None and current_end is not None and current_end > current_start:
            items.append(TimelineItem(start=current_start, end=current_end, text=text))
        current_words = []
        current_start = None
        current_end = None

    for stamp in time_stamps:
        word = str(getattr(stamp, "text", "") or "").strip()
        start = getattr(stamp, "start_time", None)
        end = getattr(stamp, "end_time", None)
        if not word or start is None or end is None:
            continue
        start_f = float(start)
        end_f = float(end)
        if current_start is None:
            current_start = start_f
        current_words.append(word)
        current_end = end_f
        if re.search(r"[.!?。！？；;]$", word) or len("".join(current_words)) >= 80:
            flush()
    flush()

    if items:
        return items
    return _proportional_timeline(fallback_text, 0.0, max(float(getattr(time_stamps[-1], "end_time", 0.0) or 0.0), 0.0))


def _proportional_timeline(text: str, start: float, end: float) -> List[TimelineItem]:
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
        items.append(TimelineItem(start=cursor, end=max(cursor + 0.1, next_end), text=sentence))
        cursor = next_end
    return items


def _text_units_for_speech_spans(text: str, span_count: int) -> List[str]:
    """Return ordered text units granular enough to distribute over VAD spans."""
    sentences = _split_sentences(text)
    if span_count <= 1 or len(sentences) >= span_count:
        return sentences

    words = re.findall(r"\S+", text or "")
    if len(words) >= span_count:
        return words

    return sentences


def _take_units_for_span(
    units: Sequence[str],
    unit_idx: int,
    *,
    target_weight: float,
    remaining_spans: int,
) -> Tuple[List[str], int]:
    """Take contiguous text units for one VAD span while preserving order."""
    remaining_units = len(units) - unit_idx
    if remaining_units <= 0:
        return [], unit_idx

    if remaining_spans <= 1:
        return list(units[unit_idx:]), len(units)

    if remaining_units <= remaining_spans:
        return [units[unit_idx]], unit_idx + 1

    max_take = max(1, remaining_units - (remaining_spans - 1))
    taken: List[str] = []
    taken_weight = 0.0
    while unit_idx < len(units) and len(taken) < max_take:
        unit = units[unit_idx]
        taken.append(unit)
        taken_weight += max(1, len(unit))
        unit_idx += 1
        if taken_weight >= target_weight:
            break

    return taken, unit_idx


def _append_units_inside_speech_span(
    items: List[TimelineItem],
    *,
    span_start: float,
    span_end: float,
    units: Sequence[str],
) -> None:
    """Create one or more timeline items inside a single VAD speech span."""
    cleaned_units = [unit.strip() for unit in units if unit and unit.strip()]
    if not cleaned_units or span_end <= span_start:
        return

    if len(cleaned_units) == 1:
        items.append(
            TimelineItem(
                start=span_start,
                end=max(span_start + 0.1, span_end),
                text=cleaned_units[0],
            )
        )
        return

    weights = [max(1, len(unit)) for unit in cleaned_units]
    total_weight = float(sum(weights))
    duration = max(0.1, span_end - span_start)
    cursor = span_start
    for idx, unit in enumerate(cleaned_units):
        if idx == len(cleaned_units) - 1:
            unit_end = span_end
        else:
            unit_end = cursor + duration * (weights[idx] / total_weight)
        items.append(
            TimelineItem(
                start=cursor,
                end=max(cursor + 0.1, min(unit_end, span_end)),
                text=unit,
            )
        )
        cursor = min(unit_end, span_end)


def _timeline_from_omnivad(audio_path: str, text: str, duration: float) -> List[TimelineItem]:
    global _OMNIVAD_MODEL_LOAD_PATH
    if not QWEN_OMNIVAD_USE_OMNIVAD or OmniVAD is None:
        return []
    vad_audio_path = _write_omnivad_input(audio_path)
    if not vad_audio_path:
        return []
    try:
        _OMNIVAD_MODEL_LOAD_PATH = _resolve_omnivad_model_path()
        vad = OmniVAD(model_path=_OMNIVAD_MODEL_LOAD_PATH) if _OMNIVAD_MODEL_LOAD_PATH else OmniVAD()
        result = vad.detect(
            vad_audio_path,
            chunk_seconds=QWEN_OMNIVAD_CHUNK_SECONDS,
            overlap_seconds=QWEN_OMNIVAD_OVERLAP_SECONDS,
        )
    except Exception as exc:
        print(f"Warning: OmniVAD detection failed: {exc}")
        return []
    finally:
        try:
            os.remove(vad_audio_path)
        except OSError:
            pass

    timestamps = result.get("timestamps") if isinstance(result, dict) else None
    if not timestamps:
        return []

    raw_speech_spans = [(float(s), float(e)) for s, e in timestamps if float(e) > float(s)]
    if not raw_speech_spans:
        return []

    # --- NEW: Merge adjacent spans with tiny silence gaps ---
    speech_spans = []
    for start, end in raw_speech_spans:
        if not speech_spans:
            speech_spans.append((start, end))
        else:
            last_start, last_end = speech_spans[-1]
            # If the gap between the last span and current span is smaller than our threshold, merge them
            if start - last_end <= QWEN_OMNIVAD_MERGE_GAP_SECONDS:
                speech_spans[-1] = (last_start, max(last_end, end))
            else:
                speech_spans.append((start, end))
    # -------------------------------------------------------

    gap_count = sum(
        1
        for idx in range(1, len(speech_spans))
        if speech_spans[idx][0] > speech_spans[idx - 1][1] + 0.05
    )
    print(
        f"OmniVAD detected {len(speech_spans)} speech span(s) "
        f"(merged gaps <= {QWEN_OMNIVAD_MERGE_GAP_SECONDS}s) "
        f"with {gap_count} preserved gap(s)."
    )

    units = _text_units_for_speech_spans(text, len(speech_spans))
    if not units:
        return []

    speech_duration = sum(end - start for start, end in speech_spans)
    if speech_duration <= 0:
        return []

    items: List[TimelineItem] = []
    unit_idx = 0
    total_unit_weight = float(sum(max(1, len(unit)) for unit in units))
    for span_idx, (span_start, span_end) in enumerate(speech_spans):
        if unit_idx >= len(units):
            break

        remaining_spans = len(speech_spans) - span_idx
        span_weight = total_unit_weight * ((span_end - span_start) / speech_duration)
        span_units, unit_idx = _take_units_for_span(
            units,
            unit_idx,
            target_weight=span_weight,
            remaining_spans=remaining_spans,
        )
        _append_units_inside_speech_span(
            items,
            span_start=max(0.0, span_start),
            span_end=max(span_start + 0.1, min(span_end, duration or span_end)),
            units=span_units,
        )

    return items


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
                "Detected by Qwen3-ASR + OmniVAD pipeline. Review for gender, age, and tone."
                if speaker != "speaker1"
                else "Default speaker for Qwen3-ASR + OmniVAD pipeline."
            ),
        }
        for speaker in (speakers or ["speaker1"])
    ]


def _items_to_segments(items: Sequence[TimelineItem]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for item in items:
        text = (item.text or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": _format_timestamp(item.start),
                "end": _format_timestamp(max(item.end, item.start + 0.1)),
                "speaker": item.speaker or "speaker1",
                "source_text": text,
                "translated_text": "",
            }
        )
    return segments


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
    print(f"PASS C [Qwen3-ASR + OmniVAD]: LLM translation -> {dest_language}")
    print("=" * 60)

    if LitaiLLM is None or _translate_texts_adaptively is None:
        print(
            "Warning: litai translation helpers are unavailable; "
            "leaving translated_text empty."
        )
        return

    total_segments = len(segments)
    translation_jobs: List[Dict[str, Any]] = []

    for start in range(0, total_segments, translation_batch_size):
        chunk = segments[start: start + translation_batch_size]
        non_empty = [seg for seg in chunk if (seg.get("source_text") or "").strip()]
        source_texts = [seg["source_text"] for seg in non_empty]
        batch_label = f"Batch {start // translation_batch_size + 1}"

        if not source_texts:
            print(
                f"  {batch_label}: all segments empty, skipping."
            )
            continue

        translation_jobs.append({
            "label": batch_label,
            "start": start + 1,
            "end": min(start + translation_batch_size, total_segments),
            "segments": non_empty,
            "source_texts": source_texts,
        })

    if not translation_jobs:
        return

    worker_count = min(translation_max_workers, len(translation_jobs))
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
        llm = LitaiLLM(model=llm_model)
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
    else:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="qwen_omnivad_translate",
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


def _run_qwen_omnivad_pipeline_sync(
    audio_bytes: bytes,
    *,
    input_mime_type: Optional[str] = None,
    source_language: Optional[str] = None,
    dest_language: str = "English",
    enable_translation: bool = True,
    translation_llm_model: Optional[str] = None,
    translation_batch_size: int = QWEN_OMNIVAD_TRANSLATION_BATCH_SIZE,
    translation_max_workers: int = QWEN_OMNIVAD_TRANSLATION_MAX_WORKERS,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
    """Run Qwen3-ASR, optional Qwen timestamps/OmniVAD timing, optional LLM translation."""
    if not is_qwen_omnivad_available():
        raise RuntimeError(
            "Qwen3-ASR is not installed. Current qwen-asr releases conflict with "
            "qwen-tts because they pin different exact transformers versions; install "
            "qwen-asr, omnivad, and litai in a separate environment for this pipeline."
        )

    source_language = _normalize_language(source_language)
    translation_batch_size = max(1, int(translation_batch_size or 1))
    try:
        translation_max_workers = int(translation_max_workers or 1)
    except (TypeError, ValueError):
        translation_max_workers = QWEN_OMNIVAD_TRANSLATION_MAX_WORKERS
    translation_max_workers = max(1, min(10, translation_max_workers))
    llm_model = translation_llm_model or QWEN_OMNIVAD_TRANSLATION_LLM
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    cache_key = _cache_key(
        audio_hash,
        source_language=source_language,
        dest_language=dest_language,
        enable_translation=enable_translation,
        translation_llm_model=llm_model,
        input_mime_type=input_mime_type,
    )
    cache_info: Dict[str, Any] = {
        "audio_md5": audio_hash,
        "hit": False,
        "force_refresh": bool(force_refresh),
        "pipeline": "qwen_omnivad",
        "qwen_model": QWEN_ASR_MODEL,
        "qwen_backend": QWEN_ASR_BACKEND,
        "qwen_model_dir": QWEN_OMNIVAD_MODEL_DIR,
        "omnivad_enabled": bool(QWEN_OMNIVAD_USE_OMNIVAD and OmniVAD is not None),
        "translation_batch_size": translation_batch_size,
        "translation_max_workers": translation_max_workers,
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

    audio_path = _write_temp_audio(audio_bytes, input_mime_type)
    try:
        duration = _duration_seconds(audio_path)
        asr = _load_qwen_model()
        cache_info["qwen_model_path"] = _ASR_MODEL_LOAD_PATH
        if _FORCED_ALIGNER_LOAD_PATH:
            cache_info["forced_aligner_path"] = _FORCED_ALIGNER_LOAD_PATH
        want_qwen_timestamps = bool(QWEN_ASR_FORCED_ALIGNER)
        print("Running Qwen3-ASR transcription...")
        results = asr.transcribe(
            audio=audio_path,
            language=source_language,
            return_time_stamps=want_qwen_timestamps,
        )
        if not results:
            raise RuntimeError("Qwen3-ASR returned no transcription results.")
        result = results[0]
        detected_language = getattr(result, "language", None) or source_language
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Qwen3-ASR returned an empty transcription.")

        timeline_source = "none"
        items = _timeline_from_omnivad(audio_path, text, duration)
        if items:
            timeline_source = "omnivad"

        if not items:
            if QWEN_OMNIVAD_REQUIRE_VAD_TIMELINE and QWEN_OMNIVAD_USE_OMNIVAD:
                raise RuntimeError(
                    "OmniVAD did not return usable speech timestamps, so the "
                    "Qwen/OmniVAD pipeline cannot preserve silence gaps. "
                    "Check the OmniVAD model path/logs, or set "
                    "QWEN_OMNIVAD_REQUIRE_VAD_TIMELINE=0 to allow the "
                    "continuous Qwen/proportional fallback."
                )

            items = _time_stamps_to_timeline(getattr(result, "time_stamps", None), text)
            if items:
                timeline_source = "qwen_forced_aligner"

        if not items:
            items = _proportional_timeline(text, 0.0, duration or 0.1)
            timeline_source = "proportional"

        cache_info["timeline_source"] = timeline_source
        cache_info["source_language"] = detected_language
        if _OMNIVAD_MODEL_LOAD_PATH:
            cache_info["omnivad_model_path"] = _OMNIVAD_MODEL_LOAD_PATH

        segments = _items_to_segments(items)
        speaker_profiles = _speaker_profiles(items)
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
            "pipeline": "qwen_omnivad",
            "qwen_model": QWEN_ASR_MODEL,
            "qwen_backend": QWEN_ASR_BACKEND,
            "qwen_model_path": _ASR_MODEL_LOAD_PATH,
            "forced_aligner_path": _FORCED_ALIGNER_LOAD_PATH,
            "omnivad_model_path": _OMNIVAD_MODEL_LOAD_PATH,
            "source_language": detected_language,
            "timeline_source": timeline_source,
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
            "timeline_source": timeline_source,
        }
        cache_path = _write_cache(cache_key, cache_record)
        if cache_path:
            cache_info["cache_file"] = os.path.basename(cache_path)
        return segments, speaker_profiles, raw_text, cache_info
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass
        _release_memory()
