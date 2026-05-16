"""
OmniVAD splits audio into speech spans; each span is transcribed with Qwen3-ASR, then texts
are translated in batch (`_translate_segments`) like Gemini/WhisperX.

Pipeline return contract (`segments`, speaker profiles, raw JSON text, cache_info) keeps
sessions, subtitle export, and IndexTTS unchanged in ``fastapi_webui_v2.py``.
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
QWEN_OMNIVAD_CACHE_VERSION = 5

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
QWEN_OMNIVAD_MERGE_GAP_SECONDS = float(os.getenv("QWEN_OMNIVAD_MERGE_GAP_SECONDS", "0.001"))

try:
    _raw_segment_workers = int(os.getenv("QWEN_OMNIVAD_ASR_SEGMENT_WORKERS", "1"))
except ValueError:
    _raw_segment_workers = 1
QWEN_OMNIVAD_ASR_SEGMENT_WORKERS = max(1, min(16, _raw_segment_workers))

try:
    QWEN_OMNIVAD_MIN_SEGMENT_SECONDS = float(
        os.getenv("QWEN_OMNIVAD_MIN_SEGMENT_SECONDS", "0.05")
    )
except ValueError:
    QWEN_OMNIVAD_MIN_SEGMENT_SECONDS = 0.05
QWEN_OMNIVAD_MIN_SEGMENT_SECONDS = max(0.02, QWEN_OMNIVAD_MIN_SEGMENT_SECONDS)


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
            f"vad_slice_pipeline=1",
            f"vad_asr_workers={QWEN_OMNIVAD_ASR_SEGMENT_WORKERS}",
            f"vad_min_seg={QWEN_OMNIVAD_MIN_SEGMENT_SECONDS:.4f}",
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


def _merge_nearby_speech_spans(
    spans: Sequence[Tuple[float, float]],
    *,
    max_gap_seconds: float,
) -> List[Tuple[float, float]]:
    """
    Merge adjacent speech spans when the silence gap between them is small.

    This reduces subtitle over-segmentation caused by VAD producing many
    neighboring spans separated by short natural pauses.
    """
    cleaned = sorted(
        (max(0.0, float(start)), float(end))
        for start, end in spans
        if float(end) > float(start)
    )
    if not cleaned or max_gap_seconds <= 0:
        return cleaned

    merged: List[Tuple[float, float]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap <= max_gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _detect_omnivad_speech_spans(
    audio_path: str,
    *,
    duration_seconds: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Run OmniVAD; return merged speech intervals and raw intervals in seconds."""
    global _OMNIVAD_MODEL_LOAD_PATH

    if not QWEN_OMNIVAD_USE_OMNIVAD or OmniVAD is None:
        return [], []

    vad_audio_path = _write_omnivad_input(audio_path)
    if not vad_audio_path:
        return [], []

    raw_spans: List[Tuple[float, float]] = []
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
        return [], []
    finally:
        try:
            os.remove(vad_audio_path)
        except OSError:
            pass

    timestamps = result.get("timestamps") if isinstance(result, dict) else None
    if not timestamps:
        return [], []

    raw_spans = [
        (float(s), float(e))
        for s, e in timestamps
        if float(e) > float(s)
    ]
    if not raw_spans:
        return [], []

    if duration_seconds > 0:
        clipped = []
        for s, e in raw_spans:
            cs = max(0.0, min(float(s), duration_seconds))
            ce = max(0.0, min(float(e), duration_seconds))
            if ce > cs:
                clipped.append((cs, ce))
        raw_spans = clipped
    else:
        raw_spans.sort(key=lambda span: span[0])

    if not raw_spans:
        return [], []

    merged = _merge_nearby_speech_spans(
        raw_spans,
        max_gap_seconds=QWEN_OMNIVAD_MERGE_GAP_SECONDS,
    )
    preserved_gaps = sum(
        1
        for idx in range(1, len(merged))
        if merged[idx][0] > merged[idx - 1][1] + 0.05
    )
    print(
        f"OmniVAD → {len(raw_spans)} speech span(s) after clipping → "
        f"{len(merged)} merged span(s) (gap≤{QWEN_OMNIVAD_MERGE_GAP_SECONDS:.2f}s); "
        f"{preserved_gaps} silence gap(s) kept."
    )
    print("OmniVAD Raw Spans:")
    for rs_idx, (rs_start, rs_end) in enumerate(raw_spans, start=1):
        print(f"  Raw Span {rs_idx}: [{rs_start:.3f}s → {rs_end:.3f}s]")
    return merged, raw_spans


def _transcribe_vad_slices(
    asr: Any,
    audio_path: str,
    speech_spans: Sequence[Tuple[float, float]],
    duration_seconds: float,
    *,
    language: Optional[str],
) -> Tuple[List[TimelineItem], str, Optional[str]]:
    """
    Crop each OmniVAD speech interval and run Qwen3-ASR on that clip alone.
    Segment start/end match VAD boundaries on the timeline.
    """
    if AudioSegment is None:
        raise RuntimeError("pydub is required for VAD slicing; install pydub.")

    dur_sec = duration_seconds if duration_seconds > 0 else _duration_seconds(audio_path)
    full = AudioSegment.from_file(audio_path)
    dur_ms = max(1, len(full))

    min_sec = QWEN_OMNIVAD_MIN_SEGMENT_SECONDS
    min_ms = max(1, int(round(min_sec * 1000.0)))

    total = len(speech_spans)
    batch_size = QWEN_ASR_MAX_BATCH_SIZE
    print(f"Running Qwen3-ASR on {total} VAD speech segment(s) using batch inference (batch_size={batch_size})...")

    VadAsrClipJob = Tuple[int, int, float, float, str]
    jobs: List[VadAsrClipJob] = []
    order_idx = 0
    for disp_idx, (start_s, end_s) in enumerate(speech_spans, start=1):
        ts = max(0.0, float(start_s))
        te = float(end_s)
        if dur_sec > 0:
            ts = min(ts, dur_sec)
            te = min(te, dur_sec)
        if te <= ts:
            print(f"  Segment {disp_idx}/{total}: skipping invalid span {(start_s, end_s)}.")
            continue
        if te - ts < min_sec:
            print(
                f"  Segment {disp_idx}/{total}: skipping span shorter than "
                f"{min_sec:.3f}s (actual {te - ts:.3f}s)."
            )
            continue

        start_ms = int(ts * 1000.0)
        end_ms = int(te * 1000.0)
        start_ms = max(0, min(start_ms, dur_ms - 1))
        end_ms = max(start_ms + min_ms, min(end_ms, dur_ms))

        chunk = full[start_ms:end_ms]
        fd, clip_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        chunk.export(clip_path, format="wav")

        ts_out = ts
        te_out = te
        if dur_sec <= 0:
            ts_out = start_ms / 1000.0
            te_out = min(end_ms / 1000.0, dur_ms / 1000.0)
        te_out = max(ts_out + min_sec, te_out)

        jobs.append((order_idx, disp_idx, ts_out, te_out, clip_path))
        order_idx += 1

    if not jobs:
        return [], "", None

    results_list = []
    want_timestamps = bool(QWEN_ASR_FORCED_ALIGNER)

    for i in range(0, total, batch_size):
        batch_jobs = jobs[i : i + batch_size]
        if not batch_jobs:
            continue
            
        batch_paths = [job[4] for job in batch_jobs]
        
        try:
            batch_results = asr.transcribe(
                audio=batch_paths,
                language=language,
                return_time_stamps=want_timestamps,
            )
        except Exception as exc:
            print(f"  Batch {i//batch_size + 1} transcription failed: {exc}")
            for job in batch_jobs:
                results_list.append((job[0], [], "", None))
            continue
            
        if not batch_results or len(batch_results) != len(batch_jobs):
            print(f"  Batch {i//batch_size + 1} returned unexpected results length.")
            for job in batch_jobs:
                results_list.append((job[0], [], "", None))
            continue
            
        for job, clip_result in zip(batch_jobs, batch_results):
            oi, disp_idx, ts_out, te_out, clip_path = job
            seg_text = (getattr(clip_result, "text", "") or "").strip()
            lg = getattr(clip_result, "language", None)
            lang_guess: Optional[str] = (
                lg.strip() if isinstance(lg, str) and lg.strip() else None
            )
            
            if not seg_text:
                print(f"  Segment {disp_idx}/{total}: empty transcript (skipped).")
                results_list.append((oi, [], "", lang_guess))
                continue
                
            preview = seg_text.replace("\n", " ")
            cut = preview if len(preview) <= 80 else preview[:77] + "..."
            print(f"  Segment {disp_idx}/{total}: [{ts_out:.2f}s → {te_out:.2f}s] {cut}")
            
            clip_items = [TimelineItem(start=ts_out, end=te_out, text=seg_text)]
            results_list.append((oi, clip_items, seg_text, lang_guess))
            
    # Cleanup audio clips
    for job in jobs:
        try:
            os.remove(job[4])
        except OSError:
            pass

    results_list.sort(key=lambda row: row[0])
    items: List[TimelineItem] = []
    fragments: List[str] = []
    detected_language: Optional[str] = None
    for _, items_list, frag, lang_guess in results_list:
        if lang_guess:
            detected_language = lang_guess
        if items_list and frag:
            items.extend(items_list)
            fragments.append(frag)

    combined_text = " ".join(fragments).strip()
    return items, combined_text, detected_language


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
                "end": _format_timestamp(max(item.end, item.start + 0.05)),
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


def translate_audio(
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
    """
    OmniVAD splits the audio → Qwen3-ASR on each speech span → `_translate_segments` in batch.

    Same return signature as Gemini/WhisperX helpers consumed by ``fastapi_webui_v2.py``.

    Optional env: ``QWEN_OMNIVAD_ASR_SEGMENT_WORKERS`` (default 1),
    ``QWEN_OMNIVAD_MIN_SEGMENT_SECONDS`` (default 0.05; floor 0.02s).
    """
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
        "vad_asr_segment_workers": QWEN_OMNIVAD_ASR_SEGMENT_WORKERS,
        "vad_min_segment_seconds": QWEN_OMNIVAD_MIN_SEGMENT_SECONDS,
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
        global _OMNIVAD_MODEL_LOAD_PATH
        _OMNIVAD_MODEL_LOAD_PATH = None
        duration = _duration_seconds(audio_path)
        asr = _load_qwen_model()
        cache_info["qwen_model_path"] = _ASR_MODEL_LOAD_PATH
        if _FORCED_ALIGNER_LOAD_PATH:
            cache_info["forced_aligner_path"] = _FORCED_ALIGNER_LOAD_PATH

        items: List[TimelineItem] = []
        timeline_source = "none"
        text = ""
        detected_language: Optional[str] = source_language

        vad_can_run = bool(QWEN_OMNIVAD_USE_OMNIVAD and OmniVAD is not None)
        speech_spans: List[Tuple[float, float]] = []
        raw_vad_spans: List[Tuple[float, float]] = []
        if vad_can_run:
            print("OmniVAD: detecting speech spans...")
            speech_spans, raw_vad_spans = _detect_omnivad_speech_spans(
                audio_path,
                duration_seconds=duration,
            )

        used_vad_asr_branch = False
        if speech_spans:
            items, agg_text, det_lang = _transcribe_vad_slices(
                asr,
                audio_path,
                speech_spans,
                duration,
                language=source_language,
            )
            if items:
                used_vad_asr_branch = True
                text = agg_text
                timeline_source = "vad_segment_asr"
                if isinstance(det_lang, str) and det_lang.strip():
                    raw = det_lang.strip()
                    detected_language = _normalize_language(raw) or raw
            else:
                print(
                    "Warning: OmniVAD returned speech spans but segment-wise ASR produced "
                    "no text; falling back to full-file Qwen3-ASR."
                )

        if not used_vad_asr_branch:
            if vad_can_run and QWEN_OMNIVAD_REQUIRE_VAD_TIMELINE and not speech_spans:
                raise RuntimeError(
                    "OmniVAD did not return usable speech timestamps, so the pipeline "
                    "cannot split audio for segmented transcription. "
                    "Check the OmniVAD model path/logs, or set "
                    "QWEN_OMNIVAD_REQUIRE_VAD_TIMELINE=0 to transcribe the full clip."
                )

            want_qwen_timestamps = bool(QWEN_ASR_FORCED_ALIGNER)
            print("Running Qwen3-ASR transcription (full audio)...")
            results = asr.transcribe(
                audio=audio_path,
                language=source_language,
                return_time_stamps=want_qwen_timestamps,
            )
            if not results:
                raise RuntimeError("Qwen3-ASR returned no transcription results.")
            result = results[0]
            raw_lang = getattr(result, "language", None)
            if isinstance(raw_lang, str) and raw_lang.strip():
                value = raw_lang.strip()
                detected_language = _normalize_language(value) or value
            text = (getattr(result, "text", "") or "").strip()
            if not text:
                raise RuntimeError("Qwen3-ASR returned an empty transcription.")

            aligner_items: List[TimelineItem] = []
            if want_qwen_timestamps:
                aligner_items = _time_stamps_to_timeline(
                    getattr(result, "time_stamps", None),
                    text,
                )

            if aligner_items:
                items = aligner_items
                timeline_source = "qwen_forced_aligner"
            else:
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
            "raw_vad_spans": raw_vad_spans,
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


_run_qwen_omnivad_pipeline_sync = translate_audio  # Historical name kept for fastapi_webui_v2 imports.
