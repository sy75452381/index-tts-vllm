"""
WhisperX-based local transcription & translation pipeline.

A drop-in alternative to the Gemini-based pipeline that runs entirely locally
using WhisperX for speech recognition + diarization and an external LLM for
translation.

The three-pass pipeline:
  Pass A — English proxy for timestamps & speaker segmentation
  Pass B — Source-language re-transcription per segment
  Pass C — LLM batch translation (source_text → dest_language)

Returns data in the same (segments, speaker_profiles, raw_text, cache_info)
format that _gemini_transcribe_translate produces, so the rest of the
translate workflow works unchanged.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import tempfile
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports — gracefully degrade when not installed
try:
    import whisperx
    from whisperx.diarize import DiarizationPipeline
    from whisperx.audio import SAMPLE_RATE
except ImportError:
    whisperx = None  # type: ignore[assignment]
    DiarizationPipeline = None  # type: ignore[assignment,misc]
    SAMPLE_RATE = 16_000

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from litai import LLM as LitaiLLM  # type: ignore[import]
except ImportError:
    LitaiLLM = None

try:
    import json_repair  # type: ignore[import]
except ImportError:
    json_repair = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration defaults (can be overridden via environment variables)
# ---------------------------------------------------------------------------
WHISPERX_MODEL_SIZE = os.getenv("WHISPERX_MODEL_SIZE", "large-v3")
WHISPERX_DEVICE = os.getenv("WHISPERX_DEVICE", "cuda")
WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "float16")
WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "4"))
WHISPERX_HF_TOKEN = os.getenv(
    "WHISPERX_HF_TOKEN",
    os.getenv("HF_TOKEN", ""),
)
WHISPERX_TRANSLATION_LLM = os.getenv(
    "WHISPERX_TRANSLATION_LLM",
    "lightning-ai/DeepSeek-V3.1",
)
WHISPERX_TRANSLATION_BATCH_SIZE = int(
    os.getenv("WHISPERX_TRANSLATION_BATCH_SIZE", "30")
)

# ---------------------------------------------------------------------------
# Model storage — redirect all downloads into checkpoints/whisperx/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPERX_MODEL_DIR = os.path.join(_SCRIPT_DIR, "checkpoints", "whisperx")
WHISPERX_ASR_DOWNLOAD_ROOT = os.path.join(WHISPERX_MODEL_DIR, "faster-whisper")
WHISPERX_ALIGN_MODEL_DIR = os.path.join(WHISPERX_MODEL_DIR, "align")
WHISPERX_PYANNOTE_CACHE = os.path.join(WHISPERX_MODEL_DIR, "pyannote")

# Ensure directories exist
for _d in (WHISPERX_MODEL_DIR, WHISPERX_ASR_DOWNLOAD_ROOT,
           WHISPERX_ALIGN_MODEL_DIR, WHISPERX_PYANNOTE_CACHE):
    os.makedirs(_d, exist_ok=True)

# Set env vars *before* any model loading as a fallback for any
# sub-library that doesn't accept an explicit cache_dir parameter.
os.environ.setdefault("HF_HOME", WHISPERX_MODEL_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(WHISPERX_MODEL_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", WHISPERX_MODEL_DIR)
os.environ.setdefault("PYANNOTE_CACHE", WHISPERX_PYANNOTE_CACHE)

print(f"📂 WhisperX model cache directory: {WHISPERX_MODEL_DIR}")

# Cache directory for WhisperX results
WHISPERX_CACHE_DIR = os.path.join(
    _SCRIPT_DIR,
    "whisperx_cache",
)
WHISPERX_CACHE_VERSION = 1


def is_whisperx_available() -> bool:
    """Check if WhisperX and its dependencies are installed."""
    return whisperx is not None and torch is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to mm:ss.xxx format."""
    if seconds is None:
        return "00:00.000"
    minutes, seconds_remainder = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds_remainder:06.3f}"


def _clean_json_response(response_text: str) -> str:
    """Strip markdown fences, normalise CJK quotes, and fix common LLM artefacts."""
    cleaned = response_text.strip()

    # Remove markdown code fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Normalise CJK quotation marks at structural positions
    cleaned = re.sub(r'(?<=[\[,])\s*「', ' "', cleaned)
    cleaned = re.sub(r'」\s*(?=[,\]])', '"', cleaned)
    cleaned = re.sub(r'(?<=[\[,])\s*『', ' "', cleaned)
    cleaned = re.sub(r'』\s*(?=[,\]])', '"', cleaned)

    # Replace any remaining CJK quotes inside the string
    cleaned = cleaned.replace('「', '"').replace('」', '"')
    cleaned = cleaned.replace('『', '"').replace('』', '"')

    return cleaned


def _parse_llm_json_array(
    raw_response: str,
    expected_count: int,
    batch_label: str = "",
) -> Optional[List[str]]:
    """Parse an LLM response as a JSON array of strings, with robust fallbacks.

    Returns the parsed list on success, or None on failure.
    Logs full diagnostics on every failure.
    """
    prefix = f"  [{batch_label}] " if batch_label else "  "

    if not raw_response or raw_response.strip() == "":
        print(f"{prefix}⚠️ LLM returned an empty response.")
        return None

    cleaned = _clean_json_response(raw_response)

    # --- Attempt 1: standard json.loads ---
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            print(f"{prefix}✅ Parsed with json.loads (len={len(result)})")
            return result
    except json.JSONDecodeError:
        pass  # fall through to repair

    # --- Attempt 2: try closing truncated arrays ---
    # LLMs often truncate mid-string; try closing the last string + array
    for suffix in ['"]', '"]', '"]', '"]\n']:
        try:
            result = json.loads(cleaned + suffix)
            if isinstance(result, list):
                print(
                    f"{prefix}✅ Parsed after closing truncated array "
                    f"(len={len(result)})"
                )
                return result
        except json.JSONDecodeError:
            continue

    # --- Attempt 3: json_repair ---
    if json_repair is not None:
        try:
            result = json_repair.loads(cleaned)
            if isinstance(result, list):
                print(
                    f"{prefix}✅ Parsed with json_repair (len={len(result)})"
                )
                return result
            else:
                print(
                    f"{prefix}⚠️ json_repair returned type "
                    f"{type(result).__name__}, expected list"
                )
        except Exception as e:
            print(f"{prefix}⚠️ json_repair also failed: {e}")
    else:
        print(f"{prefix}⚠️ json_repair not installed, cannot attempt repair")

    # --- All attempts failed — log full details ---
    print(f"{prefix}❌ All JSON parse attempts failed.")
    print(f"{prefix}   Expected {expected_count} items.")
    print(f"{prefix}   Raw response length: {len(raw_response)} chars")
    print(f"{prefix}   Cleaned length: {len(cleaned)} chars")
    print(f"{prefix}   --- BEGIN RAW RESPONSE ---")
    # Print full response for debugging (split into lines to avoid terminal issues)
    for line in raw_response.splitlines():
        print(f"{prefix}   {line}")
    print(f"{prefix}   --- END RAW RESPONSE ---")
    return None


def _convert_to_output_format(
    proxy_result: dict,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert WhisperX result dict to (segments, speaker_profiles) format.

    Each segment has: start, end, speaker, source_text, translated_text, proxy_english
    Speaker profiles have: id, description
    """
    original_segments = proxy_result.get("segments", [])
    speaker_profiles: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    speaker_map: Dict[str, str] = {}
    speaker_counter = 1

    for seg in original_segments:
        orig_speaker_id = seg.get("speaker", "UNKNOWN_SPEAKER")
        if orig_speaker_id not in speaker_map:
            new_speaker_id = f"speaker{speaker_counter}"
            speaker_map[orig_speaker_id] = new_speaker_id
            speaker_profiles.append({
                "id": new_speaker_id,
                "description": "Detected by WhisperX diarization. Review for gender, age, and tone.",
            })
            speaker_counter += 1

        segments.append({
            "start": _format_timestamp(seg.get("start", 0.0)),
            "end": _format_timestamp(seg.get("end", 0.0)),
            "speaker": speaker_map[orig_speaker_id],
            "proxy_english": seg.get("text", "").strip(),
            "source_text": seg.get("source_text", ""),
            "translated_text": "",
        })

    return segments, speaker_profiles


def _translate_batch_with_retry(
    llm: Any,
    source_texts: List[str],
    dest_language: str,
    batch_label: str,
    max_retries: int = 10,
) -> Optional[List[str]]:
    """Translate a batch of strings with retries."""
    prompt = (
        f"You are a precise professional translator. "
        f"Translate the following JSON array of strings into "
        f"{dest_language}. "
        f"CRITICAL RULES:\n"
        f"1. Maintain the exact same number of items and order.\n"
        f"2. Return ONLY a valid JSON array of strings.\n"
        f"3. Do not include any explanations, markdown code "
        f"blocks, or conversational filler.\n\n"
        f"Input Array:\n"
        f"{json.dumps(source_texts, ensure_ascii=False)}"
    )

    expected_count = len(source_texts)
    
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"  [{batch_label}] 🔄 Retry attempt {attempt}/{max_retries}...")
        
        try:
            response = llm.chat(prompt, max_tokens=50000)
            if not response or str(response).strip() == "":
                print(f"  [{batch_label}] ⚠️ LLM returned an empty response.")
                continue
                
            parsed = _parse_llm_json_array(str(response), expected_count, batch_label)
            if parsed is not None:
                if len(parsed) == expected_count:
                    return parsed
                else:
                    print(
                        f"  [{batch_label}] ⚠️ Length mismatch: got {len(parsed)}, "
                        f"expected {expected_count}"
                    )
        except Exception as e:
            print(f"  [{batch_label}] ⚠️ API Error: {e}")
            
    return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _whisperx_cache_key(
    audio_hash: str,
    *,
    source_language: Optional[str],
    dest_language: str,
    enable_translation: bool,
) -> str:
    """Build a deterministic cache key string."""
    parts = [
        f"v{WHISPERX_CACHE_VERSION}",
        audio_hash,
        f"src={source_language or 'auto'}",
        f"dst={dest_language}",
        f"translate={enable_translation}",
        f"model={WHISPERX_MODEL_SIZE}",
    ]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _whisperx_cache_path(cache_key: str) -> str:
    os.makedirs(WHISPERX_CACHE_DIR, exist_ok=True)
    return os.path.join(WHISPERX_CACHE_DIR, f"wx_{cache_key}.json")


def _load_whisperx_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    path = _whisperx_cache_path(cache_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_whisperx_cache(cache_key: str, record: Dict[str, Any]) -> Optional[str]:
    path = _whisperx_cache_path(cache_key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return path
    except Exception as exc:
        print(f"⚠️ WhisperX cache write failed: {exc}")
        return None


def _compute_audio_hash(audio_bytes: bytes) -> str:
    return hashlib.md5(audio_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Core pipeline (synchronous — runs in executor)
# ---------------------------------------------------------------------------


def _detect_source_language(audio: np.ndarray, device: str, compute_type: str) -> str:
    """Detect the real language of the audio using Whisper's built-in detector."""
    print("🔍 WhisperX: Auto-detecting source language from audio...")
    det_model = whisperx.load_model(
        WHISPERX_MODEL_SIZE, device, compute_type=compute_type,
        download_root=WHISPERX_ASR_DOWNLOAD_ROOT,
    )
    det_result = det_model.transcribe(audio, batch_size=1)
    detected = det_result.get("language", "en")
    del det_model
    gc.collect()
    if torch is not None:
        torch.cuda.empty_cache()
    print(f"  → Detected source language: {detected}")
    return detected


def _run_whisperx_pipeline_sync(
    audio_bytes: bytes,
    *,
    source_language: Optional[str] = None,
    dest_language: str = "English",
    enable_translation: bool = True,
    hf_token: Optional[str] = None,
    translation_llm_model: Optional[str] = None,
    translation_batch_size: int = WHISPERX_TRANSLATION_BATCH_SIZE,
    force_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
    """Run the full WhisperX pipeline synchronously.

    Returns: (segments, speaker_profiles, raw_json_text, cache_info)
    Same return signature as _gemini_transcribe_translate.
    """
    if not is_whisperx_available():
        raise RuntimeError(
            "WhisperX is not installed. Install with: "
            "pip install whisperx pyannote.audio"
        )

    hf_token = hf_token or WHISPERX_HF_TOKEN
    if not hf_token:
        raise RuntimeError(
            "HuggingFace token is required for WhisperX diarization. "
            "Set WHISPERX_HF_TOKEN or HF_TOKEN environment variable."
        )

    device = WHISPERX_DEVICE
    compute_type = WHISPERX_COMPUTE_TYPE
    batch_size = WHISPERX_BATCH_SIZE

    # Cache check
    audio_hash = _compute_audio_hash(audio_bytes)
    cache_key = _whisperx_cache_key(
        audio_hash,
        source_language=source_language,
        dest_language=dest_language,
        enable_translation=enable_translation,
    )
    cache_info: Dict[str, Any] = {
        "audio_md5": audio_hash,
        "hit": False,
        "force_refresh": False,
        "pipeline": "whisperx",
    }

    if force_refresh:
        cache_info["force_refresh"] = True
    else:
        cached = _load_whisperx_cache(cache_key)
        if cached and isinstance(cached.get("segments"), list):
            cache_info["hit"] = True
            cache_info["cache_file"] = os.path.basename(
                _whisperx_cache_path(cache_key)
            )
            cache_info["created_at"] = cached.get("created_at")
            print(
                f"♻️ WhisperX cache hit for audio md5={audio_hash}"
            )
            return (
                cached["segments"],
                cached.get("speaker_profiles") or [],
                cached.get("raw_text", ""),
                cache_info,
            )

    # Write audio bytes to temp file for WhisperX
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio = whisperx.load_audio(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ===================================================================
    # PASS A: English Proxy — Segmentation Scaffold
    # ===================================================================
    print("=" * 60)
    print("PASS A [WhisperX]: English proxy for timestamps & speaker segmentation")
    print("=" * 60)

    print("Loading Whisper model (English proxy)...")
    model = whisperx.load_model(
        WHISPERX_MODEL_SIZE, device, compute_type=compute_type, language="en",
        download_root=WHISPERX_ASR_DOWNLOAD_ROOT,
    )
    proxy_result = model.transcribe(audio, batch_size=batch_size)
    del model
    gc.collect()
    if torch is not None:
        torch.cuda.empty_cache()

    print("Aligning (English)...")
    model_a, metadata = whisperx.load_align_model(
        language_code=proxy_result["language"], device=device,
        model_dir=WHISPERX_ALIGN_MODEL_DIR,
    )
    proxy_result = whisperx.align(
        proxy_result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    del model_a
    gc.collect()
    if torch is not None:
        torch.cuda.empty_cache()

    print("Diarizing...")
    diarize_model = DiarizationPipeline(
        token=hf_token, device=device, cache_dir=WHISPERX_PYANNOTE_CACHE,
    )
    diarize_segments = diarize_model(audio)
    proxy_result = whisperx.assign_word_speakers(diarize_segments, proxy_result)
    del diarize_model
    gc.collect()
    if torch is not None:
        torch.cuda.empty_cache()

    print(
        f"  → Pass A produced {len(proxy_result['segments'])} "
        f"speaker-segmented segments."
    )

    # ===================================================================
    # Detect source language (if not specified)
    # ===================================================================
    if source_language is None:
        source_language = _detect_source_language(audio, device, compute_type)

    # ===================================================================
    # PASS B: Source-Language Re-transcription
    # ===================================================================
    if source_language == "en":
        print("=" * 60)
        print("PASS B [WhisperX]: Skipped (source language is English)")
        print("=" * 60)
        for seg in proxy_result["segments"]:
            seg["source_text"] = seg.get("text", "").strip()
    else:
        print("=" * 60)
        print(
            f"PASS B [WhisperX]: Re-transcribe each segment in "
            f"source language ({source_language})"
        )
        print("=" * 60)

        source_model = whisperx.load_model(
            WHISPERX_MODEL_SIZE,
            device,
            compute_type=compute_type,
            language=source_language,
            download_root=WHISPERX_ASR_DOWNLOAD_ROOT,
        )

        total = len(proxy_result["segments"])
        for idx, seg in enumerate(proxy_result["segments"]):
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)

            f1 = int(seg_start * SAMPLE_RATE)
            f2 = int(seg_end * SAMPLE_RATE)
            clip = audio[f1:f2]

            if len(clip) < int(0.3 * SAMPLE_RATE):
                seg["source_text"] = ""
                print(
                    f"  [{idx+1}/{total}] Skipped "
                    f"(too short: {seg_end - seg_start:.2f}s)"
                )
                continue

            clip_result = source_model.transcribe(clip, batch_size=1)
            source_text = "".join(
                s.get("text", "").strip()
                for s in clip_result.get("segments", [])
            )
            seg["source_text"] = source_text

            print(
                f"  [{idx+1}/{total}] {seg_start:.1f}s–{seg_end:.1f}s "
                f"speaker={seg.get('speaker', '?')} → "
                f"{source_text[:60]}{'…' if len(source_text) > 60 else ''}"
            )

        del source_model
        gc.collect()
        if torch is not None:
            torch.cuda.empty_cache()
        print("  → Pass B complete.")

    # ===================================================================
    # Convert to target schema
    # ===================================================================
    segments, speaker_profiles = _convert_to_output_format(proxy_result)

    # ===================================================================
    # PASS C: Batched LLM Translation
    # ===================================================================
    if enable_translation and dest_language:
        print("=" * 60)
        print(f"PASS C [WhisperX]: LLM translation → {dest_language}")
        print("=" * 60)

        llm_model = translation_llm_model or WHISPERX_TRANSLATION_LLM
        if LitaiLLM is None:
            print(
                "⚠️ litai package not installed — skipping LLM translation. "
                "Install with: pip install litai"
            )
        else:
            llm = LitaiLLM(model=llm_model)
            total_segments = len(segments)

            for i in range(0, total_segments, translation_batch_size):
                chunk = segments[i: i + translation_batch_size]
                source_texts = [
                    seg["source_text"] for seg in chunk if seg["source_text"]
                ]

                if not source_texts:
                    print(
                        f"  Batch {i // translation_batch_size + 1}: "
                        f"all segments empty, skipping."
                    )
                    continue

                batch_label = f"Batch {i // translation_batch_size + 1}"
                print(
                    f"\n  Translating {batch_label} "
                    f"(Segments {i + 1} to "
                    f"{min(i + translation_batch_size, total_segments)})..."
                )

                translated_texts = _translate_batch_with_retry(
                    llm=llm,
                    source_texts=source_texts,
                    dest_language=dest_language,
                    batch_label=batch_label,
                    max_retries=10,
                )

                if translated_texts:
                    non_empty = [seg for seg in chunk if seg["source_text"]]
                    for seg_ref, trans in zip(non_empty, translated_texts):
                        seg_ref["translated_text"] = trans
                    print(f"  [{batch_label}] ✅ Batch translated successfully.")
                else:
                    print(f"  [{batch_label}] ❌ Batch failed after retries.")

    # Build raw text for session storage
    raw_output = {
        "speakers": speaker_profiles,
        "segments": segments,
    }
    raw_text = json.dumps(raw_output, ensure_ascii=False, indent=2)

    # Write cache
    cache_record = {
        "version": WHISPERX_CACHE_VERSION,
        "created_at": time.time(),
        "audio_md5": audio_hash,
        "source_language": source_language,
        "dest_language": dest_language,
        "enable_translation": enable_translation,
        "model": WHISPERX_MODEL_SIZE,
        "segments": segments,
        "speaker_profiles": speaker_profiles,
        "raw_text": raw_text,
    }
    cache_path = _write_whisperx_cache(cache_key, cache_record)
    if cache_path:
        cache_info["cache_file"] = os.path.basename(cache_path)
        cache_info["stored"] = True
        print(f"💾 WhisperX cache stored for audio md5={audio_hash}")

    print("✅ WhisperX pipeline complete!")
    return segments, speaker_profiles, raw_text, cache_info
