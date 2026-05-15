"""Speaker-aware post-processing for WhisperX proxy segments.

The refiner is intentionally independent of WhisperX runtime objects. It takes
the plain dict returned after alignment + diarization and rebuilds cleaner
speaker-preserving windows from word timings.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RefinerConfig:
    """Tunable thresholds for proxy segment refinement."""

    version: str = "v1"
    turn_break_gap_seconds: float = 1.0
    min_speaker_purity: float = 0.90
    ideal_duration_seconds: float = 4.5
    max_duration_seconds: float = 9.0
    max_chars: int = 140
    min_duration_seconds: float = 1.2
    min_words: int = 2
    same_speaker_merge_gap_seconds: float = 0.6
    ultra_short_gap_seconds: float = 2.0
    pause_split_gap_seconds: float = 0.45


@dataclass(frozen=True)
class WordToken:
    text: str
    start: float
    end: float
    speaker: str
    segment_index: int
    word_index: int
    raw: Dict[str, Any]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SegmentDraft:
    tokens: List[WordToken]

    @property
    def start(self) -> float:
        return self.tokens[0].start

    @property
    def end(self) -> float:
        return self.tokens[-1].end

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def speaker(self) -> str:
        return self.tokens[0].speaker

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    @property
    def text(self) -> str:
        return _join_word_texts([token.text for token in self.tokens])

    @property
    def char_count(self) -> int:
        return len(self.text)

    def merged_with(self, other: "SegmentDraft") -> "SegmentDraft":
        return SegmentDraft(tokens=self.tokens + other.tokens)


_SENTENCE_PUNCTUATION = set(".!?。！？")
_CLAUSE_PUNCTUATION = set(",;:，；：、")
_NO_SPACE_BEFORE = set(".,!?;:%)]}，。！？；：、」』）】》…")
_NO_SPACE_AFTER = set("([{「『（【《")
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def refine_proxy_segments(
    proxy_result: dict,
    *,
    config: RefinerConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return refined proxy segments plus debug stats.

    If aligned word timings or word-level speakers are unavailable, the
    original segments are returned unchanged with ``fallback_used=True``.
    """

    raw_segments = proxy_result.get("segments", []) if isinstance(proxy_result, dict) else []
    if not isinstance(raw_segments, list):
        raw_segments = []

    stats = _new_stats(len(raw_segments))
    tokens, extraction_stats, fallback_reason = _extract_word_tokens(raw_segments)
    stats.update(extraction_stats)

    if fallback_reason:
        stats["fallback_used"] = True
        stats["fallback_reason"] = fallback_reason
        stats["refined_proxy_segments"] = len(raw_segments)
        return raw_segments, stats

    stats["speaker_impure_segments_fixed"] = _count_impure_raw_segments(
        raw_segments,
        config,
    )

    turns, turn_stats = _build_speaker_turns(tokens, config)
    stats["speaker_change_splits"] = turn_stats["speaker_change_splits"]
    stats["gap_splits"] = turn_stats["gap_splits"]
    stats["same_speaker_merges"] = turn_stats["same_speaker_segment_joins"]

    long_split_segments: List[SegmentDraft] = []
    for turn in turns:
        pieces = _split_long_turn(turn, config)
        stats["long_turn_splits"] += max(0, len(pieces) - 1)
        long_split_segments.extend(pieces)

    merged_segments, cleanup_merges = _merge_short_same_speaker_segments(
        long_split_segments,
        config,
    )
    stats["same_speaker_merges"] += cleanup_merges

    refined_segments = [
        _segment_to_dict(segment)
        for segment in merged_segments
        if segment.tokens and segment.start < segment.end
    ]
    if not refined_segments and raw_segments:
        stats["fallback_used"] = True
        stats["fallback_reason"] = "refinement_produced_no_segments"
        stats["refined_proxy_segments"] = len(raw_segments)
        return raw_segments, stats

    stats["refined_proxy_segments"] = len(refined_segments)
    return refined_segments, stats


def _new_stats(raw_count: int) -> Dict[str, Any]:
    return {
        "raw_proxy_segments": raw_count,
        "refined_proxy_segments": 0,
        "speaker_change_splits": 0,
        "same_speaker_merges": 0,
        "long_turn_splits": 0,
        "segments_without_words": 0,
        "speaker_impure_segments_fixed": 0,
        "gap_splits": 0,
        "fallback_used": False,
    }


def _extract_word_tokens(
    raw_segments: List[Dict[str, Any]],
) -> Tuple[List[WordToken], Dict[str, int], Optional[str]]:
    stats = {"segments_without_words": 0}
    tokens: List[WordToken] = []

    for segment_index, segment in enumerate(raw_segments):
        words = segment.get("words") if isinstance(segment, dict) else None
        if not words:
            stats["segments_without_words"] += 1
            if str(segment.get("text", "") if isinstance(segment, dict) else "").strip():
                return [], stats, f"segment_{segment_index}_without_words"
            continue
        if not isinstance(words, list):
            return [], stats, f"segment_{segment_index}_words_not_list"

        for word_index, word in enumerate(words):
            if not isinstance(word, dict):
                return [], stats, f"segment_{segment_index}_word_{word_index}_not_dict"

            text = str(word.get("word", word.get("text", ""))).strip()
            if not text:
                continue

            start = _safe_float(word.get("start"))
            end = _safe_float(word.get("end"))
            if start is None or end is None or end <= start:
                return [], stats, f"segment_{segment_index}_word_{word_index}_missing_timing"

            speaker = word.get("speaker")
            if speaker is None or str(speaker).strip() == "":
                return [], stats, f"segment_{segment_index}_word_{word_index}_missing_speaker"

            tokens.append(
                WordToken(
                    text=text,
                    start=start,
                    end=end,
                    speaker=str(speaker),
                    segment_index=segment_index,
                    word_index=word_index,
                    raw=dict(word),
                )
            )

    if not tokens and raw_segments:
        return [], stats, "no_aligned_word_tokens"

    tokens.sort(key=lambda item: (item.start, item.end, item.segment_index, item.word_index))
    return tokens, stats, None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_impure_raw_segments(
    raw_segments: List[Dict[str, Any]],
    config: RefinerConfig,
) -> int:
    impure_count = 0
    for segment in raw_segments:
        words = segment.get("words") if isinstance(segment, dict) else None
        if not isinstance(words, list):
            continue

        durations: Dict[str, float] = {}
        for word in words:
            if not isinstance(word, dict) or not word.get("speaker"):
                continue
            start = _safe_float(word.get("start"))
            end = _safe_float(word.get("end"))
            if start is None or end is None or end <= start:
                continue
            speaker = str(word["speaker"])
            durations[speaker] = durations.get(speaker, 0.0) + (end - start)

        total = sum(durations.values())
        if total <= 0.0 or len(durations) < 2:
            continue
        purity = max(durations.values()) / total
        if purity < config.min_speaker_purity:
            impure_count += 1
    return impure_count


def _build_speaker_turns(
    tokens: List[WordToken],
    config: RefinerConfig,
) -> Tuple[List[SegmentDraft], Dict[str, int]]:
    if not tokens:
        return [], {
            "speaker_change_splits": 0,
            "gap_splits": 0,
            "same_speaker_segment_joins": 0,
        }

    stats = {
        "speaker_change_splits": 0,
        "gap_splits": 0,
        "same_speaker_segment_joins": 0,
    }
    turns: List[SegmentDraft] = []
    active: List[WordToken] = [tokens[0]]

    for token in tokens[1:]:
        previous = active[-1]
        gap = token.start - previous.end
        split = False

        if token.speaker != previous.speaker:
            stats["speaker_change_splits"] += 1
            split = True
        elif gap > config.turn_break_gap_seconds:
            stats["gap_splits"] += 1
            split = True
        elif token.segment_index != previous.segment_index:
            stats["same_speaker_segment_joins"] += 1

        if split:
            turns.append(SegmentDraft(tokens=active))
            active = [token]
        else:
            active.append(token)

    turns.append(SegmentDraft(tokens=active))
    return turns, stats


def _split_long_turn(
    segment: SegmentDraft,
    config: RefinerConfig,
) -> List[SegmentDraft]:
    remaining = segment.tokens
    pieces: List[SegmentDraft] = []

    while remaining:
        draft = SegmentDraft(tokens=remaining)
        if not _needs_long_split(draft, config):
            pieces.append(draft)
            break

        split_at = _choose_split_index(remaining, config)
        if split_at <= 0 or split_at >= len(remaining):
            pieces.append(draft)
            break

        pieces.append(SegmentDraft(tokens=remaining[:split_at]))
        remaining = remaining[split_at:]

    return pieces


def _needs_long_split(segment: SegmentDraft, config: RefinerConfig) -> bool:
    return (
        segment.duration > config.max_duration_seconds
        or segment.char_count > config.max_chars
    )


def _choose_split_index(tokens: List[WordToken], config: RefinerConfig) -> int:
    target_time = tokens[0].start + max(0.1, config.ideal_duration_seconds)
    boundaries = list(range(1, len(tokens)))

    sentence = [
        index
        for index in boundaries
        if _ends_with_any(tokens[index - 1].text, _SENTENCE_PUNCTUATION)
    ]
    clause = [
        index
        for index in boundaries
        if _ends_with_any(tokens[index - 1].text, _CLAUSE_PUNCTUATION)
    ]
    pauses = [
        index
        for index in boundaries
        if tokens[index].start - tokens[index - 1].end >= config.pause_split_gap_seconds
    ]

    for candidates in (sentence, clause, pauses, boundaries):
        split_at = _best_boundary(tokens, candidates, target_time, config)
        if split_at is not None:
            return split_at

    return max(1, min(len(tokens) - 1, len(tokens) // 2))


def _best_boundary(
    tokens: List[WordToken],
    candidates: List[int],
    target_time: float,
    config: RefinerConfig,
) -> Optional[int]:
    viable = [
        index
        for index in candidates
        if _boundary_is_viable(tokens, index, config)
    ]
    if not viable:
        viable = [
            index
            for index in candidates
            if 0 < index < len(tokens)
            and SegmentDraft(tokens=tokens[:index]).duration <= config.max_duration_seconds
        ]
    if not viable:
        return None

    def score(index: int) -> Tuple[float, float]:
        left = SegmentDraft(tokens=tokens[:index])
        over_duration = max(0.0, left.duration - config.max_duration_seconds)
        over_chars = max(0, left.char_count - config.max_chars)
        return (
            over_duration * 100.0 + over_chars * 10.0 + abs(left.end - target_time),
            abs(left.duration - config.ideal_duration_seconds),
        )

    return min(viable, key=score)


def _boundary_is_viable(
    tokens: List[WordToken],
    index: int,
    config: RefinerConfig,
) -> bool:
    if index <= 0 or index >= len(tokens):
        return False

    left = SegmentDraft(tokens=tokens[:index])
    right = SegmentDraft(tokens=tokens[index:])
    left_big_enough = (
        left.duration >= config.min_duration_seconds
        or left.word_count >= config.min_words
    )
    right_big_enough = (
        right.duration >= config.min_duration_seconds
        or right.word_count >= config.min_words
    )
    return (
        left_big_enough
        and right_big_enough
        and left.duration <= config.max_duration_seconds
        and left.char_count <= config.max_chars
    )


def _merge_short_same_speaker_segments(
    segments: List[SegmentDraft],
    config: RefinerConfig,
) -> Tuple[List[SegmentDraft], int]:
    merged = list(segments)
    merge_count = 0

    changed = True
    while changed:
        changed = False
        for index, segment in enumerate(merged):
            if not _is_short(segment, config):
                continue

            candidates: List[Tuple[float, str]] = []
            if index > 0 and _can_merge(merged[index - 1], segment, config):
                candidates.append((segment.start - merged[index - 1].end, "prev"))
            if index + 1 < len(merged) and _can_merge(segment, merged[index + 1], config):
                candidates.append((merged[index + 1].start - segment.end, "next"))
            if not candidates:
                continue

            _, direction = min(candidates, key=lambda item: (abs(item[0]), 0 if item[1] == "prev" else 1))
            if direction == "prev":
                merged[index - 1] = merged[index - 1].merged_with(segment)
                del merged[index]
            else:
                merged[index] = segment.merged_with(merged[index + 1])
                del merged[index + 1]
            merge_count += 1
            changed = True
            break

    return merged, merge_count


def _is_short(segment: SegmentDraft, config: RefinerConfig) -> bool:
    return (
        segment.duration < config.min_duration_seconds
        or segment.word_count < config.min_words
    )


def _is_ultra_short(segment: SegmentDraft, config: RefinerConfig) -> bool:
    return (
        segment.duration < (config.min_duration_seconds / 2.0)
        or segment.word_count <= 1
    )


def _can_merge(
    left: SegmentDraft,
    right: SegmentDraft,
    config: RefinerConfig,
) -> bool:
    if left.speaker != right.speaker:
        return False

    gap = right.start - left.end
    allowed_gap = (
        config.ultra_short_gap_seconds
        if _is_ultra_short(left, config) or _is_ultra_short(right, config)
        else config.same_speaker_merge_gap_seconds
    )
    if gap > allowed_gap:
        return False

    combined = left.merged_with(right)
    return (
        combined.duration <= config.max_duration_seconds
        and combined.char_count <= config.max_chars
    )


def _segment_to_dict(segment: SegmentDraft) -> Dict[str, Any]:
    speaker_durations: Dict[str, float] = {}
    words: List[Dict[str, Any]] = []
    for token in segment.tokens:
        speaker_durations[token.speaker] = (
            speaker_durations.get(token.speaker, 0.0) + token.duration
        )
        word = dict(token.raw)
        word["word"] = word.get("word") or token.text
        word["start"] = token.start
        word["end"] = token.end
        word["speaker"] = token.speaker
        words.append(word)

    total_voiced = sum(speaker_durations.values())
    purity = 1.0
    if total_voiced > 0.0:
        purity = max(speaker_durations.values()) / total_voiced

    return {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker": segment.speaker,
        "words": words,
        "speaker_purity": round(purity, 4),
    }


def _ends_with_any(text: str, chars: set[str]) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in chars


def _join_word_texts(words: List[str]) -> str:
    output = ""
    for raw_word in words:
        word = raw_word.strip()
        if not word:
            continue
        if output and _needs_space(output[-1], word[0]):
            output += " "
        output += word
    return output.strip()


def _needs_space(previous_char: str, next_char: str) -> bool:
    if next_char in _NO_SPACE_BEFORE:
        return False
    if previous_char in _NO_SPACE_AFTER:
        return False
    if _CJK_RE.match(previous_char) or _CJK_RE.match(next_char):
        return False
    if next_char == "'":
        return False
    return True
