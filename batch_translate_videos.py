#!/usr/bin/env python3
"""
Batch Video Translation Script

This script processes videos in an input folder sequentially:
1. Extracts audio using ffmpeg
2. Sends audio to IndexTTS translate_audio API endpoint
3. Downloads translated audio and SRT subtitles
4. Replaces video audio with translated audio
5. Optionally burns SRT subtitles into video

Default behavior (matches UI defaults):
- Voice enhancement: ON
- Merge backing track: ON
- Burn subtitles: OFF (optional)

Requirements:
- ffmpeg installed and in PATH
- requests library: pip install requests
- Running IndexTTS FastAPI server

Usage:
    python batch_translate_videos.py --input ./videos --output ./translated --dest-language English
    python batch_translate_videos.py --input ./videos --output ./translated --dest-language Chinese --burn-srt
    python batch_translate_videos.py --help
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("Error: requests library is required. Install with: pip install requests")
    sys.exit(1)


# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


@dataclass
class TranslationResult:
    """Result of a single video translation."""
    input_path: str
    output_path: Optional[str] = None
    audio_url: Optional[str] = None
    srt_url: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationConfig:
    """Configuration for batch translation."""
    input_dir: str
    output_dir: Optional[str]  # None means same as input_dir
    dest_language: str
    api_base_url: str = "http://localhost:8000"
    burn_srt: bool = False  # Optional: burn subtitles into video
    replace_audio: bool = True
    keep_temp_files: bool = False
    enhance_voice: bool = True  # Matches UI default
    super_resolution_voice: bool = False
    merge_backing_track: bool = True  # Matches UI default
    gemini_model: str = "flash"
    gemini_api_key: Optional[str] = None
    default_speaker_preset: Optional[str] = None
    timeout_seconds: int = 3600  # 1 hour timeout for long videos
    verbose: bool = False
    skip_existing: bool = True  # Skip videos that already have translated output
    
    @property
    def resolved_output_dir(self) -> str:
        """Get output directory, defaulting to input directory."""
        return self.output_dir if self.output_dir else self.input_dir


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_ffprobe() -> bool:
    """Check if ffprobe is available."""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def extract_audio(video_path: str, output_audio_path: str, verbose: bool = False) -> bool:
    """Extract audio from video using ffmpeg as MP3 for smaller file size."""
    # Use MP3 format for much smaller file sizes (typically 10x smaller than WAV)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "libmp3lame",  # MP3 codec
        "-ab", "192k",  # Bitrate (good quality)
        "-ar", "44100",  # Sample rate
        "-ac", "2",  # Stereo
        output_audio_path
    ]
    
    if verbose:
        print(f"  Extracting audio: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Warning: Audio extraction timed out for {video_path}")
        return False


def download_file(url: str, output_path: str, timeout: int = 300) -> bool:
    """Download a file from URL to local path."""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def parse_ndjson_stream(response: requests.Response, verbose: bool = False) -> List[Dict[str, Any]]:
    """Parse newline-delimited JSON (NDJSON) streaming response.
    
    The API returns JSON objects separated by newlines, not SSE format.
    Each line is a complete JSON object like: {"event": "status", "message": "..."}
    """
    events = []
    buffer = ""
    
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            buffer += chunk
            
            # Process complete lines (NDJSON - each line is a JSON object)
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                
                if not line:
                    continue
                
                # Try to parse as JSON directly (NDJSON format)
                try:
                    event_data = json.loads(line)
                    events.append(event_data)
                    if verbose:
                        event_type = event_data.get('event', 'unknown')
                        print(f"    NDJSON event: {event_type}")
                except json.JSONDecodeError:
                    # Maybe it's SSE format with "data:" prefix
                    if line.startswith("data:"):
                        data_content = line[5:].strip()
                        if data_content:
                            try:
                                event_data = json.loads(data_content)
                                events.append(event_data)
                                if verbose:
                                    print(f"    SSE event: {event_data.get('event', 'unknown')}")
                            except json.JSONDecodeError as e:
                                if verbose:
                                    print(f"    Parse error: {e}, line: {line[:100]}")
                    elif verbose:
                        print(f"    Skipping non-JSON line: {line[:100]}")
    
    # Process any remaining data in buffer
    if buffer.strip():
        try:
            event_data = json.loads(buffer.strip())
            events.append(event_data)
            if verbose:
                print(f"    NDJSON event (final): {event_data.get('event', 'unknown')}")
        except json.JSONDecodeError:
            if verbose:
                print(f"    Could not parse remaining buffer: {buffer[:100]}")
    
    return events


def translate_audio_api(
    audio_path: str,
    config: TranslationConfig,
    progress_callback: Optional[callable] = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    """
    Call the translate_audio API endpoint.
    
    Returns:
        Tuple of (audio_url, translated_srt_url, original_srt_url, session_id, metadata)
    """
    url = urljoin(config.api_base_url, "/api/translate_audio")
    
    # Prepare form data
    form_data = {
        "dest_language": config.dest_language,
        "enhance_voice": str(config.enhance_voice).lower(),
        "super_resolution_voice": str(config.super_resolution_voice).lower(),
        "merge_backing_track": str(config.merge_backing_track).lower(),
        "gemini_model": config.gemini_model,
    }
    
    if config.gemini_api_key:
        form_data["gemini_api_key"] = config.gemini_api_key
    
    if config.default_speaker_preset:
        form_data["default_speaker_preset"] = config.default_speaker_preset
    
    # Prepare file - detect MIME type from extension
    audio_ext = os.path.splitext(audio_path)[1].lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    mime_type = mime_types.get(audio_ext, "audio/mpeg")
    
    files = {
        "audio_file": (os.path.basename(audio_path), open(audio_path, "rb"), mime_type)
    }
    
    try:
        if config.verbose:
            print(f"  Calling API: {url}")
            print(f"  Form data: {form_data}")
        
        # Make streaming request
        response = requests.post(
            url,
            data=form_data,
            files=files,
            stream=True,
            timeout=config.timeout_seconds
        )
        
        if response.status_code != 200:
            error_text = response.text[:500] if response.text else "Unknown error"
            return None, None, None, {"error": f"API returned {response.status_code}: {error_text}"}
        
        # Parse SSE events
        audio_url = None
        translated_srt_url = None
        original_srt_url = None
        session_id = None
        metadata = {}
        
        events = parse_ndjson_stream(response, config.verbose)
        
        if config.verbose:
            print(f"  Total SSE events received: {len(events)}")
        
        for event in events:
            event_type = event.get("event", "")
            
            if config.verbose:
                print(f"  SSE Event: {event_type} - {str(event)[:200]}")
            
            if progress_callback and event.get("message"):
                progress_callback(event.get("message", ""))
            
            if event_type == "complete":
                audio_url = event.get("audio_url")
                event_metadata = event.get("metadata") or {}
                
                # Debug: print all keys in complete event
                if config.verbose:
                    print(f"  Complete event keys: {list(event.keys())}")
                    print(f"  Metadata keys: {list(event_metadata.keys()) if event_metadata else 'None'}")
                
                # Get translated subtitle URL - check multiple possible locations (matching UI logic)
                # Priority: subtitle_url > metadata.subtitle.url > subtitle_translated.url > subtitle
                translated_srt_url = None
                if event.get("subtitle_url"):
                    translated_srt_url = event.get("subtitle_url")
                elif event_metadata.get("subtitle") and isinstance(event_metadata.get("subtitle"), dict):
                    translated_srt_url = event_metadata["subtitle"].get("url")
                elif event.get("subtitle_translated"):
                    sub = event.get("subtitle_translated")
                    translated_srt_url = sub.get("url") if isinstance(sub, dict) else sub
                elif event.get("subtitle"):
                    sub = event.get("subtitle")
                    translated_srt_url = sub.get("url") if isinstance(sub, dict) else sub
                    
                # Get original subtitle URL - check multiple possible locations (matching UI logic)
                # Priority: original_subtitle_url > metadata.subtitle_original.url > subtitle_original.url
                original_srt_url = None
                if event.get("original_subtitle_url"):
                    original_srt_url = event.get("original_subtitle_url")
                elif event_metadata.get("subtitle_original") and isinstance(event_metadata.get("subtitle_original"), dict):
                    original_srt_url = event_metadata["subtitle_original"].get("url")
                elif event.get("subtitle_original"):
                    sub = event.get("subtitle_original")
                    original_srt_url = sub.get("url") if isinstance(sub, dict) else sub
                    
                session_id = event.get("session_id") or event_metadata.get("session_id")
                metadata = event
                
                if config.verbose:
                    print(f"  Complete event - audio_url: {audio_url}")
                    print(f"  Complete event - translated_srt_url: {translated_srt_url}")
                    print(f"  Complete event - original_srt_url: {original_srt_url}")
                    
            elif event_type == "error":
                return None, None, None, None, {"error": event.get("message", "Unknown error")}
        
        # If no complete event found, check if there's any audio_url in the last event
        if not audio_url and events:
            last_event = events[-1]
            audio_url = last_event.get("audio_url")
            if audio_url and config.verbose:
                print(f"  Found audio_url in last event: {audio_url}")
        
        return audio_url, translated_srt_url, original_srt_url, session_id, metadata
        
    except requests.exceptions.Timeout:
        return None, None, None, None, {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return None, None, None, None, {"error": str(e)}
    finally:
        files["audio_file"][1].close()


def replace_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    translated_srt_path: Optional[str] = None,
    original_srt_path: Optional[str] = None,
    dest_language: str = "eng",
    verbose: bool = False
) -> bool:
    """Replace video audio track with translated audio and embed SRT subtitles using ffmpeg."""
    # Build command based on available subtitle files
    has_translated_srt = translated_srt_path and os.path.exists(translated_srt_path)
    has_original_srt = original_srt_path and os.path.exists(original_srt_path)
    
    # Map language name to ISO 639-2 code for subtitle metadata
    lang_codes = {
        "english": "eng", "chinese": "chi", "japanese": "jpn", "korean": "kor",
        "spanish": "spa", "french": "fra", "german": "deu", "italian": "ita",
        "portuguese": "por", "russian": "rus", "arabic": "ara", "hindi": "hin",
    }
    dest_lang_code = lang_codes.get(dest_language.lower(), "eng")
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-i", audio_path,
    ]
    
    # Add subtitle inputs
    if has_translated_srt:
        cmd.extend(["-i", translated_srt_path])
    if has_original_srt:
        cmd.extend(["-i", original_srt_path])
    
    # Video and audio codecs
    cmd.extend([
        "-c:v", "copy",  # Copy video stream
        "-c:a", "aac",  # Re-encode audio to AAC for better compatibility
    ])
    
    # Subtitle codec
    if has_translated_srt or has_original_srt:
        cmd.extend(["-c:s", "mov_text"])  # Subtitle codec for mp4
    
    # Map streams
    cmd.extend([
        "-map", "0:v:0",  # Video from original
        "-map", "1:a:0",  # Audio from translated audio file
    ])
    
    # Map subtitle streams and set metadata
    subtitle_input_idx = 2  # Subtitles start at input index 2
    subtitle_stream_idx = 0
    
    if has_translated_srt:
        cmd.extend(["-map", f"{subtitle_input_idx}:0"])
        cmd.extend([f"-metadata:s:s:{subtitle_stream_idx}", f"language={dest_lang_code}"])
        cmd.extend([f"-metadata:s:s:{subtitle_stream_idx}", f"title={dest_language} (Translated)"])
        subtitle_input_idx += 1
        subtitle_stream_idx += 1
    
    if has_original_srt:
        cmd.extend(["-map", f"{subtitle_input_idx}:0"])
        cmd.extend([f"-metadata:s:s:{subtitle_stream_idx}", "language=und"])  # Original language unknown
        cmd.extend([f"-metadata:s:s:{subtitle_stream_idx}", "title=Original"])
        subtitle_stream_idx += 1
    
    cmd.extend(["-shortest", output_path])
    
    # Log what we're doing
    if has_translated_srt:
        print(f"    Embedding translated subtitles: {translated_srt_path}")
    if has_original_srt:
        print(f"    Embedding original subtitles: {original_srt_path}")
    if not has_translated_srt and not has_original_srt:
        print(f"    No subtitles to embed")
    
    if verbose:
        print(f"  ffmpeg command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"  ffmpeg error (exit {result.returncode}): {result.stderr[:500]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Warning: Audio replacement timed out")
        return False


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    verbose: bool = False
) -> bool:
    """Burn SRT subtitles into video using ffmpeg."""
    # Escape special characters in path for ffmpeg filter
    srt_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    
    # Simple subtitle filter - let ffmpeg decide the font
    subtitle_filter = f"subtitles='{srt_path_escaped}'"
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-vf", subtitle_filter,
        "-c:a", "copy",  # Copy audio stream
        output_path
    ]
    
    if verbose:
        print(f"  Burning subtitles: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout for re-encoding
        )
        if result.returncode != 0 and verbose:
            print(f"  ffmpeg stderr: {result.stderr[:500]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Warning: Subtitle burning timed out")
        return False


def get_translated_filename(video_name: str, video_ext: str, dest_language: str) -> str:
    """Generate the translated video filename."""
    # Clean language name for filename
    safe_lang = re.sub(r'[^\w\-]', '_', dest_language.lower())
    return f"{video_name}_{safe_lang}{video_ext}"


def process_single_video(
    video_path: str,
    config: TranslationConfig
) -> TranslationResult:
    """Process a single video file."""
    result = TranslationResult(input_path=video_path)
    start_time = time.time()
    
    video_name = Path(video_path).stem
    video_ext = Path(video_path).suffix
    output_dir = config.resolved_output_dir
    
    # Generate output filename
    safe_lang = re.sub(r'[^\w\-]', '_', config.dest_language.lower())
    output_filename = get_translated_filename(video_name, video_ext, config.dest_language)
    final_output_path = os.path.join(output_dir, output_filename)
    translated_srt_output_path = os.path.join(output_dir, f"{video_name}_{safe_lang}.srt")
    original_srt_output_path = os.path.join(output_dir, f"{video_name}_original.srt")
    
    # Check if already processed
    if config.skip_existing and os.path.exists(final_output_path):
        print(f"\n⏭️  Skipping (already exists): {video_path}")
        print(f"   Output: {final_output_path}")
        result.success = True
        result.output_path = final_output_path
        result.metadata = {"skipped": True, "reason": "output_exists"}
        return result
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_path}")
    print(f"Output: {final_output_path}")
    print(f"{'='*60}")
    
    # Get video duration
    duration = get_video_duration(video_path)
    if duration:
        print(f"  Video duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Extract audio as MP3 (much smaller than WAV)
        print("  Step 1: Extracting audio (MP3)...")
        audio_path = os.path.join(temp_dir, f"{video_name}.mp3")
        
        if not extract_audio(video_path, audio_path, config.verbose):
            result.error = "Failed to extract audio from video"
            print(f"  ERROR: {result.error}")
            return result
        
        audio_size = os.path.getsize(audio_path)
        print(f"  Extracted audio: {audio_size / (1024*1024):.1f} MB")
        
        # Step 2: Translate audio
        print(f"  Step 2: Translating to {config.dest_language}...")
        
        def progress_cb(msg: str):
            if config.verbose:
                print(f"    Progress: {msg}")
        
        audio_url, translated_srt_url, original_srt_url, session_id, metadata = translate_audio_api(
            audio_path, config, progress_cb
        )
        
        if "error" in metadata:
            result.error = metadata["error"]
            print(f"  ERROR: {result.error}")
            return result
        
        if not audio_url:
            result.error = "No audio URL returned from API"
            print(f"  ERROR: {result.error}")
            return result
        
        result.audio_url = audio_url
        result.srt_url = translated_srt_url
        result.session_id = session_id
        result.metadata = metadata
        
        print(f"  Translation complete!")
        print(f"    Audio URL: {audio_url}")
        if translated_srt_url:
            print(f"    Translated SRT URL: {translated_srt_url}")
        if original_srt_url:
            print(f"    Original SRT URL: {original_srt_url}")
        
        # Step 3: Download translated audio
        print("  Step 3: Downloading translated audio...")
        translated_audio_path = os.path.join(temp_dir, f"{video_name}_translated.mp3")
        
        full_audio_url = urljoin(config.api_base_url, audio_url)
        if not download_file(full_audio_url, translated_audio_path, config.timeout_seconds):
            result.error = "Failed to download translated audio"
            print(f"  ERROR: {result.error}")
            return result
        
        print(f"  Downloaded: {os.path.getsize(translated_audio_path) / 1024:.1f} KB")
        
        # Step 4: Download SRT files if available
        translated_srt_local_path = None
        original_srt_local_path = None
        
        print("  Step 4: Downloading subtitles...")
        print(f"    Translated SRT URL: {translated_srt_url}")
        print(f"    Original SRT URL: {original_srt_url}")
        
        if translated_srt_url:
            translated_srt_local_path = os.path.join(temp_dir, f"{video_name}_translated.srt")
            full_translated_srt_url = urljoin(config.api_base_url, translated_srt_url)
            print(f"    Downloading: {full_translated_srt_url}")
            if download_file(full_translated_srt_url, translated_srt_local_path, 60):
                srt_size = os.path.getsize(translated_srt_local_path)
                print(f"    ✓ Translated SRT downloaded: {srt_size} bytes")
                if srt_size < 10:
                    print(f"    Warning: SRT file seems too small, may be empty")
            else:
                print("    ✗ Failed to download translated SRT")
                translated_srt_local_path = None
        else:
            print("    No translated SRT URL available")
        
        if original_srt_url:
            original_srt_local_path = os.path.join(temp_dir, f"{video_name}_original.srt")
            full_original_srt_url = urljoin(config.api_base_url, original_srt_url)
            print(f"    Downloading: {full_original_srt_url}")
            if download_file(full_original_srt_url, original_srt_local_path, 60):
                srt_size = os.path.getsize(original_srt_local_path)
                print(f"    ✓ Original SRT downloaded: {srt_size} bytes")
                if srt_size < 10:
                    print(f"    Warning: SRT file seems too small, may be empty")
            else:
                print("    ✗ Failed to download original SRT")
                original_srt_local_path = None
        else:
            print("    No original SRT URL available")
        
        # Step 5: Create output video
        os.makedirs(output_dir, exist_ok=True)
        
        if config.replace_audio:
            if config.burn_srt and translated_srt_local_path:
                # Burn subtitles into video (requires re-encoding)
                print("  Step 5: Replacing audio and embedding subtitles...")
                
                # First replace audio with embedded SRT tracks
                temp_output = os.path.join(temp_dir, f"{video_name}_audio_replaced{video_ext}")
                
                if not replace_video_audio(
                    video_path, translated_audio_path, temp_output,
                    translated_srt_local_path, original_srt_local_path,
                    config.dest_language, config.verbose
                ):
                    result.error = "Failed to replace video audio"
                    print(f"  ERROR: {result.error}")
                    return result
                
                # Then burn translated subtitles visually
                print("  Step 6: Burning translated subtitles into video...")
                
                if not burn_subtitles(
                    temp_output,
                    translated_srt_local_path,
                    final_output_path,
                    config.verbose
                ):
                    # Fallback: use video with embedded subtitle tracks but not burned
                    print("  Warning: Failed to burn subtitles, using video with embedded subtitle tracks")
                    shutil.copy(temp_output, final_output_path)
                
                result.output_path = final_output_path
            else:
                # Replace audio and embed both SRT tracks (no burning)
                print("  Step 5: Replacing audio and embedding subtitles...")
                
                if not replace_video_audio(
                    video_path, translated_audio_path, final_output_path,
                    translated_srt_local_path, original_srt_local_path,
                    config.dest_language, config.verbose
                ):
                    result.error = "Failed to replace video audio"
                    print(f"  ERROR: {result.error}")
                    return result
                
                result.output_path = final_output_path
        else:
            # Just save the translated audio
            final_audio_output = os.path.join(
                output_dir,
                f"{video_name}_{safe_lang}.mp3"
            )
            shutil.copy(translated_audio_path, final_audio_output)
            result.output_path = final_audio_output
        
        # Also save SRT files as separate files
        if translated_srt_local_path:
            shutil.copy(translated_srt_local_path, translated_srt_output_path)
            print(f"  Saved translated SRT: {translated_srt_output_path}")
        if original_srt_local_path:
            shutil.copy(original_srt_local_path, original_srt_output_path)
            print(f"  Saved original SRT: {original_srt_output_path}")
        
        result.success = True
        result.duration_seconds = time.time() - start_time
        
        print(f"\n  ✅ SUCCESS!")
        print(f"  Output: {result.output_path}")
        print(f"  Processing time: {result.duration_seconds:.1f} seconds")
    
    return result


def find_videos(input_dir: str, dest_language: str = "") -> List[str]:
    """Find all video files in input directory, excluding already-translated files."""
    videos = set()  # Use set to avoid duplicates on case-insensitive filesystems
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return []
    
    # Build pattern to exclude translated files
    safe_lang = re.sub(r'[^\w\-]', '_', dest_language.lower()) if dest_language else ""
    
    for ext in VIDEO_EXTENSIONS:
        # Use case-insensitive glob pattern
        for f in input_path.glob(f"*{ext}"):
            videos.add(str(f.resolve()))  # Use resolved path to normalize
        for f in input_path.glob(f"*{ext.upper()}"):
            videos.add(str(f.resolve()))
    
    # Filter out already-translated files (files ending with _{language}.ext)
    if safe_lang:
        filtered = set()
        for v in videos:
            video_stem = Path(v).stem.lower()
            # Skip files that end with the language suffix (e.g., _chinese, _english)
            if video_stem.endswith(f"_{safe_lang}"):
                continue
            filtered.add(v)
        videos = filtered
    
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Batch translate videos using IndexTTS API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic translation to English (output in same folder as input)
  python batch_translate_videos.py -i ./videos -l English

  # Translation with subtitle burning (optional)
  python batch_translate_videos.py -i ./videos -l Chinese --burn-srt

  # Specify different output directory
  python batch_translate_videos.py -i ./videos -o ./translated -l Japanese

  # Audio only (no video processing)
  python batch_translate_videos.py -i ./videos -l Korean --no-replace-audio

  # Custom API server
  python batch_translate_videos.py -i ./videos -l English --api-url http://192.168.1.100:8000
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing video files"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for translated videos (default: same as input)"
    )
    parser.add_argument(
        "-l", "--dest-language",
        default="Chinese",
        help="Target language (e.g., English, Chinese, Japanese, Korean)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="IndexTTS API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--burn-srt",
        action="store_true",
        help="Burn subtitles into video"
    )
    parser.add_argument(
        "--no-replace-audio",
        action="store_true",
        help="Don't replace video audio, just output translated audio file"
    )
    parser.add_argument(
        "--super-resolution",
        action="store_true",
        help="Enable ClearVoice super resolution"
    )
    parser.add_argument(
        "--gemini-model",
        default="flash",
        choices=["flash", "pro"],
        help="Gemini model to use (default: flash)"
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Override Gemini API key"
    )
    parser.add_argument(
        "--speaker-preset",
        help="Default speaker preset name"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="API timeout in seconds (default: 3600)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process videos even if output already exists"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    print("Checking dependencies...")
    
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)
    print("  ✓ ffmpeg found")
    
    if not check_ffprobe():
        print("Warning: ffprobe not found, video duration detection disabled")
    else:
        print("  ✓ ffprobe found")
    
    # Check API server
    try:
        response = requests.get(urljoin(args.api_url, "/server_info"), timeout=10)
        if response.status_code == 200:
            server_info = response.json()
            print(f"  ✓ API server: {server_info.get('model_name', 'IndexTTS')}")
        else:
            print(f"Warning: API server returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Cannot connect to API server at {args.api_url}")
        print(f"  {e}")
        print("Please ensure the IndexTTS FastAPI server is running")
        sys.exit(1)
    
    # Create config
    config = TranslationConfig(
        input_dir=args.input,
        output_dir=args.output,  # None means same as input
        dest_language=args.dest_language,
        api_base_url=args.api_url,
        burn_srt=args.burn_srt,
        replace_audio=not args.no_replace_audio,
        enhance_voice=True,  # Matches UI default
        super_resolution_voice=args.super_resolution,
        merge_backing_track=True,  # Matches UI default
        gemini_model=args.gemini_model,
        gemini_api_key=args.gemini_api_key,
        default_speaker_preset=args.speaker_preset,
        timeout_seconds=args.timeout,
        verbose=args.verbose,
        skip_existing=not args.no_skip_existing
    )
    
    # Find videos
    videos = find_videos(config.input_dir, config.dest_language)
    
    if not videos:
        print(f"\nNo video files found in {config.input_dir}")
        print(f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)
    
    print(f"\nFound {len(videos)} video(s) to process:")
    for v in videos:
        print(f"  - {os.path.basename(v)}")
    
    print(f"\nConfiguration:")
    print(f"  Target language: {config.dest_language}")
    print(f"  Output directory: {config.resolved_output_dir}")
    print(f"  Burn subtitles: {config.burn_srt}")
    print(f"  Replace audio: {config.replace_audio}")
    print(f"  Voice enhancement: {config.enhance_voice}")
    print(f"  Super resolution: {config.super_resolution_voice}")
    print(f"  Merge backing: {config.merge_backing_track}")
    print(f"  Gemini model: {config.gemini_model}")
    print(f"  Skip existing: {config.skip_existing}")
    
    # Process videos sequentially
    results: List[TranslationResult] = []
    start_time = time.time()
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing video...")
        try:
            result = process_single_video(video, config)
            results.append(result)
        except Exception as e:
            results.append(TranslationResult(
                input_path=video,
                error=str(e)
            ))
    
    # Summary
    total_time = time.time() - start_time
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n{'='*60}")
    print("BATCH TRANSLATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        print(f"\n✅ Successful translations:")
        for r in successful:
            print(f"  - {os.path.basename(r.input_path)} -> {os.path.basename(r.output_path or 'N/A')}")
    
    if failed:
        print(f"\n❌ Failed translations:")
        for r in failed:
            print(f"  - {os.path.basename(r.input_path)}: {r.error}")
    
    # Exit with error code if any failed
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()


