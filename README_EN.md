<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

quick start

```bash
conda update -n base -c defaults conda
conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge sox
pip install whisperx
pip install "nemo_toolkit[asr]"
pip install json-repair
git clone https://github.com/garyswansrs/index-tts-vllm
cd index-tts-vllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install v-diffusion alias-free-torch dill einops-exts huggingface_hub importlib-resources nnAudio PyWavelets safetensors scipy soxr torchsde tqdm transformers v-diffusion-pytorch vector-quantize-pytorch
pip install --no-deps --ignore-requires-python "git+https://github.com/Stability-AI/stable-audio-tools.git"
pip install -U "yt-dlp[default]"
pip install -U yt-dlp-ejs
pip install -U bgutil-ytdlp-pot-provider
pip install pydub
pip install flashinfer-python
pip install flash-attn --no-build-isolation --no-cache-dir
pip install audio-separator
pip install clearvoice
pip install google-genai
pip install qwen-asr
pip install omnivad
pip install sentencepiece
pip install "numpy<2"
sudo apt install ffmpeg
hf download garyswansrs/index_tts_2_vllm --local-dir checkpoints
export YTDLP_NODE_PATH="$(node -p 'process.execPath')"
python fastapi_webui_v2.py --use_torch_compile
```

```bash
npx localtunnel --port 8000
ssh -p 443 -R0:localhost:8000 a.pinggy.io
ssh -R 80:localhost:8000 serveo.net
```

---

## Project Introduction
This project provides a high-performance implementation of **IndexTTS2** using the **vLLM v2** backend. It focuses on extreme inference speed, high concurrency, and a feature-rich user interface for both text-to-speech and advanced audio translation workflows.

By leveraging vLLM's PagedAttention and continuous batching, this implementation achieves significant speedups:
- **RTF (Real-Time Factor)**: Achieved **0.02** on RTX Pro 6000 Blackwell.
- **High Concurrency**: Handles **100** concurrent requests efficiently.
- **IndexTTS2 Support**: Full support for the latest IndexTTS2 model with enhanced emotional expression and duration control.

---

## Web UI Modules
The modern Web UI [index_new.html](file:///d:/repo/index_tts_2/index-tts-vllm/index_new.html) is built around 7 key functional modules:

1. **🎵 Speech Synthesis (Voice Studio)**
   - Generate speech using registered speaker presets.
   - Adjust emotion text prompts (e.g., "excited", "whispering") and intensity weights (0.0 to 1.0).
   - Control diffusion quality steps and text splitting thresholds (`max_text_tokens_per_sentence`).
   - Listen to real-time audio playback or stream chunks instantly for low-latency feedback.
2. **Stable Audio 3 Music / SFX**
   - Generate instrumental music, ambience, and sound effects from text prompts.
   - Uses `stable-audio-3-medium` as the default model on high-VRAM RTX Pro 6000 Blackwell servers.
   - Supports optional negative prompts, sampler controls, seed control, init audio, and inpainting inputs.
   - Loads from `checkpoints/stable-audio-3/*` at runtime, so an HF token is not needed once checkpoints are downloaded.
3. **🌐 Translate & Edit**
   - Translate speech audio or video into another language while preserving timing and speaker diarization.
   - **Interactive Segment Editor**: Inspect, tweak timings/transcriptions, assign different speaker presets to specific segments, and regenerate modified segments selectively.
   - Export generated subtitles as standard SRT files.
4. **DL Video Download & replacement**
   - Download source videos from YouTube or other sites using `yt-dlp`.
   - Extract audio for translation and burn the translated audio/subtitles back into the original video with a single click.
5. **🎭 Speaker Presets Library**
   - Register new reference voices by uploading reference audio files.
   - **Smart Silence Trimming**: Automatically splits and trims long references at silence points to a 3-15 second sweet spot for optimal voice cloning.
   - **ClearVoice Enhancement**: Clean reference audio via speech enhancement (MossFormerGAN_SE_16K, FRCRN_SE_16K, MossFormer2_SE_48K) and 48kHz super-resolution.
6. **🎨 Qwen3-TTS Voice Design**
   - Describe a voice in natural language (e.g., "A warm, deep male voice speaking calmly with a British accent").
   - Test-synthesize text and save the designed voice directly to the Preset Library.
7. **📚 API Integration Docs**
   - Interactive, styled REST API documentation integrated right into the interface.

---

## Advanced Pipelines & Technical Workflows

### 1. ASR (Speech-to-Text) & Diarization
The translation workflow supports multiple transcription and alignment backends:
- **Cloud Gemini**: Fast and highly accurate cloud-based ASR and translation.
- **Local WhisperX**: Precise phoneme-level word alignment and segment timing.
- **Qwen3-ASR + OmniVAD**: Extremely robust local diarized ASR. Supports Sortformer and Pyannote diarization backends.
- **NVIDIA Parakeet**: Fast local ASR optimized for English and European languages.

### 2. Vocal & Backing Track Separation
If **audio separation** is enabled, the system uses `audio-separator` (Mel-Band Roformer, BS-Roformer, or UVR-MDX-NET) to split incoming audio into vocals and instrumental backing tracks. After translating the vocals, it automatically blends the new voice with the original instrumental track to produce a high-quality localized output.

### 3. Cookies Management
To download videos that require authentication (e.g., age-restricted YouTube videos), the Web UI supports domain-level cookie management:
- Import cookies directly from browser request cURL commands.
- Upload standard Netscape `cookies.txt` files.

---

## Installation

### 1. System Dependencies
Requires `ffmpeg` for audio processing and `sox` for some backend utilities.
```bash
sudo apt update && sudo apt install ffmpeg sox libstdc++6 -y
```

### 2. Environment Setup
We recommend using Conda to manage your environment:
```bash
conda create -n indextts python=3.12 -y
conda activate indextts
```

### 3. Python Dependencies
Install the core requirements and specialized libraries for advanced features:
```bash
# CUDA 13.0 / RTX Pro 6000 Blackwell torch stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt

# For optimized GPU inference
pip install flashinfer-python flash-attn --no-build-isolation

# For Stable Audio 3. Keep --no-deps so it does not downgrade torch/torchaudio.
# The current GitHub source is required for SA3 configs that use local_add_cond_dim.
pip install -U --force-reinstall --no-deps --ignore-requires-python "git+https://github.com/Stability-AI/stable-audio-tools.git"
pip install alias-free-torch dill einops-exts huggingface_hub importlib-resources nnAudio PyWavelets safetensors scipy soxr torchsde tqdm transformers v-diffusion-pytorch vector-quantize-pytorch

# For advanced audio features
pip install audio-separator[gpu] clearvoice google-genai whisperx pydub

# Optional higher-quality ASR/alignment pipeline
# Install this in a separate environment from qwen-tts for now.
pip install qwen-asr omnivad litai

# Optional fast English / European-language ASR pipeline
pip install -U "nemo_toolkit[asr]"
```

> [!NOTE]
> Do not install `qwen-asr` into the same environment as `qwen-tts` for now: current releases pin incompatible exact `transformers` versions (`qwen-tts` pins `4.57.3`, while `qwen-asr` pins `4.57.6`). Also do not install `qwen-asr[vllm]` into this environment; this project pins `vllm==0.10.2` for IndexTTS2.

### 4. Model Weights
Download the pre-converted IndexTTS2 vLLM weights and Stable Audio 3 checkpoints:
```bash
huggingface-cli download garyswansrs/index_tts_2_vllm --local-dir checkpoints

# Stable Audio 3 repos are gated. Accept the model terms on Hugging Face first,
# then run `hf auth login` or `huggingface-cli login` before downloading.
huggingface-cli download stabilityai/stable-audio-3-medium --local-dir checkpoints/stable-audio-3/medium
huggingface-cli download stabilityai/stable-audio-3-small-music --local-dir checkpoints/stable-audio-3/small-music
huggingface-cli download stabilityai/stable-audio-3-small-sfx --local-dir checkpoints/stable-audio-3/small-sfx
```

The server loads Stable Audio 3 from these local folders first. Once the model files are present, runtime inference does not need `HF_TOKEN`.

---

## Startup Reference & CLI Parameters

Launch the FastAPI WebUI and API server:
```bash
python fastapi_webui_v2.py [OPTIONS]
```

### Options:
- `--host` (string): Host IP to bind the server to (default: `0.0.0.0`).
- `--port` (integer): Port number to run the web API on (default: `8000`).
- `--model_dir` (string): Path to model checkpoints directory (default: `checkpoints`).
- `--is_fp16` (flag): Enable FP16 inference precision.
- `--use_torch_compile` (flag): Enable `torch.compile` for faster model execution.
- `--gpu_memory_utilization` (float): vLLM GPU memory utilization limit (default: `0.25`).
- `--verbose` (flag): Enable verbose logging output in the console.

---

## API Reference

The FastAPI server [fastapi_webui_v2.py](file:///d:/repo/index_tts_2/index-tts-vllm/fastapi_webui_v2.py) exposes a rich, asynchronous REST API.

### 🎙️ Speech Generation

#### 1. POST `/speak`
Synthesize speech from text using an existing speaker preset.
- **Request Body (JSON)**:
  - `text` (string, required): The text to synthesize.
  - `name` (string, required): The speaker preset name.
  - `emotion_text` (string, optional): Emotional description.
  - `emotion_weight` (float, optional): Intensity 0.0-1.0 (default: `0.6`).
  - `diffusion_steps` (int, optional): Quality steps (default: `10`).
  - `max_text_tokens_per_sentence` (int, optional): Text split threshold (default: `120`).
- **Response**: `audio/mpeg` binary audio data (MP3).

#### 2. POST `/clone_voice`
Clone a voice using an uploaded reference audio file (zero-shot synthesis).
- **Request Body (`multipart/form-data`)**:
  - `text` (string, required): Text to synthesize.
  - `reference_audio_file` (file, required): Audio file containing the target voice.
  - `emotion_text`, `emotion_weight`, `diffusion_steps`, `max_text_tokens_per_sentence`: Optional settings.
- **Response**: `audio/mpeg` binary audio data (MP3).

#### 3. POST `/speak_stream`
Generate speech with streaming chunks for ultra-low latency.
- **Request Body (JSON)**: Same parameters as `/speak`.
- **Response**: `text/event-stream` SSE chunks.
  - Stream Format: `CHUNK:{idx}:{size}:{status}\n{audio_bytes}`
  - Status: `CONTINUE` (more chunks follow) or `LAST` (final chunk).

#### 4. POST `/clone_voice_stream`
Clone a voice with streaming chunk outputs.
- **Request Body (`multipart/form-data`)**: Same parameters as `/clone_voice`.
- **Response**: Same format as `/speak_stream`.

#### 5. POST `/audio/speech`
OpenAI-compatible text-to-speech API.
- **Request Body (JSON)**:
  - `model` (string, required): e.g., `index-tts-2`.
  - `input` (string, required): Text to synthesize.
  - `voice` (string, required): Speaker preset name.
  - `response_format` (string, optional): Output format (default: `mp3`).
  - `speed` (float, optional): Speed factor (default: `1.0`).
- **Response**: Binary audio stream.

---

### Stable Audio 3 Music / SFX

#### 1. GET `/api/stable-audio/models`
List Stable Audio 3 variants, checkpoint readiness, and loaded model state.

#### 2. POST `/api/stable-audio/generate`
Generate music, ambience, or sound effects from a prompt. Defaults to `stable-audio-3-medium`.
- **Request Body (`multipart/form-data` or JSON)**:
  - `prompt` (string, required): Audio description.
  - `variant_key` (string, optional): `medium`, `small-music`, or `small-sfx`.
  - `negative_prompt` (string, optional): Concepts to avoid.
  - `duration` (int, optional): Target length in seconds.
  - `steps`, `cfg_scale`, `sampler_type`, `seed` (optional): Sampling controls.
  - `init_audio_file`, `inpaint_audio_file` (files, optional): Audio-to-audio and inpainting inputs.
  - `response_format` (string, optional): `mp3`, `wav`, `flac`, `ogg`, `opus`, `aac`, or `webm`.
- **Response**: Binary audio data in the requested format.

#### 3. POST `/api/stable-audio/unload`
Unload Stable Audio 3 models from GPU memory. Pass `variant_key` to unload one model, or omit it to unload all.

---

### 👥 Speaker Management

#### 1. POST `/add_speaker`
Register a new reference speaker.
- **Request Body (`multipart/form-data`)**:
  - `name` (string, required): Target speaker preset name.
  - `audio_files` (files, required): One or more reference audio files.
  - `enhance_voice` (bool, optional): Run speech enhancement on the reference.
  - `enhancement_model` (string, optional): Model key (default: `MossFormerGAN_SE_16K`).
  - `super_resolution_voice` (bool, optional): Enable 48kHz super resolution.
- **Response**: JSON confirmation with status and details.

#### 2. POST `/delete_speaker`
Remove a registered speaker preset.
- **Request Body (`multipart/form-data` or JSON)**:
  - `name` (string, required): Speaker name to delete.
- **Response**: JSON status.

#### 3. GET `/audio_roles`
List all available speakers and presets.
- **Response**: JSON list of speaker names and metadata.

#### 4. GET `/api/speaker_preview/{speaker_name}`
Retrieve the reference MP3 preview for a speaker.
- **Response**: `audio/mpeg` binary data.

---

### 🌐 Speech Translation & Editing

#### 1. POST `/api/translate_audio`
Translate a full speech audio file into another language.
- **Request Body (`multipart/form-data`)**:
  - `audio_file` (file, required): Input audio file.
  - `dest_language` (string, required): Target language (e.g., "English", "Chinese").
  - `audio_separator_enabled` (bool, optional): Enable instrumental separation.
  - `audio_separator_model` (string, optional): 'fast', 'balance', or 'quality'.
  - `enhance_voice` (bool, optional): Enable voice enhancement.
  - `enhancement_model` (string, optional): MossFormer/FRCRN model choice.
  - `super_resolution_voice` (bool, optional): Enable 48kHz upsampling.
  - `merge_backing_track` (bool, optional): Merge backend instrumental track (default: `true`).
  - `transcription_pipeline` (string, optional): 'gemini', 'whisperx', 'qwen_omnivad', or 'parakeet'.
  - `translation_llm_model` (string, optional): Translation LLM.
- **Response**: `audio/mpeg` binary translated audio with `X-Translation-Segments` headers containing detailed segment metadata.

#### 2. POST `/api/translate_segments`
Transcribe/translate a voice file and return editable segment metadata (first step of advanced workflow).
- **Request Body (`multipart/form-data`)**: Similar parameters to `/api/translate_audio`.
- **Response**: `text/event-stream` SSE progress updates, ending with a final `complete` event containing the `session_id` and list of diarized segments.

#### 3. POST `/api/translate_generate_segments`
Synthesize final translated audio using modified segment metadata and speaker assignments.
- **Request Body (JSON)**:
  - `session_id` (string, required): Session ID.
  - `segments` (array of objects, required): Edited segments with translation text, timings, and speaker assignments.
  - `speaker_overrides` (object, optional): Map of speaker IDs to presets.
- **Response**: `text/event-stream` SSE progress events, culminating in `complete` with audio output URLs.

#### 4. POST `/api/translate_segment_preview`
Quickly test-generate a single edited segment.
- **Request Body (JSON)**:
  - `session_id` (string, required): Session ID.
  - `segment` (object, required): A single segment definition.
- **Response**: JSON containing the temporary preview URL.

#### 5. GET `/api/segment_preview/{session_id}/{segment_index}`
Download preview audio for a specific segment.
- **Response**: `audio/mpeg` file.

---

### 📦 Parallel Chunk Processing

#### 1. POST `/api/translate_split_audio`
Split a long audio file into manageable chunks at silence points.
- **Request Body (`multipart/form-data`)**:
  - `audio_file` (file, required): Source audio.
  - `chunk_min_minutes` (float, optional): Default `3.0`.
  - `chunk_max_minutes` (float, optional): Default `6.0`.
  - `min_silence_ms` (int, optional): Silence split window (default: `2000`).
- **Response**: SSE stream containing chunk session details and a batch ID.

#### 2. POST `/api/translate_generate_chunks`
Translate split chunks concurrently.
- **Request Body (JSON)**:
  - `chunk_session_ids` (array, required): List of session IDs to process.
  - `dest_language` (string, required): Target language.
- **Response**: SSE stream tracking parallel chunk synthesis progress.

#### 3. POST `/api/translate_merge_chunks`
Merge translated chunks back into a unified file and export subtitle files.
- **Request Body (JSON)**:
  - `chunk_batch_id` (string, required): The batch ID.
  - `merge_backing_track` (bool, optional): Merge back original instrumental backing tracks.
- **Response**: JSON containing merged `audio_url` and `subtitle_url`.

---

### 🎨 Voice Design (Qwen3-TTS)

#### 1. POST `/api/design-voice`
Synthesize text with a custom voice designed from natural language descriptions.
- **Request Body (JSON)**:
  - `text` (string, required): Text to speak.
  - `voice_description` (string, required): Description (e.g. "soft whispers, calm female voice").
  - `language` (string, optional): Target language.
- **Response**: Binary audio file.

#### 2. POST `/api/design-voice/save-preset`
Save the last designed voice as a reusable speaker preset.
- **Request Body (JSON)**:
  - `preset_name` (string, required): Target preset name.
  - `description` (string, optional): Override description.
- **Response**: JSON status.

#### 3. GET `/api/design-voice/languages`
Get a list of supported voice design languages.

#### 4. GET `/api/design-voice/status`
Check if the Qwen3-TTS voice design backend model is loaded and ready.

---

### 📹 Video Downloader & Cookies

#### 1. POST `/api/video_info`
Extract video info and formats from a URL via `yt-dlp`.
- **Request Body (JSON)**: `url` (string, required).
- **Response**: JSON metadata including formats, title, and durations.

#### 2. POST `/api/video_download`
Download video files.
- **Request Body (JSON)**: `url` (required), `quality` (optional).
- **Response**: SSE progress stream.

#### 3. POST `/api/video_replace_audio`
Mux a translated audio file and subtitles back into a downloaded video.
- **Request Body (JSON)**:
  - `downloaded_video_id` (string, required): Video filename.
  - `audio_file_name` (string, required): Translated audio filename.
  - `subtitle_file_name` (string, optional): SRT filename.
  - `output_filename` (string, optional): Custom name.
- **Response**: JSON containing the download link of the output video.

#### 4. GET `/api/cookies`
Get all saved cookies by domain.

#### 5. POST `/api/cookies/import_curl`
Import site cookies from cURL command headers.
- **Request Body (JSON)**: `curl_command` (string, required), `domain` (string, optional).

#### 6. POST `/api/cookies/upload`
Upload a Netscape `cookies.txt` file.
- **Request Body (`multipart/form-data`)**: `file` (file), `domain` (string).

#### 7. DELETE `/api/cookies/{domain}`
Delete cookies associated with a domain.

---

### 🛠️ Utilities

#### 1. GET `/server_info`
Get server capabilities, version, and hardware backend state.

#### 2. GET `/api/prompt_templates`
Get default Gemini transcription and translation system prompts.

#### 3. POST `/api/estimate_duration`
Estimate speech duration in milliseconds before rendering.
- **Request Body (JSON)**: `text` (string), `language` (string).
- **Response**: JSON containing `duration_ms`.

#### 4. POST `/api/clear_outputs`
Clean temporary files and free space in the `outputs/` folder.

---

## Code Examples

### Basic Speech Synthesis (Python)
```python
import requests

url = "http://127.0.0.1:8000/speak"
payload = {
    "text": "Hello! Welcome to the high performance Voice Studio.",
    "name": "my_speaker_preset",
    "emotion_text": "friendly and professional",
    "emotion_weight": 0.7
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    with open("output.mp3", "wb") as f:
        f.write(response.content)
    print("Audio saved successfully!")
else:
    print(f"Error: {response.json()}")
```

### Streaming Audio Chunking (JavaScript / Node.js)
```javascript
const fetch = require('node-fetch');
const fs = require('fs');

async function streamSpeech() {
    const response = await fetch("http://127.0.0.1:8000/speak_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            text: "This is a demonstration of streaming audio chunks.",
            name: "my_speaker_preset"
        })
    });

    const fileStream = fs.createWriteStream("streaming_output.mp3");

    // Custom SSE chunk parsing loop
    response.body.on('data', (buffer) => {
        let offset = 0;
        while (offset < buffer.length) {
            // Find boundaries or write incoming bytes
            // Note: In browser environments, EventSource or readable streams can read the custom CHUNK syntax:
            // CHUNK:{idx}:{size}:{status}\n{audio_bytes}
            // For simple clients, parsing the chunk headers extracts raw MP3 bytes.
        }
    });
}
```
