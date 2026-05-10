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
pip install json-repair
git clone https://github.com/garyswansrs/index-tts-vllm.git
cd index-tts-vllm
pip install -r requirements.txt
pip install pydub
pip install flashinfer-python
pip install flash-attn --no-build-isolation --no-cache-dir
pip install audio-separator
pip install clearvoice
pip install google-genai
sudo apt install ffmpeg
hf download garyswansrs/index_tts_2_vllm --local-dir checkpoints
python fastapi_webui_v2.py --use_torch_compile
```

```bash
npx localtunnel --port 8000
ssh -p 443 -R0:localhost:8000 a.pinggy.io
ssh -R 80:localhost:8000 serveo.net
```

## Project Introduction
This project provides a high-performance implementation of **IndexTTS2** using the **vLLM v2** backend. It focuses on extreme inference speed, high concurrency, and a feature-rich user interface for both text-to-speech and advanced audio translation workflows.

By leveraging vLLM's PagedAttention and continuous batching, this implementation achieves significant speedups:
- **RTF (Real-Time Factor)**: Achieved **0.02** on RTX Pro 6000 Blackwell.
- **High Concurrency**: Handles **100** concurrent requests efficiently.
- **IndexTTS2 Support**: Full support for the latest IndexTTS2 model with enhanced emotional expression and duration control.

## Key Features
- **🚀 Ultra-Fast Inference**: Re-implemented with vLLM v2 and PagedAttention for high-speed, concurrent TTS generation.
- **👥 Speaker Preset Management**: Save reference voices as presets to skip pre-processing overhead on every request.
- **⚡ Parallel Chunk Generation**: Automatically splits long text into chunks and processes them in parallel, achieving up to 50x speedup for long synthesis tasks.
- **🎵 MP3 & Multi-Format Output**: Support for MP3 output (via pydub/ffmpeg) for significantly smaller file sizes without sacrificing quality.
- **🌍 Advanced Translation Workflow**:
    - **Transcription**: Local (WhisperX) or Cloud (Gemini) speech-to-text.
    - **Translation**: High-quality translation preserving speaker diarization.
    - **Diarization**: Automatic speaker detection and voice cloning for each speaker.
    - **Edit Mode**: Interactive segment editing, selective regeneration, and timestamp control.
- **🎨 Qwen3-TTS Voice Design**: Describe a voice in natural language (e.g., "A calm, deep male voice with a slight accent") and generate it instantly.
- **🎛️ Audio Enhancement & Separation**:
    - **ClearVoice**: Integrated speech enhancement and super-resolution (MossFormer2).
    - **Audio-Separator**: High-quality vocal/instrumental separation (Mel-Band Roformer).

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
pip install -r requirements.txt

# For optimized GPU inference
pip install flashinfer-python flash-attn --no-build-isolation

# For advanced audio features
pip install audio-separator[gpu] clearvoice google-genai whisperx pydub
```

### 4. Model Weights
Download the pre-converted IndexTTS2 vLLM weights:
```bash
huggingface-cli download garyswansrs/index_tts_2_vllm --local-dir checkpoints
```
If you have official weights in other formats, use the conversion script provided (see below).

## Usage

### 1. Launch Modern FastAPI WebUI (Recommended)
This UI supports the full advanced workflow including translation, voice design, and speaker presets.
```bash
python fastapi_webui_v2.py --model_dir checkpoints --use_torch_compile
```

### 2. Launch Gradio WebUI with Presets
A simpler interface focused on TTS with speaker preset support.
```bash
python webui_with_presets.py --model_dir checkpoints
```

### 3. Model Conversion
To convert official IndexTTS/IndexTTS-1.5 weights to vLLM format:
```bash
bash convert_hf_format.sh /path/to/official_model_dir
```

## API Reference

The modern WebUI (`fastapi_webui_v2.py`) also serves as a comprehensive API server compatible with OpenAI's audio format and custom IndexTTS2 features.

### Start the API Server
```bash
python fastapi_webui_v2.py --model_dir checkpoints --port 8000 --use_torch_compile
```

### Key Endpoints
- `POST /audio/speech`: OpenAI-compatible text-to-speech.
- `POST /api/translate_audio`: Advanced audio-to-audio translation with diarization.
- `POST /add_speaker`: Register a new speaker from reference audio.
- `GET /audio_roles`: List all registered speakers and presets.
- `POST /api/design-voice`: Generate voice from natural language description.

### Startup Parameters
- `--model_dir`: Path to model weights (default: `checkpoints`).
- `--gpu_memory_utilization`: vLLM memory limit (default: `0.25`).
- `--use_torch_compile`: Enable torch.compile for faster startup/inference.
- `--is_fp16`: Use FP16 precision.

## Performance Benchmarks
Tested on RTX Pro 6000 Blackwell:
- **RTF**: **0.02** (IndexTTS2-vLLM)
- **Throughput**: **100** concurrent requests supported.
