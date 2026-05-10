import modal
import os
import asyncio
from typing import List, Dict, Optional

# Use CUDA 12.8.1 as requested
cuda_version = "12.8.1"
flavor = "devel" 
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create Modal image for IndexTTS v2 with vLLM optimization
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("ffmpeg", "git", "wget", "build-essential", "gcc", "g++", "cmake", "sox", "libsox-fmt-all")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/usr/local/cuda", 
        "TORCH_CUDA_ARCH_LIST": "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0",
        "FORCE_CUDA": "1",
        "CXX": "g++",
        "CC": "gcc",
        
        # PyTorch Memory Optimizations for v2 batch processing
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
        "TORCH_CUDNN_BENCHMARK": "1",  # Enable cuDNN autotuning
        "TORCH_COMPILE_MODE": "reduce-overhead",  # Optimize for TTS workloads

        # Cache directories for faster subsequent runs
        "HF_HOME": "/persistent_cache/huggingface",
        "TORCH_HOME": "/persistent_cache/torch", 
        "TRANSFORMERS_CACHE": "/persistent_cache/transformers",
        "CUDA_CACHE_PATH": "/persistent_cache/cuda_cache",
        "VLLM_CACHE": "/persistent_cache/vllm_cache"
    })
    .run_commands("pip install --upgrade pip setuptools wheel")
    .pip_install(
        "torch", 
        "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu128"
    )
    .pip_install(
        "litai",
        "whisperx"
    )
    .run_commands(
        "git clone https://github.com/garyswansrs/index-tts-vllm.git /app/index-tts-vllm"
    )
    .run_commands(
        "cd /app/index-tts-vllm && pip install -r requirements.txt"
    )
    .pip_install(
        "pydub",
        "flashinfer-python"
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .run_commands("pip install audio-separator")
    .run_commands("pip install clearvoice google-genai")
)

app = modal.App("vllm-indextts-v2", image=image)

# Create persistent storage volumes
app_storage = modal.Volume.from_name("indextts-v2-app", create_if_missing=True)
cache_storage = modal.Volume.from_name("indextts-v2-cache", create_if_missing=True)

# Configuration
PERSISTENT_APP_DIR = "/persistent_app"
PERSISTENT_CACHE_DIR = "/persistent_cache"

@app.function(
    image=image,
    timeout=3600,
    volumes={
        PERSISTENT_APP_DIR: app_storage,
        PERSISTENT_CACHE_DIR: cache_storage
    },
    cpu=4.0,
    memory=8192
)
def prepare_model():
    """
    CPU function to:
    1. Copy the entire /app/index-tts-vllm to persistent storage
    2. Update the application with latest code from git (git pull origin)
    3. Download the IndexTTS v2 model into the persistent checkpoints folder
       (includes Qwen3-TTS Voice Design model)
    
    This is a one-time setup that creates a fully self-contained persistent app.
    """
    import subprocess
    import shutil
    from pathlib import Path
    
    print("🚀 Preparing IndexTTS v2 application and model...")
    
    # Step 1: Copy the entire application to persistent storage
    persistent_app_path = Path(PERSISTENT_APP_DIR)
    source_app_path = Path("/app/index-tts-vllm")
    
    if not persistent_app_path.exists() or len(list(persistent_app_path.iterdir())) == 0:
        print("📂 Copying application to persistent storage...")
        persistent_app_path.mkdir(exist_ok=True)
        
        # Copy all files from source to persistent storage
        for item in source_app_path.iterdir():
            dest_item = persistent_app_path / item.name
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)
                print(f"   📁 Copied directory: {item.name}")
            else:
                shutil.copy2(item, dest_item)
                print(f"   📄 Copied file: {item.name}")
        
        print("✅ Application copied to persistent storage successfully!")
    else:
        print("✅ Application already exists in persistent storage")
    
    # Step 2: Update the application with latest code from git (force override local changes)
    print("\n📥 Step 2: Updating application from git repository (force override)...")
    repo_update_status = {
        "success": False,
        "message": "",
        "output": ""
    }
    try:
        # Change to persistent app directory
        os.chdir(str(persistent_app_path))
        print(f"   📁 Changed to directory: {persistent_app_path}")
        
        # Step 2a: Fetch latest from all remotes
        print("   📥 Fetching latest from all remotes...")
        fetch_result = subprocess.run(
            ["git", "fetch", "--all"],
            capture_output=True,
            text=True,
            cwd=str(persistent_app_path)
        )
        
        if fetch_result.returncode != 0:
            print(f"⚠️ Git fetch failed: {fetch_result.stderr.strip()}")
            repo_update_status["message"] = f"Git fetch failed: {fetch_result.stderr.strip()}"
            repo_update_status["output"] = fetch_result.stderr.strip()
        else:
            print("   ✅ Fetch completed")
            
            # Step 2b: Get the default branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(persistent_app_path)
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "main"
            print(f"   📌 Current branch: {current_branch}")
            
            # Step 2c: Hard reset to origin/branch (force override local changes)
            print(f"   🔄 Resetting to origin/{current_branch} (force override local changes)...")
            reset_result = subprocess.run(
                ["git", "reset", "--hard", f"origin/{current_branch}"],
                capture_output=True,
                text=True,
                cwd=str(persistent_app_path)
            )
            
            if reset_result.returncode == 0:
                print(f"✅ Git reset successful!")
                repo_update_status["success"] = True
                repo_update_status["message"] = f"Repository updated to origin/{current_branch}"
                repo_update_status["output"] = reset_result.stdout.strip()
                if reset_result.stdout.strip():
                    print(f"   📋 Output: {reset_result.stdout.strip()}")
            else:
                print(f"⚠️ Git reset failed with exit code: {reset_result.returncode}")
                repo_update_status["message"] = f"Git reset failed with exit code {reset_result.returncode}"
                repo_update_status["output"] = reset_result.stderr.strip() or reset_result.stdout.strip()
                if reset_result.stderr.strip():
                    print(f"   ⚠️ Stderr: {reset_result.stderr.strip()}")
    except Exception as e:
        print(f"⚠️ Git update failed (non-fatal): {str(e)}")
        print("   Continuing with existing code...")
        repo_update_status["message"] = f"Git update exception: {str(e)}"
    
    # Step 3: Download model directly into persistent checkpoints folder
    checkpoints_dir = persistent_app_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    print(f"📦 Downloading IndexTTS v2 model to: {checkpoints_dir}")
    
    try:
        # Use huggingface-hub to download the model directly into checkpoints
        result = subprocess.run([
            "python", "-c", 
            f"""
from huggingface_hub import snapshot_download
import os

print("Downloading IndexTTS v2 model...")
snapshot_download(
    repo_id="garyswansrs/index_tts_2_vllm",
    local_dir="{checkpoints_dir}",
    local_dir_use_symlinks=False
)
print("Model download completed!")
"""
        ], check=True, capture_output=True, text=True, cwd=str(persistent_app_path))
        
        print("✅ IndexTTS model download completed successfully!")
        
        # Step 3b: Voice Design model is now included in the checkpoints directory
        voice_design_dir = checkpoints_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        
        # Step 4: List downloaded files for verification
        print("🔍 Listing downloaded model files...")
        for file_path in checkpoints_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
                relative_path = file_path.relative_to(checkpoints_dir)
                print(f"   📄 {relative_path} ({file_size:.1f} MB)")
        
        # Check for vLLM directory (should exist in pre-converted model)
        vllm_dir = checkpoints_dir / "gpt"
        if vllm_dir.exists():
            print(f"   ✅ vLLM model directory: {vllm_dir}")
            vllm_files = list(vllm_dir.iterdir())
            print(f"   📁 vLLM files: {[f.name for f in vllm_files]}")
        else:
            print(f"   ⚠️ vLLM directory not found: {vllm_dir}")
        
        # Step 5: List complete application structure for verification
        print("\n📋 Persistent application structure:")
        def show_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            items = sorted(list(path.iterdir()))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                if item.is_dir() and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    show_tree(item, prefix + extension, max_depth, current_depth + 1)
        
        show_tree(persistent_app_path)
        
        print(f"\n✅ IndexTTS v2 application and models preparation completed!")
        print(f"📁 Persistent app location: {persistent_app_path}")
        print(f"📁 IndexTTS model location: {checkpoints_dir}")
        print(f"📁 Voice Design model location: {voice_design_dir}")
        print("🚀 Ready for inference deployment!")
        
        return {
            "status": "success",
            "message": "IndexTTS v2 application and models prepared successfully",
            "app_dir": str(persistent_app_path),
            "model_dir": str(checkpoints_dir),
            "voice_design_dir": str(voice_design_dir),
            "vllm_ready": vllm_dir.exists(),
            "voice_design_ready": voice_design_dir.exists(),
            "repo_update": repo_update_status
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to download model: {e.stderr}"
        print(f"❌ {error_msg}")
        return {
            "status": "error", 
            "message": error_msg,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "repo_update": repo_update_status
        }
    except Exception as e:
        error_msg = f"Model preparation failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "repo_update": repo_update_status
        }



@app.function(
    image=image,
    timeout=600,
    volumes={
        PERSISTENT_APP_DIR: app_storage,
        PERSISTENT_CACHE_DIR: cache_storage
    }
)
def clear_cache():
    """
    Function to clear persistent caches while preserving the app and models.
    
    Usage:
        modal run deploy_vllm_indextts_v2.py::clear_cache
    """
    import os
    import shutil
    import glob
    from pathlib import Path
    
    print("🧹 Starting cache clearing process for IndexTTS v2...")
    print("⚠️  This will clear caches but preserve the app and models.")
    
    # Clear general persistent caches
    cache_dirs_to_clear = [
        "/persistent_cache/huggingface",
        "/persistent_cache/torch", 
        "/persistent_cache/transformers",
        "/persistent_cache/cuda_cache",
        "/persistent_cache/vllm_cache",
        "/persistent_cache/torch_compile_cache"  # torch.compile artifacts
    ]
    
    # Clear app-specific caches (but keep the app and models)
    persistent_app_path = Path(PERSISTENT_APP_DIR)
    app_cache_dirs_to_clear = []
    if persistent_app_path.exists():
        app_cache_dirs_to_clear = [
            persistent_app_path / "speaker_presets",
            persistent_app_path / "emotion_cache",
            persistent_app_path / "emb_cache",
            persistent_app_path / "outputs"
        ]
    
    # Calculate total cache size before clearing
    total_size_before = 0
    all_dirs = cache_dirs_to_clear + [str(d) for d in app_cache_dirs_to_clear]
    
    for cache_dir in all_dirs:
        if os.path.exists(cache_dir):
            try:
                for dirpath, dirnames, filenames in os.walk(cache_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size_before += os.path.getsize(filepath)
            except Exception as e:
                print(f"⚠️ Could not calculate size for {cache_dir}: {e}")
    
    print(f"💾 Total cache size before clearing: {total_size_before / (1024 * 1024):.2f} MB")
    
    # Clear cache directories
    cleared_dirs = []
    failed_dirs = []
    
    # Clear general caches
    for cache_dir in cache_dirs_to_clear:
        try:
            if os.path.exists(cache_dir):
                print(f"🗑️ Clearing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
                # Recreate empty directory
                os.makedirs(cache_dir, exist_ok=True)
                cleared_dirs.append(cache_dir)
                print(f"✅ Cleared: {cache_dir}")
            else:
                print(f"⏭️ Skipped (doesn't exist): {cache_dir}")
        except Exception as e:
            failed_dirs.append({"dir": cache_dir, "error": str(e)})
            print(f"❌ Failed to clear {cache_dir}: {e}")
    
    # Clear app-specific caches
    for cache_dir in app_cache_dirs_to_clear:
        try:
            if cache_dir.exists():
                print(f"🗑️ Clearing app cache directory: {cache_dir}")
                shutil.rmtree(str(cache_dir))
                # Recreate empty directory
                cache_dir.mkdir(exist_ok=True)
                cleared_dirs.append(str(cache_dir))
                print(f"✅ Cleared: {cache_dir}")
            else:
                print(f"⏭️ Skipped (doesn't exist): {cache_dir}")
        except Exception as e:
            failed_dirs.append({"dir": str(cache_dir), "error": str(e)})
            print(f"❌ Failed to clear {cache_dir}: {e}")
    
    # Also clear any CUDA/vLLM compilation caches
    additional_cache_patterns = [
        "/tmp/nvcc_*",  # CUDA compilation temps
        "/tmp/tmpxft_*",  # More CUDA temps
        "/tmp/cuda_*",  # CUDA runtime temps
        "/tmp/vllm_*",  # vLLM temps
    ]
    
    for pattern in additional_cache_patterns:
        try:
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.unlink(path)
                    else:
                        shutil.rmtree(path)
                    cleared_dirs.append(path)
                    print(f"✅ Cleared: {path}")
        except Exception as e:
            failed_dirs.append({"pattern": pattern, "error": str(e)})
            print(f"❌ Failed to clear pattern {pattern}: {e}")
    
    # Summary
    print("\n🧹 Cache clearing completed!")
    print(f"✅ Successfully cleared: {len(cleared_dirs)} directories/files")
    print(f"❌ Failed to clear: {len(failed_dirs)} directories/files")
    print(f"💾 Total space freed: {total_size_before / (1024 * 1024):.2f} MB")
    
    if failed_dirs:
        print("\n⚠️ Failed operations:")
        for failed in failed_dirs:
            print(f"  - {failed}")
    
    print("\n📋 What was preserved:")
    print("  ✅ Application code and files")
    print("  ✅ Model weights and checkpoints")
    print("  ✅ Application directory structure")
    
    print("\n📋 What was cleared:")
    print("  🗑️ Speaker preset caches")
    print("  🗑️ Emotion analysis caches")
    print("  🗑️ Embedding caches")
    print("  🗑️ PyTorch/HuggingFace caches")
    print("  🗑️ CUDA compilation caches")
    print("  🗑️ Output files")
    
    print("\n📋 Next steps:")
    print("  1. Caches will rebuild automatically on next use")
    print("  2. No need to re-download models or re-copy application")
    print("  3. Redeploy with: modal deploy deploy_vllm_indextts_v2.py")
    
    return {
        "status": "completed",
        "cleared_count": len(cleared_dirs),
        "failed_count": len(failed_dirs),
        "space_freed_mb": total_size_before / (1024 * 1024)
    }

@app.function(
    image=image,
    gpu="L40s",  # L40S as requested
    cpu=4.0,
    memory=8192,
    timeout=3600,
    scaledown_window=600,
    volumes={
        PERSISTENT_APP_DIR: app_storage,
        PERSISTENT_CACHE_DIR: cache_storage
    },
    min_containers=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("custom-secret")],
)
@modal.concurrent(max_inputs=100)  # 100 concurrent requests
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """
    Serve the IndexTTS v2 FastAPI application by running python fastapi_webui_v2.py directly.
    """
    import os
    import sys
    from pathlib import Path
    import subprocess
    
    print("🚀 Starting IndexTTS v2 vLLM FastAPI WebUI...")
    
    # ========================================================================
    # STEP 1: Setup Persistent Cache System
    # ========================================================================
    print("\n💾 Configuring persistent cache system...")
    print("   📌 CUDA kernels will compile on FIRST startup (needs GPU)")
    print("   📌 Subsequent startups will reuse cached artifacts from persistent volume\n")
    
    # 1.1: Set cache environment variables (before any Python imports that use them)
    cache_env_vars = {
        "HF_HOME": "/persistent_cache/huggingface",
        "TORCH_HOME": "/persistent_cache/torch",
        "TRANSFORMERS_CACHE": "/persistent_cache/transformers",
        "CUDA_CACHE_PATH": "/persistent_cache/cuda_cache",
        "VLLM_CACHE": "/persistent_cache/vllm_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/persistent_cache/torch_compile_cache",
        "XDG_CACHE_HOME": "/persistent_cache",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1"
    }
    
    print("   Setting environment variables:")
    for key, value in cache_env_vars.items():
        os.environ[key] = value
        print(f"      ✅ {key}={value}")
    
    # 1.2: Create cache directories in persistent volume
    cache_dirs = [
        "/persistent_cache/huggingface",
        "/persistent_cache/torch", 
        "/persistent_cache/transformers",
        "/persistent_cache/cuda_cache",
        "/persistent_cache/vllm_cache",
        "/persistent_cache/torch_compile_cache"
    ]
    
    print("\n   Creating cache directories:")
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"      📁 {cache_dir}")
    
    # 1.3: Create symlinks from standard cache locations to persistent volume
    local_cache_map = {
        "/root/.cache/huggingface": "/persistent_cache/huggingface",
        "/root/.cache/torch": "/persistent_cache/torch",
        "/root/.cache/transformers": "/persistent_cache/transformers",
        "/root/.cache/vllm": "/persistent_cache/vllm_cache"
    }
    
    print("\n   Creating cache symlinks:")
    for local_path, persistent_path in local_cache_map.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        if os.path.exists(local_path):
            if os.path.islink(local_path):
                os.unlink(local_path)
            else:
                import shutil
                shutil.rmtree(local_path)
        
        os.symlink(persistent_path, local_path)
        print(f"      🔗 {local_path} -> {persistent_path}")
    
    # ========================================================================
    # STEP 2: Verify Application and Models
    # ========================================================================
    print("\n📂 Verifying application and models...")
    
    # 2.1: Verify persistent application exists  
    persistent_app_path = Path(PERSISTENT_APP_DIR)
    if not persistent_app_path.exists():
        print("❌ Persistent application not found! Run prepare_model first.")
        raise FileNotFoundError(f"Application not found at {persistent_app_path}")
    
    print(f"   ✅ Application: {persistent_app_path}")
    
    # 2.2: Verify model files exist
    checkpoints_dir = persistent_app_path / "checkpoints"
    if not checkpoints_dir.exists():
        print("❌ Model checkpoints not found!")
        raise FileNotFoundError(f"Checkpoints missing at {checkpoints_dir}")
    
    print(f"   ✅ Checkpoints: {checkpoints_dir}")
    
    # ========================================================================
    # STEP 3: Setup Application Environment
    # ========================================================================
    print("\n🔧 Configuring application environment...")
    
    # 3.1: Change to persistent app directory
    os.chdir(str(persistent_app_path))
    print(f"   📁 Working directory: {os.getcwd()}")
    
    # 3.2: Setup Python path for vLLM worker processes
    os.environ["PYTHONPATH"] = str(persistent_app_path)
    print(f"   🐍 PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    # 3.3: Setup Qwen3-TTS Voice Design model path (use local pre-downloaded model)
    voice_design_model_path = persistent_app_path / "checkpoints" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    if voice_design_model_path.exists():
        os.environ["QWEN3_VOICE_DESIGN_MODEL"] = str(voice_design_model_path)
        print(f"   🎤 QWEN3_VOICE_DESIGN_MODEL: {voice_design_model_path}")
    else:
        print(f"   ⚠️ Voice Design model not found at {voice_design_model_path}, will use HuggingFace download")
    
    # ========================================================================
    # STEP 4: Start FastAPI Server
    # ========================================================================
    print("\n🚀 Starting FastAPI server...")
    
    # Build the command
    cmd = [
        "python", "fastapi_webui_v2.py"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Working dir: {os.getcwd()}\n")
    print("="*80)
    print("🎉 IndexTTS v2 vLLM initialization complete!")
    print("="*80 + "\n")
    
    # Start the FastAPI server (this will keep running)
    subprocess.Popen(" ".join(cmd), shell=True, cwd=str(persistent_app_path))
