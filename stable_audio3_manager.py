"""Stable Audio 3 local checkpoint manager for the FastAPI WebUI.

The Hugging Face Stable Audio 3 repos are gated, so this module prefers
checkpoint folders downloaded ahead of time and does not require HF auth at
runtime when local files are present.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StableAudio3Variant:
    key: str
    repo: str
    local_subdir: str
    label: str
    default_duration: int
    placeholder: str


STABLE_AUDIO3_VARIANTS: Tuple[StableAudio3Variant, ...] = (
    StableAudio3Variant(
        key="medium",
        repo="stabilityai/stable-audio-3-medium",
        local_subdir="medium",
        label="Medium - general audio",
        default_duration=60,
        placeholder=(
            "A dream-like Synthpop instrumental that would accompany a "
            "dream-sequence in a surrealist movie 120 BPM"
        ),
    ),
    StableAudio3Variant(
        key="small-music",
        repo="stabilityai/stable-audio-3-small-music",
        local_subdir="small-music",
        label="Small Music - 0.6B music-focused",
        default_duration=60,
        placeholder=(
            "Cinematic neo-soul groove with electric piano, brushed drums, "
            "walking upright bass, smoky vibe 92 BPM"
        ),
    ),
    StableAudio3Variant(
        key="small-sfx",
        repo="stabilityai/stable-audio-3-small-sfx",
        local_subdir="small-sfx",
        label="Small SFX - 0.6B sound effects",
        default_duration=7,
        placeholder="Chugging train coming into station with horn",
    ),
)

STABLE_AUDIO3_DEFAULT_VARIANT = "medium"
STABLE_AUDIO3_SAMPLERS: Tuple[str, ...] = ("pingpong", "euler", "rk4", "dpmpp")
STABLE_AUDIO3_OUTPUT_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "ogg", "webm"}


@dataclass
class StableAudio3GenerateOptions:
    prompt: str
    variant_key: str = STABLE_AUDIO3_DEFAULT_VARIANT
    negative_prompt: str = ""
    duration: int = 60
    steps: int = 8
    cfg_scale: float = 1.0
    sampler_type: str = "pingpong"
    seed: int = 0
    sigma_max: float = 1.0
    apg_scale: float = 1.0
    duration_padding_sec: float = 6.0
    cut_to_seconds_total: bool = True
    init_audio_path: Optional[str] = None
    init_noise_level: float = 0.9
    inpaint_audio_path: Optional[str] = None
    mask_start_sec: float = 0.0
    mask_end_sec: float = 0.0
    output_dir: Optional[str] = None


@dataclass
class StableAudio3LoadedVariant:
    variant: StableAudio3Variant
    model: Any
    sample_rate: int
    sample_size: int
    max_seconds: int
    source: str
    loaded_at: float


@dataclass
class StableAudio3GenerationResult:
    output_path: str
    variant_key: str
    sample_rate: int
    duration_seconds: int
    elapsed_seconds: float
    seed: int
    source: str


class StableAudio3Manager:
    """Load and run Stable Audio 3 models from local checkpoints."""

    def __init__(
        self,
        checkpoints_root: Path | str,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        allow_hf_download: Optional[bool] = None,
    ) -> None:
        self.checkpoints_root = Path(checkpoints_root)
        self.device = device or os.environ.get("STABLE_AUDIO_DEVICE", "cuda")
        self.dtype = (dtype or os.environ.get("STABLE_AUDIO_DTYPE", "float16")).lower()
        self.allow_hf_download = (
            _env_flag("STABLE_AUDIO_ALLOW_HF_DOWNLOAD", False)
            if allow_hf_download is None
            else allow_hf_download
        )
        self._loaded: Dict[str, StableAudio3LoadedVariant] = {}
        self._load_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._variants = {variant.key: variant for variant in STABLE_AUDIO3_VARIANTS}

    def normalize_variant_key(self, value: Optional[str]) -> str:
        key = (value or STABLE_AUDIO3_DEFAULT_VARIANT).strip().lower()
        return key if key in self._variants else STABLE_AUDIO3_DEFAULT_VARIANT

    def normalize_sampler(self, value: Optional[str]) -> str:
        sampler = (value or "pingpong").strip().lower()
        return sampler if sampler in STABLE_AUDIO3_SAMPLERS else "pingpong"

    def checkpoint_dir(self, variant_key: str) -> Path:
        variant = self._variants[self.normalize_variant_key(variant_key)]
        return self.checkpoints_root / variant.local_subdir

    def checkpoint_status(self, variant_key: str) -> Dict[str, Any]:
        directory = self.checkpoint_dir(variant_key)
        ckpt_path = self._find_checkpoint_file(directory)
        config_path = directory / "model_config.json"
        text_encoder_path = directory / "t5gemma-b-b-ul2"
        return {
            "path": str(directory),
            "config_exists": config_path.exists(),
            "checkpoint_exists": ckpt_path is not None,
            "checkpoint_path": str(ckpt_path) if ckpt_path else None,
            "text_encoder_exists": text_encoder_path.exists(),
            "ready": config_path.exists() and ckpt_path is not None,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for variant in STABLE_AUDIO3_VARIANTS:
            checkpoint = self.checkpoint_status(variant.key)
            loaded = self._loaded.get(variant.key)
            max_seconds = loaded.max_seconds if loaded else None
            models.append(
                {
                    "key": variant.key,
                    "repo": variant.repo,
                    "label": variant.label,
                    "default_duration": variant.default_duration,
                    "placeholder": variant.placeholder,
                    "checkpoint": checkpoint,
                    "loaded": loaded is not None,
                    "sample_rate": loaded.sample_rate if loaded else None,
                    "sample_size": loaded.sample_size if loaded else None,
                    "max_seconds": max_seconds,
                }
            )
        return models

    def status(self) -> Dict[str, Any]:
        missing = self._missing_packages()
        return {
            "available": not missing,
            "missing_packages": missing,
            "default_variant": STABLE_AUDIO3_DEFAULT_VARIANT,
            "checkpoints_root": str(self.checkpoints_root),
            "device": self.device,
            "dtype": self.dtype,
            "allow_hf_download": self.allow_hf_download,
            "samplers": list(STABLE_AUDIO3_SAMPLERS),
            "loaded_models": list(self._loaded.keys()),
            "models": self.list_models(),
        }

    def unload(self, variant_key: Optional[str] = None) -> List[str]:
        import gc

        removed: List[str] = []
        with self._load_lock:
            if variant_key:
                key = self.normalize_variant_key(variant_key)
                if key in self._loaded:
                    del self._loaded[key]
                    removed.append(key)
            else:
                removed = list(self._loaded.keys())
                self._loaded.clear()

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return removed

    def load_variant(self, variant_key: Optional[str]) -> StableAudio3LoadedVariant:
        key = self.normalize_variant_key(variant_key)
        with self._load_lock:
            if key in self._loaded:
                return self._loaded[key]

            self._raise_for_missing_packages()
            self._raise_for_unsupported_stable_audio_tools()
            variant = self._variants[key]
            local_status = self.checkpoint_status(key)
            if local_status["ready"]:
                loaded = self._load_local_variant(variant)
            elif self.allow_hf_download:
                loaded = self._load_hf_variant(variant)
            else:
                raise RuntimeError(
                    "Stable Audio 3 checkpoint is not ready. Download "
                    f"{variant.repo} to {self.checkpoint_dir(key)} first, or set "
                    "STABLE_AUDIO_ALLOW_HF_DOWNLOAD=1 to allow runtime HF downloads."
                )

            self._loaded[key] = loaded
            return loaded

    def generate(self, options: StableAudio3GenerateOptions) -> StableAudio3GenerationResult:
        prompt = (options.prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")

        with self._generation_lock:
            loaded = self.load_variant(options.variant_key)
            return self._generate_locked(loaded, options)

    def generate_many(
        self,
        options_list: List[StableAudio3GenerateOptions],
    ) -> List[StableAudio3GenerationResult]:
        """Generate multiple clips in one locked backend job.

        Stable Audio / PyTorch CUDA inference is not safe to call concurrently
        against the same loaded model from multiple Python threads. Keep the
        request batched at the API layer while serializing model access here.
        """
        if not options_list:
            return []
        with self._generation_lock:
            loaded_by_key: Dict[str, StableAudio3LoadedVariant] = {}
            results: List[StableAudio3GenerationResult] = []
            for options in options_list:
                prompt = (options.prompt or "").strip()
                if not prompt:
                    raise ValueError("Prompt is required.")
                variant_key = self.normalize_variant_key(options.variant_key)
                loaded = loaded_by_key.get(variant_key)
                if loaded is None:
                    loaded = self.load_variant(variant_key)
                    loaded_by_key[variant_key] = loaded
                results.append(self._generate_locked(loaded, options))
            return results

    def _generate_locked(
        self,
        loaded: StableAudio3LoadedVariant,
        options: StableAudio3GenerateOptions,
    ) -> StableAudio3GenerationResult:
        import soundfile as sf
        import torch
        import torchaudio
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

        duration = max(1, min(int(options.duration), loaded.max_seconds))
        steps = max(1, min(int(options.steps), 500))
        sampler = self.normalize_sampler(options.sampler_type)
        prompt = options.prompt.strip()
        negative_prompt = (options.negative_prompt or "").strip()

        conditioning = [{"prompt": prompt, "seconds_total": int(duration)}]
        negative_conditioning = (
            [{"prompt": negative_prompt, "seconds_total": int(duration)}]
            if negative_prompt
            else None
        )

        model_dtype = next(loaded.model.parameters()).dtype

        def prep_audio(path: Optional[str]) -> Optional[Tuple[int, Any]]:
            if not path:
                return None
            waveform, sample_rate = torchaudio.load(path)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.float()
            if int(sample_rate) != loaded.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    int(sample_rate),
                    loaded.sample_rate,
                )
            return loaded.sample_rate, waveform.to(model_dtype)

        init_audio = prep_audio(options.init_audio_path)
        inpaint_audio = prep_audio(options.inpaint_audio_path)
        mask_start = max(0.0, float(options.mask_start_sec))
        mask_end = min(float(duration), float(options.mask_end_sec))
        use_mask = inpaint_audio is not None and mask_end > mask_start
        seed = int(options.seed) if options.seed and int(options.seed) > 0 else -1

        gen_kwargs: Dict[str, Any] = {
            "steps": steps,
            "cfg_scale": float(options.cfg_scale),
            "conditioning": conditioning,
            "negative_conditioning": negative_conditioning,
            "sample_size": loaded.sample_size,
            "sampler_type": sampler,
            "seed": seed,
            "device": self.device,
            "sigma_max": float(options.sigma_max),
            "apg_scale": float(options.apg_scale),
            "duration_padding_sec": float(options.duration_padding_sec),
        }
        if init_audio is not None:
            gen_kwargs["init_audio"] = init_audio
            gen_kwargs["init_noise_level"] = float(options.init_noise_level)
        if inpaint_audio is not None:
            gen_kwargs["inpaint_audio"] = inpaint_audio
        if use_mask:
            gen_kwargs["inpaint_mask_start_seconds"] = mask_start
            gen_kwargs["inpaint_mask_end_seconds"] = mask_end

        start_time = time.time()
        with torch.inference_mode():
            output = generate_diffusion_cond_inpaint(loaded.model, **gen_kwargs)

        audio = rearrange(output, "b d n -> d (b n)")
        audio = (
            audio.to(torch.float32)
            .div(torch.max(torch.abs(audio)).clamp(min=1e-9))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        if options.cut_to_seconds_total:
            audio = audio[:, : int(duration) * loaded.sample_rate]

        output_dir = Path(options.output_dir or "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"stable_audio_{loaded.variant.key}_{uuid.uuid4().hex}.wav"
        sf.write(str(output_path), audio.numpy().T, loaded.sample_rate, subtype="PCM_16")

        return StableAudio3GenerationResult(
            output_path=str(output_path),
            variant_key=loaded.variant.key,
            sample_rate=loaded.sample_rate,
            duration_seconds=duration,
            elapsed_seconds=time.time() - start_time,
            seed=seed,
            source=loaded.source,
        )

    def _load_local_variant(self, variant: StableAudio3Variant) -> StableAudio3LoadedVariant:
        import torch
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        directory = self.checkpoint_dir(variant.key)
        config_path = directory / "model_config.json"
        checkpoint_path = self._find_checkpoint_file(directory)
        if checkpoint_path is None:
            raise FileNotFoundError(f"No model.safetensors or model.ckpt found in {directory}")

        with open(config_path, "r", encoding="utf-8") as fh:
            raw_config = json.load(fh)
        model_config = self._patch_config_for_local_assets(raw_config, directory, variant)

        model = create_model_from_config(model_config)
        state_dict = load_ckpt_state_dict(str(checkpoint_path))
        state_dict = self._unwrap_state_dict(state_dict, str(model_config.get("model_type") or ""))
        model.load_state_dict(state_dict)
        model.to(self.device).eval().requires_grad_(False)
        self._apply_dtype(model, torch)

        sample_rate = int(model_config["sample_rate"])
        sample_size = int(model_config["sample_size"])
        return StableAudio3LoadedVariant(
            variant=variant,
            model=model,
            sample_rate=sample_rate,
            sample_size=sample_size,
            max_seconds=max(1, sample_size // sample_rate),
            source=str(directory),
            loaded_at=time.time(),
        )

    def _load_hf_variant(self, variant: StableAudio3Variant) -> StableAudio3LoadedVariant:
        import torch
        from stable_audio_tools import get_pretrained_model

        model, model_config = get_pretrained_model(variant.repo)
        model.to(self.device).eval().requires_grad_(False)
        self._apply_dtype(model, torch)
        sample_rate = int(model_config["sample_rate"])
        sample_size = int(model_config["sample_size"])
        return StableAudio3LoadedVariant(
            variant=variant,
            model=model,
            sample_rate=sample_rate,
            sample_size=sample_size,
            max_seconds=max(1, sample_size // sample_rate),
            source=variant.repo,
            loaded_at=time.time(),
        )

    def _apply_dtype(self, model: Any, torch_module: Any) -> None:
        if self.dtype in {"fp16", "float16", "half"}:
            model.to(torch_module.float16)
        elif self.dtype in {"bf16", "bfloat16"}:
            model.to(torch_module.bfloat16)

    def _patch_config_for_local_assets(
        self,
        model_config: Dict[str, Any],
        directory: Path,
        variant: StableAudio3Variant,
    ) -> Dict[str, Any]:
        patched = copy.deepcopy(model_config)
        text_encoder_subdir = directory / "t5gemma-b-b-ul2"

        def patch_obj(value: Any) -> Any:
            if isinstance(value, dict):
                repo_id = value.get("repo_id")
                subfolder = value.get("subfolder")
                model_name = value.get("model_name")

                if repo_id == variant.repo and subfolder and (directory / subfolder).exists():
                    value["model_path"] = str(directory)
                    value.pop("repo_id", None)
                elif (
                    model_name == "google/t5gemma-b-b-ul2"
                    and text_encoder_subdir.exists()
                    and not value.get("model_path")
                ):
                    if subfolder and (directory / subfolder).exists():
                        value["model_path"] = str(directory)
                    else:
                        value["model_path"] = str(text_encoder_subdir)
                    value.pop("repo_id", None)

                for path_key in ("ckpt_path", "pretransform_ckpt_path"):
                    raw_path = value.get(path_key)
                    if isinstance(raw_path, str) and raw_path and not os.path.isabs(raw_path):
                        candidate = directory / raw_path
                        if candidate.exists():
                            value[path_key] = str(candidate)

                for child_key, child_value in list(value.items()):
                    value[child_key] = patch_obj(child_value)
            elif isinstance(value, list):
                return [patch_obj(item) for item in value]
            return value

        return patch_obj(patched)

    @staticmethod
    def _find_checkpoint_file(directory: Path) -> Optional[Path]:
        for name in ("model.safetensors", "model.ckpt"):
            path = directory / name
            if path.exists():
                return path
        return None

    @staticmethod
    def _unwrap_state_dict(state_dict: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        prefixes = {
            "diffusion_cond": "diffusion.",
            "diffusion_cond_inpaint": "diffusion.",
            "diffusion_uncond": "diffusion.",
            "diffusion_autoencoder": "diffusion.",
            "autoencoder": "autoencoder.",
            "lm": "lm.",
            "clap": "clap.",
        }
        prefix = prefixes.get(model_type)
        if not prefix or not any(key.startswith(prefix) for key in state_dict):
            return state_dict

        ema_prefix = prefix.replace(".", "_ema.ema_model.")
        has_ema = any(key.startswith(ema_prefix) for key in state_dict)
        ema_wraps_whole_model = model_type in {"autoencoder"}
        unwrapped: Dict[str, Any] = {}

        if has_ema:
            for key, value in state_dict.items():
                if key.startswith(ema_prefix):
                    suffix = key[len(ema_prefix):]
                    unwrapped[suffix if ema_wraps_whole_model else f"model.{suffix}"] = value
            if not ema_wraps_whole_model:
                for key, value in state_dict.items():
                    if key.startswith(prefix + "conditioner.") or key.startswith(prefix + "pretransform."):
                        unwrapped[key[len(prefix):]] = value
        else:
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    unwrapped[key[len(prefix):]] = value

        return unwrapped or state_dict

    def _missing_packages(self) -> List[str]:
        required = {
            "stable_audio_tools": "stable-audio-tools",
            "torch": "torch",
            "torchaudio": "torchaudio",
            "einops": "einops",
            "soundfile": "soundfile",
            "safetensors": "safetensors",
            "transformers": "transformers",
        }
        missing = []
        for module_name, package_name in required.items():
            if importlib.util.find_spec(module_name) is None:
                missing.append(package_name)
        return missing

    def _raise_for_missing_packages(self) -> None:
        missing = self._missing_packages()
        if missing:
            raise RuntimeError(
                "Stable Audio 3 dependencies are missing: "
                + ", ".join(missing)
                + ". Install the README_EN Stable Audio quickstart dependencies first."
            )

    def _raise_for_unsupported_stable_audio_tools(self) -> None:
        import inspect

        try:
            from stable_audio_tools.models.transformer import TransformerBlock
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect stable-audio-tools: {exc}") from exc

        signature = inspect.signature(TransformerBlock.__init__)
        if "local_add_cond_dim" not in signature.parameters:
            raise RuntimeError(
                "Installed stable-audio-tools is too old for Stable Audio 3 Medium "
                "checkpoints: TransformerBlock lacks `local_add_cond_dim`. Reinstall "
                "the current GitHub version without dependencies, for example: "
                "`pip install -U --force-reinstall --no-deps --ignore-requires-python "
                "git+https://github.com/Stability-AI/stable-audio-tools.git`."
            )
