"""
Qwen3-TTS Voice Design Manager
Provides interface for Voice Design feature with Speaker Preset integration.
"""

from typing import Optional, Dict, Any
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch


@dataclass
class Qwen3TTSConfig:
    """Configuration for Qwen3-TTS Voice Design model."""
    voice_design_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    use_flash_attention: bool = True
    max_new_tokens: int = 2048
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05


@dataclass
class DesignedVoiceResult:
    """Result from voice design generation, can be saved to preset."""
    audio_waveform: np.ndarray
    sample_rate: int
    text: str
    voice_description: str
    language: str


class Qwen3VoiceDesignManager:
    """Manager class for Qwen3-TTS Voice Design feature."""
    
    SUPPORTED_LANGUAGES = [
        "Auto", "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
    ]
    
    def __init__(self, config: Qwen3TTSConfig, preset_manager=None):
        """
        Initialize Voice Design Manager.
        
        Args:
            config: Qwen3TTS configuration
            preset_manager: SpeakerPresetManager instance for saving voices as presets
        """
        self.config = config
        self.preset_manager = preset_manager
        self._voice_design_model = None
        self._last_generated: Optional[DesignedVoiceResult] = None
        
    @property
    def voice_design_model(self):
        """Lazy load voice design model."""
        if self._voice_design_model is None:
            from qwen_tts import Qwen3TTSModel
            attn_impl = "flash_attention_2" if self.config.use_flash_attention else None
            print(f"[Qwen3-TTS] Loading VoiceDesign model from {self.config.voice_design_model_path}...")
            self._voice_design_model = Qwen3TTSModel.from_pretrained(
                self.config.voice_design_model_path,
                device_map=self.config.device,
                dtype=self.config.dtype,
                attn_implementation=attn_impl,
            )
            print(f"[Qwen3-TTS] VoiceDesign model loaded successfully.")
        return self._voice_design_model
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._voice_design_model is not None
    
    def generate_voice_design(
        self,
        text: str,
        voice_description: str,
        language: str = "Auto",
        **kwargs
    ) -> DesignedVoiceResult:
        """
        Generate speech with voice design.
        
        Args:
            text: Text to synthesize
            voice_description: Natural language description of desired voice
            language: Target language
            **kwargs: Additional generation parameters
            
        Returns:
            DesignedVoiceResult containing audio and metadata
        """
        gen_kwargs = self._build_gen_kwargs(**kwargs)
        
        print(f"[Qwen3-TTS] Generating voice design: lang={language}, desc_len={len(voice_description)}, text_len={len(text)}")
        
        wavs, sr = self.voice_design_model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            **gen_kwargs,
        )
        
        result = DesignedVoiceResult(
            audio_waveform=wavs[0],
            sample_rate=sr,
            text=text.strip(),
            voice_description=voice_description.strip(),
            language=language,
        )
        
        # Store for potential save-to-preset
        self._last_generated = result
        
        print(f"[Qwen3-TTS] Generated audio: {len(wavs[0]) / sr:.2f}s at {sr}Hz")
        
        return result
    
    def save_to_preset(
        self,
        preset_name: str,
        result: Optional[DesignedVoiceResult] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a designed voice result directly to Speaker Preset Manager.
        
        Args:
            preset_name: Name for the new speaker preset
            result: DesignedVoiceResult to save (uses last generated if None)
            description: Optional description override for the preset
            
        Returns:
            Dict with preset info
        """
        if self.preset_manager is None:
            raise ValueError("Speaker preset manager not available. Please ensure IndexTTS is fully initialized before saving presets.")
        
        if result is None:
            result = self._last_generated
            
        if result is None:
            raise ValueError("No voice design result available to save. Generate one first.")
        
        # Create temporary file with the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, result.audio_waveform, result.sample_rate)
            temp_path = tmp.name
        
        try:
            # Use preset manager to create the preset from the audio file
            preset_description = description or f"Voice Design: {result.voice_description[:100]}"
            
            # Call SpeakerPresetManager.add_speaker_preset()
            success = self.preset_manager.add_speaker_preset(
                preset_name=preset_name,
                audio_path=temp_path,
                description=preset_description,
            )
            
            if not success:
                raise ValueError(f"Failed to add preset '{preset_name}' to speaker preset manager")
            
            print(f"[Qwen3-TTS] Saved preset '{preset_name}' from voice design")
            
            return {
                "success": True,
                "preset_name": preset_name,
                "description": preset_description,
                "source": "voice_design",
            }
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _build_gen_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Build generation kwargs with defaults from config."""
        defaults = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
        }
        defaults.update({k: v for k, v in kwargs.items() if v is not None})
        return defaults
    
    def get_last_generated(self) -> Optional[DesignedVoiceResult]:
        """Get the last generated voice design result."""
        return self._last_generated
    
    def clear_last_generated(self):
        """Clear the cached last generated result."""
        self._last_generated = None
