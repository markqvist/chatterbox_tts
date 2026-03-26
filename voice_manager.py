"""
Voice management and conditionals caching for Chatterbox TTS.

Handles:
- Reference audio file storage
- Voice configuration persistence
- Conditionals caching for fast repeated generation
- Model-agnostic voice configuration
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict

import librosa
import soundfile as sf
import torch
from loguru import logger

from config import CONFIG
from models import VoiceConfig, VoiceInfo


@dataclass
class VoiceMetadata:
    """Extended voice metadata including audio properties."""
    voice_id: str
    model_type: str
    config_path: Path
    audio_path: Path
    duration_seconds: float
    sample_rate: int
    channels: int


class VoiceManager:
    """
    Manages voice configurations, reference audio, and conditionals caching.
    
    Each voice consists of:
    1. Reference audio file (wav, mp3, flac, ogg)
    2. Voice configuration JSON (generation parameters)
    3. Optional: Cached conditionals (computed speaker embeddings)
    """
    
    def __init__(self):
        self.voices_dir = CONFIG.voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory conditionals cache: voice_id -> Conditionals object
        self._conditionals_cache: dict[str, Any] = {}
        
        # Supported audio formats
        self._supported_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    
    def _get_voice_config_path(self, voice_id: str) -> Path:
        """Get path to voice configuration file."""
        return self.voices_dir / f"{voice_id}.json"
    
    def _get_voice_audio_path(self, voice_id: str) -> Path | None:
        """Get path to voice reference audio file (searches for any supported extension)."""
        for ext in self._supported_extensions:
            path = self.voices_dir / f"{voice_id}{ext}"
            if path.exists():
                return path
        return None
    
    def _sanitize_voice_id(self, voice_id: str) -> str:
        """Sanitize voice ID for filesystem safety."""
        # Replace unsafe characters with underscore
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', voice_id)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Strip leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or "voice"
    
    def _analyze_audio(self, audio_path: Path) -> dict[str, Any]:
        """Analyze audio file and return metadata."""
        try:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
            
            # Handle mono vs stereo
            if audio.ndim == 1:
                duration = len(audio) / sr
                channels = 1
            else:
                duration = audio.shape[1] / sr
                channels = audio.shape[0]
            
            return {
                "duration_seconds": float(duration),
                "sample_rate": int(sr),
                "channels": channels,
            }
        except Exception as e:
            logger.warning(f"Could not analyze audio {audio_path}: {e}")
            return {
                "duration_seconds": 0.0,
                "sample_rate": 0,
                "channels": 0,
            }
    
    def _convert_to_wav(self, audio_path: Path) -> Path:
        """
        Convert audio to WAV format if needed.
        Returns path to WAV file (may be same as input if already WAV).
        """
        if audio_path.suffix.lower() == ".wav":
            return audio_path
        
        # Convert to WAV
        try:
            audio, sr = librosa.load(str(audio_path), sr=None)
            wav_path = audio_path.with_suffix(".wav")
            sf.write(str(wav_path), audio, sr)
            logger.info(f"Converted {audio_path.name} to WAV format")
            return wav_path
        except Exception as e:
            logger.error(f"Failed to convert {audio_path} to WAV: {e}")
            raise
    
    def create_voice(
        self,
        voice_id: str,
        audio_source: Path | bytes,
        config: VoiceConfig | None = None,
    ) -> VoiceMetadata:
        """
        Create a new voice from reference audio.
        
        Args:
            voice_id: Unique voice identifier
            audio_source: Path to audio file or raw bytes
            config: Voice configuration (uses defaults if None)
            
        Returns:
            VoiceMetadata for the created voice
        """
        voice_id = self._sanitize_voice_id(voice_id)
        
        # Check if voice already exists
        if self.voice_exists(voice_id):
            raise ValueError(f"Voice '{voice_id}' already exists")
        
        # Handle audio source
        if isinstance(audio_source, bytes):
            # Save bytes to temp file first to validate
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_source)
                temp_path = Path(f.name)
            audio_path = temp_path
        else:
            audio_path = Path(audio_source)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Convert to WAV if needed
        final_audio_path = self._convert_to_wav(audio_path)
        
        # If we used a temp file, clean it up
        if isinstance(audio_source, bytes) and temp_path != final_audio_path:
            temp_path.unlink(missing_ok=True)
        
        # Move/copy to voices directory with voice_id name
        dest_audio_path = self.voices_dir / f"{voice_id}.wav"
        if final_audio_path != dest_audio_path:
            shutil.copy2(str(final_audio_path), str(dest_audio_path))
            if isinstance(audio_source, bytes):
                final_audio_path.unlink(missing_ok=True)
        
        # Analyze audio
        audio_meta = self._analyze_audio(dest_audio_path)
        
        # Validate duration (Chatterbox requires > 5 seconds)
        if audio_meta["duration_seconds"] < 5.0:
            dest_audio_path.unlink(missing_ok=True)
            raise ValueError(
                f"Reference audio must be at least 5 seconds long. "
                f"Got {audio_meta['duration_seconds']:.1f}s"
            )
        
        # Create default config if none provided
        if config is None:
            config = VoiceConfig(
                voice_id=voice_id,
                reference_audio=str(dest_audio_path),
            )
        else:
            config.voice_id = voice_id
            config.reference_audio = str(dest_audio_path)
        
        # Save voice configuration
        config_path = self._save_voice_config(config)
        
        logger.info(f"Created voice '{voice_id}' with {audio_meta['duration_seconds']:.1f}s of audio")
        
        return VoiceMetadata(
            voice_id=voice_id,
            model_type=config.model_type,
            config_path=config_path,
            audio_path=dest_audio_path,
            duration_seconds=audio_meta["duration_seconds"],
            sample_rate=audio_meta["sample_rate"],
            channels=audio_meta["channels"],
        )
    
    def _save_voice_config(self, config: VoiceConfig) -> Path:
        """Save voice configuration to JSON file."""
        config_path = self._get_voice_config_path(config.voice_id)
        
        # Convert to dict and save
        config_dict = config.model_dump()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return config_path
    
    def load_voice_config(self, voice_id: str) -> VoiceConfig:
        """Load voice configuration from JSON file."""
        config_path = self._get_voice_config_path(voice_id)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Voice configuration not found: {voice_id}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return VoiceConfig(**config_dict)
    
    def update_voice_config(self, voice_id: str, **updates) -> VoiceConfig:
        """Update voice configuration with new values."""
        config = self.load_voice_config(voice_id)
        
        # Update fields
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Save updated config
        self._save_voice_config(config)
        
        # Clear conditionals cache since parameters changed
        self.clear_conditionals_cache(voice_id)
        
        return config
    
    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a voice and all associated files.
        
        Returns:
            True if voice was deleted, False if not found
        """
        config_path = self._get_voice_config_path(voice_id)
        audio_path = self._get_voice_audio_path(voice_id)
        
        deleted = False
        
        if config_path.exists():
            config_path.unlink()
            deleted = True
        
        if audio_path and audio_path.exists():
            audio_path.unlink()
            deleted = True
        
        # Clear from conditionals cache
        self.clear_conditionals_cache(voice_id)
        
        if deleted:
            logger.info(f"Deleted voice '{voice_id}'")
        
        return deleted
    
    def list_voices(self) -> list[VoiceInfo]:
        """List all available voices with metadata."""
        voices = []
        
        for config_file in self.voices_dir.glob("*.json"):
            voice_id = config_file.stem
            try:
                config = self.load_voice_config(voice_id)
                audio_path = self._get_voice_audio_path(voice_id)
                
                if audio_path:
                    audio_meta = self._analyze_audio(audio_path)
                    voices.append(VoiceInfo(
                        voice_id=voice_id,
                        model_type=config.model_type,
                        reference_audio=str(audio_path),
                        duration_seconds=audio_meta["duration_seconds"],
                        sample_rate=audio_meta["sample_rate"],
                    ))
            except Exception as e:
                logger.warning(f"Could not load voice '{voice_id}': {e}")
        
        return sorted(voices, key=lambda v: v.voice_id)
    
    def voice_exists(self, voice_id: str) -> bool:
        """Check if a voice exists."""
        return self._get_voice_config_path(voice_id).exists()
    
    def get_voice_audio_path(self, voice_id: str) -> Path:
        """Get the path to a voice's reference audio file."""
        audio_path = self._get_voice_audio_path(voice_id)
        if not audio_path:
            raise FileNotFoundError(f"Reference audio not found for voice: {voice_id}")
        return audio_path
    
    # =====================================================================
    # Conditionals Caching
    # =====================================================================
    
    def get_cached_conditionals(self, voice_id: str) -> Any | None:
        """
        Get cached conditionals for a voice.
        
        Returns:
            Conditionals object if cached, None otherwise
        """
        return self._conditionals_cache.get(voice_id)
    
    def cache_conditionals(self, voice_id: str, conditionals: Any):
        """Cache conditionals for a voice."""
        self._conditionals_cache[voice_id] = conditionals
        logger.debug(f"Cached conditionals for voice '{voice_id}'")
    
    def clear_conditionals_cache(self, voice_id: str | None = None):
        """
        Clear conditionals cache.
        
        Args:
            voice_id: Specific voice to clear, or None to clear all
        """
        if voice_id:
            self._conditionals_cache.pop(voice_id, None)
            logger.debug(f"Cleared conditionals cache for voice '{voice_id}'")
        else:
            count = len(self._conditionals_cache)
            self._conditionals_cache.clear()
            logger.info(f"Cleared all conditionals caches ({count} voices)")
    
    def is_conditionals_cached(self, voice_id: str) -> bool:
        """Check if conditionals are cached for a voice."""
        return voice_id in self._conditionals_cache
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get conditionals cache statistics."""
        return {
            "cached_voices": list(self._conditionals_cache.keys()),
            "cache_size": len(self._conditionals_cache),
        }


# Global voice manager instance
VOICE_MANAGER = VoiceManager()
