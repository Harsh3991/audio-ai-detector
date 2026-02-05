"""
Audio Preprocessing Utilities

This module handles all audio preprocessing operations required before
feeding audio data to ML models. Standard preprocessing includes:
- Converting to mono (single channel)
- Resampling to 16kHz (standard for speech models)
- Normalization and format conversion

Why preprocessing is necessary:
1. Models are trained on specific sample rates (typically 16kHz for speech)
2. Mono audio reduces computational load and model complexity
3. Consistent format ensures reliable inference across different input files
"""

import os
import tempfile
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple


def preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    to_mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file to model-ready format.
    
    Args:
        audio_path: Path to audio file (.wav, .mp3, .m4a, etc.)
        target_sr: Target sample rate in Hz (default: 16000)
        to_mono: Convert to mono if True (default: True)
    
    Returns:
        Tuple of (audio_array, sample_rate)
        - audio_array: Preprocessed audio as numpy array
        - sample_rate: Sample rate after preprocessing (should equal target_sr)
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading or processing fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load audio file (librosa automatically resamples if needed)
        audio, sr = librosa.load(
            audio_path,
            sr=target_sr,  # Resample to target sample rate
            mono=to_mono   # Convert to mono if needed
        )
        
        # Normalize audio to [-1, 1] range (if not already)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        return audio, sr
    
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess audio: {str(e)}")


def save_preprocessed_audio(
    audio: np.ndarray,
    sample_rate: int,
    output_dir: str = None
) -> str:
    """
    Save preprocessed audio to a temporary WAV file.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
        output_dir: Directory to save temp file (default: system temp dir)
    
    Returns:
        Path to the saved temporary audio file
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    # Create temporary file with .wav extension
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
        dir=output_dir
    )
    temp_path = temp_file.name
    temp_file.close()
    
    # Save audio to temp file
    sf.write(temp_path, audio, sample_rate)
    
    return temp_path


def cleanup_temp_file(file_path: str) -> None:
    """
    Safely delete a temporary file.
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        # Log but don't raise - cleanup failures shouldn't break the app
        print(f"Warning: Failed to delete temp file {file_path}: {str(e)}")


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
    """
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {str(e)}")


def validate_audio_file(file_path: str, max_duration: float = 300.0) -> Tuple[bool, str]:
    """
    Validate if audio file is acceptable for processing.
    
    Args:
        file_path: Path to audio file
        max_duration: Maximum allowed duration in seconds (default: 300s = 5min)
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if file is valid, False otherwise
        - error_message: Description of validation error (empty if valid)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size (reject files > 50MB)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 50:
        return False, f"File too large: {file_size_mb:.1f}MB (max 50MB)"
    
    # Check duration
    try:
        duration = get_audio_duration(file_path)
        if duration > max_duration:
            return False, f"Audio too long: {duration:.1f}s (max {max_duration}s)"
    except Exception as e:
        return False, f"Failed to read audio file: {str(e)}"
    
    return True, ""
