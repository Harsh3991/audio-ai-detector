"""
FastAPI Audio Analyzer Backend

This is the main entry point for the AI-powered audio analysis API.
Provides endpoints for analyzing audio files to detect language and
determine if voice is AI-generated or human.

API Endpoints:
- POST /analyze-audio/ - Analyze uploaded audio file
- GET / - Health check / API info
- GET /supported-languages/ - List supported languages
"""

import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

from utils import (
    preprocess_audio,
    cleanup_temp_file,
    validate_audio_file
)
from models import AudioAnalyzer


# Initialize FastAPI app
app = FastAPI(
    title="AI Audio Analyzer",
    description="Analyze audio files for language detection and AI voice detection",
    version="1.0.0"
)

# Add CORS middleware (allow requests from any origin for demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance (loaded once at startup)
analyzer: AudioAnalyzer = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize ML models when the server starts.
    This happens once and models are reused for all requests.
    """
    global analyzer
    print("Starting up Audio Analyzer API...")
    print("Loading ML models (this may take a minute)...")
    
    try:
        analyzer = AudioAnalyzer()
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load models: {str(e)}")
        print("Server will start but /analyze-audio/ endpoint will fail")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on server shutdown.
    """
    print("Shutting down Audio Analyzer API...")


# Pydantic models for API responses
class LanguageInfo(BaseModel):
    language_code: str
    language_name: str


class AIVoiceScore(BaseModel):
    label: str
    score: float


class AnalysisResponse(BaseModel):
    language_detected: str
    language_name: str
    language_confidence: float
    ai_voice_result: List[AIVoiceScore]
    device_used: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


# API Routes

@app.get("/")
async def root():
    """
    Root endpoint - API information and health check.
    """
    return {
        "name": "AI Audio Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/analyze-audio/ (POST)",
            "supported_languages": "/supported-languages/ (GET)",
            "health": "/health/ (GET)"
        }
    }


@app.get("/health/")
async def health_check():
    """
    Health check endpoint.
    """
    model_status = "loaded" if analyzer is not None else "not loaded"
    
    return {
        "status": "healthy",
        "models": model_status
    }


@app.get("/supported-languages/", response_model=List[LanguageInfo])
async def get_supported_languages():
    """
    Get list of supported languages for analysis.
    """
    supported = [
        {"language_code": "en", "language_name": "English"},
        {"language_code": "hi", "language_name": "Hindi"},
        {"language_code": "te", "language_name": "Telugu"},
        {"language_code": "ta", "language_name": "Tamil"},
        {"language_code": "ml", "language_name": "Malayalam"}
    ]
    return supported


@app.post("/analyze-audio/", response_model=AnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a)")
):
    """
    Analyze uploaded audio file.
    
    Performs:
    - Language detection (English, Hindi, Telugu, Tamil, Malayalam)
    - AI voice detection (AI-generated vs Human voice)
    
    Args:
        file: Audio file upload (supported formats: .wav, .mp3, .m4a, .flac)
    
    Returns:
        AnalysisResponse with:
            - language_detected: Detected language code
            - language_name: Full language name
            - language_confidence: Detection confidence
            - ai_voice_result: AI vs REAL scores
            - device_used: CPU or CUDA
    
    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Check if models are loaded
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Please wait for server initialization."
        )
    
    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file to temporary location
    temp_input_path = None
    temp_processed_path = None
    
    try:
        # Create temp file for upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_input_path = temp_file.name
            # Read and write uploaded file
            content = await file.read()
            temp_file.write(content)
        
        # Validate audio file
        is_valid, error_msg = validate_audio_file(temp_input_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {error_msg}")
        
        # Preprocess audio (convert to mono, resample to 16kHz)
        print(f"Processing audio file: {file.filename}")
        audio_data, sample_rate = preprocess_audio(temp_input_path)
        
        # Run analysis
        print("Running audio analysis...")
        result = analyzer.analyze(audio_data, sample_rate)
        
        # Format response
        response = {
            "language_detected": result["language_detected"],
            "language_name": result["language_name"],
            "language_confidence": result["language_confidence"],
            "ai_voice_result": result["ai_voice_result"],
            "device_used": result["device_used"]
        }
        
        print(f"Analysis complete: {result['language_name']} detected")
        return JSONResponse(content=response)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Handle unexpected errors
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    
    finally:
        # Cleanup temporary files
        if temp_input_path:
            cleanup_temp_file(temp_input_path)
        if temp_processed_path:
            cleanup_temp_file(temp_processed_path)


@app.get("/info/")
async def get_info():
    """
    Get detailed information about the API and models.
    """
    return {
        "api_version": "1.0.0",
        "models": {
            "language_detection": {
                "model": "openai/whisper-small",
                "task": "automatic-speech-recognition",
                "supported_languages": ["en", "hi", "te", "ta", "ml"]
            },
            "ai_voice_detection": {
                "model": "heuristic-based (placeholder)",
                "note": "Production systems require specialized training",
                "outputs": ["AI", "REAL"]
            }
        },
        "limitations": [
            "AI voice detection is a placeholder implementation",
            "Maximum file size: 50MB",
            "Maximum duration: 5 minutes",
            "Processing time depends on audio length and hardware"
        ],
        "preprocessing": {
            "sample_rate": "16kHz",
            "channels": "mono",
            "format": "normalized float32"
        }
    }


# Main entry point
if __name__ == "__main__":
    """
    Run the FastAPI server using uvicorn.
    
    Usage:
        python main.py
    
    Server will start on http://localhost:8000
    API docs available at http://localhost:8000/docs
    """
    print("=" * 60)
    print("AI Audio Analyzer API")
    print("=" * 60)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("OpenAPI Schema: http://localhost:8000/openapi.json")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )
