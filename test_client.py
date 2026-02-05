"""
Example client script to test the Audio Analyzer API

This script demonstrates how to use the API programmatically.
Run this AFTER starting the server with: python backend/main.py
"""

import requests
import sys
import os
from pathlib import Path


API_URL = "http://localhost:8000"


def check_server():
    """Check if the server is running."""
    try:
        response = requests.get(f"{API_URL}/health/", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
            return True
        else:
            print("✗ Server returned unexpected status")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running")
        print("Please start the server first: python backend/main.py")
        return False
    except Exception as e:
        print(f"✗ Error checking server: {e}")
        return False


def get_supported_languages():
    """Get list of supported languages."""
    try:
        response = requests.get(f"{API_URL}/supported-languages/")
        if response.status_code == 200:
            languages = response.json()
            print("\n📋 Supported Languages:")
            for lang in languages:
                print(f"  - {lang['language_name']} ({lang['language_code']})")
            return True
        return False
    except Exception as e:
        print(f"Error getting languages: {e}")
        return False


def analyze_audio(audio_file_path):
    """
    Analyze an audio file.
    
    Args:
        audio_file_path: Path to audio file
    """
    if not os.path.exists(audio_file_path):
        print(f"✗ File not found: {audio_file_path}")
        return False
    
    print(f"\n🎵 Analyzing: {os.path.basename(audio_file_path)}")
    print("Please wait...")
    
    try:
        # Open and send file
        with open(audio_file_path, "rb") as f:
            files = {"file": (os.path.basename(audio_file_path), f)}
            response = requests.post(
                f"{API_URL}/analyze-audio/",
                files=files,
                timeout=120  # 2 minute timeout for processing
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "="*50)
            print("📊 ANALYSIS RESULTS")
            print("="*50)
            
            # Language detection
            print(f"\n🗣️  Language Detected: {result['language_name']} ({result['language_detected']})")
            print(f"   Confidence: {result['language_confidence']:.2%}")
            
            # AI voice detection
            print(f"\n🤖 AI Voice Detection:")
            for item in result['ai_voice_result']:
                label = item['label']
                score = item['score']
                bar = "█" * int(score * 20)
                print(f"   {label:5s}: {score:.2%} {bar}")
            
            # Device info
            print(f"\n⚙️  Processed on: {result['device_used'].upper()}")
            print("="*50 + "\n")
            
            return True
        
        elif response.status_code == 400:
            error = response.json()
            print(f"✗ Invalid file: {error.get('detail', 'Unknown error')}")
            return False
        
        elif response.status_code == 503:
            print("✗ Server models not loaded yet. Please wait and try again.")
            return False
        
        else:
            print(f"✗ Server error: {response.status_code}")
            print(response.text)
            return False
    
    except requests.exceptions.Timeout:
        print("✗ Request timeout. File may be too large or server is overloaded.")
        return False
    
    except Exception as e:
        print(f"✗ Error analyzing audio: {e}")
        return False


def get_api_info():
    """Get detailed API information."""
    try:
        response = requests.get(f"{API_URL}/info/")
        if response.status_code == 200:
            info = response.json()
            print("\n📚 API Information:")
            print(f"  Version: {info['api_version']}")
            print(f"\n  Models:")
            print(f"    Language: {info['models']['language_detection']['model']}")
            print(f"    AI Voice: {info['models']['ai_voice_detection']['model']}")
            return True
        return False
    except Exception as e:
        print(f"Error getting API info: {e}")
        return False


def main():
    """Main function."""
    print("="*60)
    print("🎙️  AI Audio Analyzer - Test Client")
    print("="*60)
    
    # Check if server is running
    if not check_server():
        sys.exit(1)
    
    # Get API info
    get_api_info()
    
    # Get supported languages
    get_supported_languages()
    
    # Check if audio file provided
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("📝 USAGE")
        print("="*60)
        print(f"\nPython: python test_client.py <audio_file>")
        print(f"\nExample:")
        print(f"  python test_client.py sample.wav")
        print(f"  python test_client.py my_audio.mp3")
        print(f"\nSupported formats: .wav, .mp3, .m4a, .flac, .ogg")
        print("\n" + "="*60)
        return
    
    # Analyze each provided file
    for audio_file in sys.argv[1:]:
        analyze_audio(audio_file)


if __name__ == "__main__":
    main()
