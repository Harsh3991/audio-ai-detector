# 🎙️ AI Audio Analyzer

An AI-powered audio analysis system that detects spoken language and determines whether a voice is AI-generated or human, built with FastAPI and Hugging Face models.

## 🌟 Features

✅ **Language Detection** - Identifies spoken language from audio  
✅ **AI Voice Detection** - Determines if voice is AI-generated or human  
✅ **Multi-language Support** - English, Hindi, Telugu, Tamil, Malayalam  
✅ **REST API** - Easy-to-use FastAPI endpoints  
✅ **No GPU Required** - Runs on CPU (GPU support available)  
✅ **Production Ready** - Clean code, error handling, validation

## 📋 Supported Languages

| Language | Code | Status |
|----------|------|--------|
| English | `en` | ✅ Supported |
| Hindi | `hi` | ✅ Supported |
| Telugu | `te` | ✅ Supported |
| Tamil | `ta` | ✅ Supported |
| Malayalam | `ml` | ✅ Supported |

## 🧠 Models Used

### 1. Language Detection: `openai/whisper-small`

**Why this model?**
- **Robust multilingual support**: Trained on 680k hours of multilingual audio data
- **99+ languages**: Supports all required languages (en, hi, te, ta, ml) and many more
- **Balanced performance**: Small variant provides good accuracy without excessive computational requirements
- **Automatic language detection**: No need to specify language upfront
- **Well-maintained**: OpenAI model with active community support

**How it works:**
- Whisper uses an encoder-decoder transformer architecture
- Processes audio spectrograms to generate text transcriptions
- Simultaneously detects the language spoken
- Outputs language code directly from internal language classification

### 2. AI Voice Detection: Heuristic-based Placeholder

**Important Note:**
Currently uses a **heuristic-based placeholder** implementation because:
- Very few reliable open-source models exist for AI voice detection as of 2024-2026
- AI voice generation technology evolves rapidly (TTS, voice cloning, etc.)
- Production systems require models trained specifically on synthetic vs. real voice datasets

**Why this limitation exists:**
- AI voice detection is an emerging field
- Most effective models are proprietary or research-only
- Requires continuous training on latest AI voice generation techniques

**For Production:**
Would recommend:
- Training a custom model on datasets like:
  - ASVspoof (Anti-Spoofing Voice)
  - Fake-or-Real (FoR) dataset
  - Custom dataset of latest AI voices (Eleven Labs, Play.ht, etc.)
- Using audio features like:
  - Mel-frequency cepstral coefficients (MFCCs)
  - Phase information
  - Spectral anomalies
  - Periodicity analysis

## 🗂️ Project Structure

```
audio-ai-detector/
│
├── backend/
│   ├── main.py        # FastAPI application
│   ├── models.py      # ML model loading & inference
│   ├── utils.py       # Audio preprocessing utilities
│
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (8GB+ recommended)
- Internet connection (for downloading models on first run)

### Installation

1. **Clone or download the project**
```bash
cd audio-ai-detector
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (web server)
- PyTorch & Transformers (ML frameworks)
- Librosa & Soundfile (audio processing)
- NumPy, SciPy (numerical computing)

**Note:** First install may take 5-10 minutes depending on your connection.

4. **Verify installation**
```bash
python -c "import torch; import transformers; import librosa; print('✓ All packages installed')"
```

## 🎯 Usage

### Starting the Server

```bash
cd backend
python main.py
```

Server will start on `http://localhost:8000`

**First startup:**
- Models will download automatically (~500MB for Whisper small)
- This happens only once, subsequent starts are fast

### API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Testing the API

#### Using cURL

```bash
curl -X POST "http://localhost:8000/analyze-audio/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_audio.wav"
```

#### Using Python

```python
import requests

# Analyze an audio file
with open("sample_audio.wav", "rb") as f:
    files = {"file": ("sample_audio.wav", f, "audio/wav")}
    response = requests.post(
        "http://localhost:8000/analyze-audio/",
        files=files
    )
    
result = response.json()
print(f"Language: {result['language_name']}")
print(f"AI Voice: {result['ai_voice_result']}")
```

#### Using the Interactive Docs

1. Go to http://localhost:8000/docs
2. Click on `POST /analyze-audio/`
3. Click "Try it out"
4. Upload your audio file
5. Click "Execute"
6. View results below

### Example Response

```json
{
  "language_detected": "ta",
  "language_name": "Tamil",
  "language_confidence": 0.85,
  "ai_voice_result": [
    {
      "label": "AI",
      "score": 0.35
    },
    {
      "label": "REAL",
      "score": 0.65
    }
  ],
  "device_used": "cpu"
}
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and health check |
| `/analyze-audio/` | POST | Analyze uploaded audio file |
| `/supported-languages/` | GET | List supported languages |
| `/health/` | GET | Server health status |
| `/info/` | GET | Detailed API information |

## 🔊 Audio Preprocessing

All uploaded audio files are automatically preprocessed before inference:

### Why Preprocessing is Required

1. **Consistent Sample Rate (16kHz)**
   - ML models are trained on specific sample rates
   - Whisper expects 16kHz audio
   - Resampling ensures compatibility

2. **Mono Conversion**
   - Reduces computational complexity
   - Most speech models expect single-channel audio
   - Preserves all linguistic information

3. **Normalization**
   - Ensures audio amplitude is in expected range [-1, 1]
   - Prevents numerical issues during inference
   - Improves model accuracy

### Supported Formats

- ✅ WAV (.wav)
- ✅ MP3 (.mp3)
- ✅ M4A (.m4a)
- ✅ FLAC (.flac)
- ✅ OGG (.ogg)

### Limitations

- **Maximum file size**: 50MB
- **Maximum duration**: 5 minutes (300 seconds)
- **Minimum quality**: 8kHz sample rate recommended

## ⚠️ Important Limitations

### AI Voice Detection

**Current Implementation:**
- Uses heuristic-based features (placeholder)
- Provides confidence scores but NOT production-ready
- Should be treated as a framework/proof-of-concept

**Why it's limited:**
1. No robust open-source models available for AI voice detection
2. AI voice technology evolves rapidly (new TTS systems emerge constantly)
3. Effective detection requires specialized training data

**False Positives/Negatives:**
- May incorrectly classify high-quality AI voices as real
- May incorrectly classify low-quality recordings as AI
- Background noise can affect results
- Compression artifacts may trigger false positives

**For Production Use:**
- Train a custom binary classifier on labeled data
- Use datasets: ASVspoof 2019/2021, WaveFake, FakeAVCeleb
- Extract audio features: MFCCs, phase, spectral features
- Regularly update model as AI voice tech advances
- Consider ensemble of multiple detection methods

### Language Detection

**Generally Reliable but:**
- Accuracy depends on audio quality
- Very short clips (<3 seconds) may be unreliable
- Mixed-language audio may produce unexpected results
- Heavy accents or dialects can affect detection

## 🔮 Future Improvements

### Short Term
- [ ] Add batch processing for multiple files
- [ ] Implement confidence score aggregation
- [ ] Add audio quality assessment
- [ ] Support for real-time streaming analysis

### Medium Term
- [ ] Train proper AI voice detection model
- [ ] Add speaker diarization (multiple speakers)
- [ ] Implement audio enhancement/denoising
- [ ] Add support for more languages

### Long Term
- [ ] Web frontend (React/Vue)
- [ ] Real-time microphone input
- [ ] Mobile app integration
- [ ] Emotion detection from voice
- [ ] Voice biometrics / speaker identification

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Quality

```bash
# Format code
pip install black
black backend/

# Lint code
pip install flake8
flake8 backend/
```

### Performance Tips

1. **Use GPU if available**
   - Models will automatically use CUDA if available
   - Check with: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Reduce model size**
   - Use `openai/whisper-tiny` for faster inference (less accurate)
   - Use `openai/whisper-base` for balanced performance

3. **Batch processing**
   - Process multiple files together for efficiency
   - Implement queue system for high-volume scenarios

## 🐛 Troubleshooting

### Models not loading
```
Error: Could not load model...
```
**Solution:** Ensure you have stable internet connection for first download

### Out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Use CPU mode or reduce batch size

### Audio format not supported
```
Error: Failed to preprocess audio
```
**Solution:** Convert audio to WAV using ffmpeg:
```bash
ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav
```

### Slow processing
**Solution:**
- Use GPU if available
- Reduce audio file length
- Use smaller Whisper model (tiny/base)

## 📝 License

This project is for educational/portfolio purposes. Model licenses:
- Whisper: MIT License (OpenAI)
- Other dependencies: See individual package licenses

## 👨‍💻 Author

Built as a demonstration of ML engineering best practices:
- Clean architecture with separation of concerns
- Production-ready error handling
- Comprehensive documentation
- No training required (inference-only)
- Modular and extensible design

## 🙏 Acknowledgments

- OpenAI for Whisper model
- Hugging Face for Transformers library
- FastAPI for excellent web framework
- Librosa for audio processing utilities

---

**Note:** This is a portfolio/demonstration project. For production deployment:
- Add authentication & rate limiting
- Implement proper logging & monitoring
- Add comprehensive test coverage
- Train or integrate proper AI voice detection model
- Set up CI/CD pipeline
- Add database for request tracking
- Implement caching for repeated analyses
