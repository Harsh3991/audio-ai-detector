# 🎙️ AI Audio Analyzer

An AI-powered audio analysis and speech forensics system that detects spoken language and determines whether a voice is AI-generated (synthetic / deepfake) or real human speech, built with FastAPI, PyTorch, and Hugging Face Transformers.

```
                                ┌─────────────────────────┐
                                │   Client Audio Input    │
                                │ (WAV, MP3, M4A, FLAC)   │
                                └────────────┬────────────┘
                                             │ HTTP POST /analyze-audio/
                                             ▼
                                ┌─────────────────────────┐
                                │   Audio Preprocessing   │
                                │  • 16 kHz Mono Resample │
                                │  • Peak Normalization   │
                                │  • Silence / Windowing  │
                                └────────────┬────────────┘
                                             │ Normalized Tensor
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
      ┌─────────────────────────────┐                 ┌─────────────────────────────┐
      │  Spoken Language Detection  │                 │    Synthetic Voice Engine   │
      │  (openai/whisper-small)     │                 │   (Multi-Tier Architecture) │
      │                             │                 └──────────────┬──────────────┘
      │  • 80-channel log-mel specs │                                │
      │  • Multi-lingual token IDs  │         ┌──────────────────────┼──────────────────────┐
      │  • Indic script verification│         ▼                      ▼                      ▼
      └──────────────┬──────────────┘  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
                     │                 │ Primary: AST │       │  Secondary:  │       │  Fallback:   │
                     │                 │ Spectrogram  │       │ 54-dim Audio │       │ Rule-based   │
                     │                 │ Transformers │       │  Classifiers │       │ Spectral/ZCR │
                     │                 └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
                     │                        │                      │                      │
                     └────────────────────────┼──────────────────────┴──────────────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │     FastAPI Response        │
                               │  • ISO Language + Conf %    │
                               │  • Real vs AI Probabilities │
                               │  • Device & Latency Context │
                               └─────────────────────────────┘
```

## 🌟 Key Highlights

✅ **Multi-language Identification** - Identifies spoken language across English, Hindi, Telugu, Tamil, and Malayalam with Indic script verification fallback.  
✅ **Multi-Tier AI Voice Detection** - Combines transformer-based audio spectrogram models (AST / Wav2Vec2) with a 54-dimensional acoustic feature extraction pipeline.  
✅ **Acoustic Signal Processing** - Extracts MFCCs, spectral contrast across 7 sub-bands, spectral centroid/rolloff, zero-crossing rate (ZCR), and RMS energy to analyze synthetic vocoder artifacts.  
✅ **High-Performance REST API** - Built with FastAPI, asynchronous file streaming, singleton model loading in memory, and automatic CPU/CUDA hardware detection.  
✅ **Format-Agnostic** - Seamlessly handles WAV, MP3, M4A, FLAC, and OGG with automatic 16 kHz mono conversion and peak normalization.

## 📋 Supported Languages

| Language | Code | Detection Model | Script Verification |
|----------|------|-----------------|---------------------|
| English | `en` | Whisper Small Token ID | Latin |
| Hindi | `hi` | Whisper Small Token ID | Devanagari (`0x0900-0x097F`) |
| Telugu | `te` | Whisper Small Token ID | Telugu (`0x0C00-0x0C7F`) |
| Tamil | `ta` | Whisper Small Token ID | Tamil (`0x0B80-0x0BFF`) |
| Malayalam | `ml` | Whisper Small Token ID | Malayalam (`0x0D00-0x0D7F`) |

## 🧠 Model Architecture & Methodology

### 1. Language Identification: `openai/whisper-small`

- **Architecture**: Sequence-to-sequence encoder-decoder Transformer trained on 680,000+ hours of multilingual audio.
- **Inference Optimization**: Instead of running a costly full-text transcription pass, the system extracts the 80-channel log-mel spectrogram and samples the first token prediction from the decoder prompt head to identify language tokens in constant time.
- **Indic Script Cross-Verification**: To handle known phonetic overlap between South Asian languages (e.g., Whisper occasionally biasing Telugu towards Hindi under noisy conditions), a secondary short decoding verification inspects Unicode character ranges to validate true script identity.

### 2. AI Voice & Synthetic Speech Detection: Multi-Tier Architecture

Synthetic speech generated by modern neural vocoders (HiFi-GAN, WaveGlow, diffusion TTS) often contains subtle phase discontinuities, unnatural spectral regularity, and reduced acoustic micro-jitter. The detection pipeline employs a robust multi-tier strategy:

1. **Primary Tier — Audio Spectrogram Transformer (AST)**:
   - Slices input spectrograms into overlapping 2D patches and applies multi-head self-attention trained on ASVspoof benchmarks to differentiate organic glottal pulses from neural synthesis patterns.
   - Fallback supports fine-tuned Wav2Vec2 audio classification backends.
2. **Secondary Tier — 54-Dimensional Acoustic Feature Extraction**:
   - Computes a rich feature representation extracted via Librosa:
     - **MFCCs (20 coefficients)**: Mean and variance representing vocal tract resonance.
     - **Spectral Dynamics**: Spectral centroid (brightness), spectral rolloff, and spectral bandwidth.
     - **Spectral Contrast**: Dynamic range energy across 7 octave sub-bands to detect vocoder synthesis gaps.
     - **Temporal & Energy Stability**: Zero-crossing rate (ZCR) and RMS energy variations across time.
   - Evaluated using a trained ensemble Random Forest classifier (`train_classifier.py` / `ai_voice_classifier.pkl`).
3. **Tertiary Tier — Acoustic Anomaly Scoring**:
   - If standalone model weights are not loaded, a calibrated heuristic scoring engine evaluates variance across MFCCs and energy consistency to detect synthetic traits without external dependencies.

## 🗂️ Project Structure

```
audio-ai-detector/
│
├── backend/
│   ├── main.py              # FastAPI endpoints, CORS, lifecycle & schemas
│   ├── models.py            # Model loaders (Whisper, AST, feature extractors)
│   ├── utils.py             # Audio preprocessing (resampling, mono downmix)
│   └── __init__.py
│
├── training_data/           # (Optional) Audio samples for classifier training
│   ├── real/                # Real human audio samples (.wav, .mp3)
│   └── ai/                  # AI/synthetic audio samples (.wav, .mp3)
│
├── train_classifier.py      # Acoustic feature extraction & classifier training script
├── test_client.py           # Comprehensive CLI test client for API testing
├── requirements.txt         # Dependencies (FastAPI, PyTorch, Transformers, Librosa)
└── README.md                # Documentation & Architecture Guide
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+ (tested on Python 3.10 - 3.12)
- 4GB+ RAM (8GB recommended)
- Optional: NVIDIA GPU with CUDA for accelerated inference

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/Harsh3991/audio-ai-detector.git
cd audio-ai-detector
```

2. **Create and activate a virtual environment**
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify environment**
```bash
python -c "import torch, transformers, librosa, fastapi; print('Environment ready! CUDA available:', torch.cuda.is_available())"
```

## 🎯 Running & Using the API

### Start the Backend Server

```bash
cd backend
python main.py
```
Or run directly with Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*The API will start at `http://localhost:8000`. On first launch, model weights download and cache automatically in your Hugging Face directory.*

### API Documentation
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc UI**: `http://localhost:8000/redoc`

---

### Request & Response Examples

#### 1. Using cURL

```bash
curl -X POST "http://localhost:8000/analyze-audio/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_voice.wav"
```

#### 2. Using Python (`requests`)

```python
import requests

url = "http://localhost:8000/analyze-audio/"
file_path = "sample_voice.wav"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "audio/wav")}
    response = requests.post(url, files=files)

data = response.json()
print("Language Detected:", data["language_name"], f"({data['language_detected']})")
print("Confidence:", f"{data['language_confidence'] * 100:.1f}%")
print("Voice Classification:")
for result in data["ai_voice_result"]:
    print(f"  - {result['label']}: {result['score'] * 100:.2f}%")
```

#### 3. Using Built-in Test Client

```bash
python test_client.py
```

#### Sample Response JSON

```json
{
  "language_detected": "en",
  "language_name": "English",
  "language_confidence": 0.9824,
  "ai_voice_result": [
    {
      "label": "REAL",
      "score": 0.8942
    },
    {
      "label": "AI",
      "score": 0.1058
    }
  ],
  "device_used": "cuda"
}
```

## 📊 API Endpoints Reference

| Endpoint | Method | Params / Body | Description |
|----------|--------|---------------|-------------|
| `/` | `GET` | None | API metadata and root health check |
| `/health/` | `GET` | None | Detailed health status and model load status |
| `/analyze-audio/` | `POST` | `multipart/form-data` (`file`) | Uploads audio and returns language + AI classification |
| `/supported-languages/` | `GET` | None | Lists all supported language codes and names |
| `/info/` | `GET` | None | System configuration, device type, and audio constraints |

## 🔊 Audio Processing Pipeline

Before tensors reach models, the raw stream goes through automated conditioning:
1. **Format Agnosticism**: Ingests WAV, MP3, M4A, FLAC, and OGG up to 50MB.
2. **Resampling (16,000 Hz)**: Standardizes sampling rates to align with acoustic receptive fields of transformer backbones.
3. **Channel Downmixing**: Dual/multichannel audio is downmixed to mono float32.
4. **Peak Amplitude Normalization**: Tensors are scaled to `[-1.0, 1.0]` range to prevent numerical explosion.
5. **Windowing & Silence Trimming**: Detects leading/trailing non-speech and extracts a centered 10-second window for bounded inference latency.

## 🧪 Training a Custom Voice Classifier

To train the secondary acoustic feature classifier on your custom voice dataset:

1. Place real human audio in `training_data/real/` and AI-generated samples in `training_data/ai/`.
2. Run the training script:
```bash
python train_classifier.py
```
3. The script extracts the 54-dimensional acoustic feature matrices, trains a Random Forest ensemble with cross-validation, and serializes the model to `ai_voice_classifier.pkl`. The backend automatically picks up this model on next boot.

## 📝 License

Distributed under the MIT License. See `LICENSE` for details. Model weights from OpenAI and Hugging Face are governed by their respective licenses.
