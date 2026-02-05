# Quick Start Guide

## ⚡ Fast Setup (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
cd backend
python main.py
```

Wait for models to download (~500MB, one-time only). You'll see:
```
✓ Models loaded successfully!
```

### 3. Test the API

**Option A: Web Browser**
1. Open http://localhost:8000/docs
2. Click "POST /analyze-audio/"
3. Click "Try it out"
4. Upload an audio file
5. Click "Execute"

**Option B: Command Line**
```bash
# In a new terminal
python test_client.py your_audio_file.wav
```

**Option C: Python Script**
```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze-audio/",
        files={"file": f}
    )
    print(response.json())
```

## 📝 Expected Output

```json
{
  "language_detected": "en",
  "language_name": "English",
  "language_confidence": 0.85,
  "ai_voice_result": [
    {"label": "AI", "score": 0.35},
    {"label": "REAL", "score": 0.65}
  ],
  "device_used": "cpu"
}
```

## 🔍 Troubleshooting

**Server won't start?**
- Check Python version: `python --version` (need 3.8+)
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

**Models not loading?**
- Ensure stable internet connection
- Check disk space (need ~2GB free)

**Slow processing?**
- First run downloads models (one-time wait)
- Use shorter audio files (<1 minute)
- Consider GPU if available

## 🎯 What to Test

Create audio samples in different languages:
- English: "Hello, this is a test"
- Hindi: "नमस्ते, यह एक परीक्षण है"
- Tamil: "வணக்கம், இது ஒரு சோதனை"

Use any recording app or text-to-speech tool.

## 📚 Next Steps

1. Read full [README.md](README.md) for detailed documentation
2. Explore API at http://localhost:8000/docs
3. Check [backend/models.py](backend/models.py) to understand model choices
4. Review [backend/utils.py](backend/utils.py) for preprocessing logic

## ⚠️ Important Notes

- **AI Voice Detection**: Current implementation is a placeholder/demonstration
  - Not production-ready
  - Would need proper training for real use
  
- **First Run**: Downloads models from Hugging Face (~500MB)
  - Subsequent runs are fast
  
- **File Limits**: 
  - Maximum: 50MB
  - Maximum duration: 5 minutes

---

**Need Help?** Check README.md or examine the code - it's well-commented!
