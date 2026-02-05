# Data Collection Guide

## Quick Start (2 Hours)

### 1. Create Directories
```bash
mkdir training_data
mkdir training_data\real
mkdir training_data\ai
```

### 2. Collect REAL Voices (50-100 samples)

**Option A: Record Yourself**
- Use Voice Recorder app (Windows) or QuickTime (Mac)
- Speak for 5-10 seconds each
- Different sentences, tones, volumes
- Save as WAV files in `training_data/real/`

**Option B: YouTube Audio**
```bash
# Install yt-dlp
pip install yt-dlp

# Download audio from interviews, podcasts
yt-dlp -x --audio-format wav -o "training_data/real/%(title)s.%(ext)s" "YOUTUBE_URL"
```

**Option C: Public Datasets**
- LibriSpeech: https://www.openslr.org/12
- Common Voice: https://commonvoice.mozilla.org

### 3. Collect AI Voices (50-100 samples)

**Free TTS Services:**

**ElevenLabs** (Best Quality)
1. Go to: https://elevenlabs.io
2. Free tier: 10k characters/month
3. Generate speech, download as MP3
4. Convert to WAV: `ffmpeg -i input.mp3 output.wav`

**Play.ht**
1. https://play.ht
2. Free trial available
3. Generate and download

**Google Cloud TTS** (Free $300 credit)
```python
from google.cloud import texttospeech
client = texttospeech.TextToSpeechClient()
# Generate speech...
```

**Edge-TTS** (Completely Free)
```bash
pip install edge-tts
edge-tts --text "Hello world" --write-media ai_voice_01.mp3
```

 **Local TTS (Offline)**
```bash
pip install TTS
tts --text "Your text here" --out_path ai_voice.wav
```

### 4. Organize Files
```
training_data/
├── real/
│   ├── real_01.wav
│   ├── real_02.wav
│   └── ... (50+ files)
└── ai/
    ├── ai_01.wav
    ├── ai_02.wav
    └── ... (50+ files)
```

### 5. Train Model
```bash
python train_classifier.py
```

### 6. Deploy
```bash
copy ai_voice_classifier.pkl backend\
python backend\main.py
```

## Expected Results

- **50 samples each**: ~65-70% accuracy
- **100 samples each**: ~75-80% accuracy
- **200+ samples each**: ~80-85% accuracy

## Tips

1. **Diversity Matters**
   - Different speakers
   - Different accents
   - Different recording conditions
   - Different AI voice models

2. **Quality Over Quantity**
   - Clear audio (minimal background noise)
   - Good microphone quality
   - Similar length (5-15 seconds)

3. **Balance Dataset**
   - Equal number of real/AI samples
   - Similar audio characteristics

4. **Test Thoroughly**
   - Save 20% for testing
   - Try with voices not in training set
   - Check for overfitting
