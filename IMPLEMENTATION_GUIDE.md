# Making AI Voice Detection Reliable

## Quick Summary
Current implementation: **Heuristic placeholder (40-60% random)**  
To make it reliable: **Need a trained deepfake detection model**  
Best approach: **Use existing research models or train your own**

---

## Option 1: Use Pre-trained Research Models (Easiest)

### Step 1: Find Available Models

Check these repositories:
```bash
# Hugging Face models for audio deepfake detection
https://huggingface.co/models?search=deepfake%20audio
https://huggingface.co/models?search=anti-spoofing

# GitHub repositories
- ASVspoof baselines: https://github.com/asvspoof-challenge
- RawNet2: https://github.com/Jungjee/RawNet
- AASIST: https://github.com/clovaai/aasist
```

### Step 2: Integrate into Your Code

Replace the `AIVoiceDetector` class in `backend/models.py`:

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch

class AIVoiceDetector:
    def __init__(self, device: str = None):
        print("Loading AI voice detection model...")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Example: Using a hypothetical deepfake detection model
        # Replace with actual model name when found
        model_name = "username/audio-deepfake-detector"  # Find on Hugging Face
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"AI detector loaded on {self.device}")
    
    def detect_ai_voice(self, audio: np.ndarray, sample_rate: int):
        """Detect AI-generated voice using trained model."""
        
        # Preprocess audio for model
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Format results
        # Assumes model outputs: [spoof, bonafide] or [ai, real]
        ai_score = probs[0].item()
        real_score = probs[1].item()
        
        return [
            {"label": "AI", "score": round(ai_score, 2)},
            {"label": "REAL", "score": round(real_score, 2)}
        ]
```

---

## Option 2: Train Your Own Model (Intermediate)

### What You Need:

**1. Dataset (~5-20GB)**
- **ASVspoof 2019/2021**: Standard benchmark dataset
  - Download: https://www.asvspoof.org/index2021.html
  - Contains: Real + synthetic voices (TTS, voice conversion)
  - Size: ~30,000 utterances

- **WaveFake**: Deepfake audio detection
  - GitHub: https://github.com/RUB-SysSec/WaveFake
  - Contains: Real + AI-generated (various vocoders)

- **FakeAVCeleb**: Celebrity voice clones
  - Real + deepfake celebrity voices

**2. Model Architecture**

Use one of these proven architectures:

```python
# A. Fine-tune Wav2Vec 2.0 (Recommended)
from transformers import Wav2Vec2ForSequenceClassification

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2  # AI vs REAL
)

# B. Build Custom CNN
import torch.nn as nn

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers for spectrograms
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64 * 28 * 28, 2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

**3. Training Script**

```python
# training.py
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Trainer

# Load dataset
train_dataset = load_asvspoof_data("path/to/train")
eval_dataset = load_asvspoof_data("path/to/eval")

# Initialize model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/ai-voice-detector",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    fp16=True  # Use mixed precision
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

**4. Evaluation Metrics**

Track these metrics:
- **EER (Equal Error Rate)**: Lower is better (~5-10% is good)
- **Accuracy**: Should be >90% on test set
- **False Positive Rate**: How often real voices flagged as AI
- **False Negative Rate**: How often AI voices pass as real

---

## Option 3: Use Audio Features + Classical ML (Fast)

Quick implementation for better-than-random results:

### Extract Robust Features:

```python
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_features(audio, sr):
    """Extract audio features for AI detection."""
    
    # 1. MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    # 2. Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
    
    # 3. Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # 4. Chroma features
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))
    
    # 5. Temporal envelope
    rms = np.mean(librosa.feature.rms(y=audio))
    
    # Combine all features
    features = np.concatenate([
        mfcc_mean,
        [spectral_centroid, spectral_rolloff, spectral_contrast, zcr, chroma, rms]
    ])
    
    return features

# Train a classifier
# (You'd need labeled data: collect ~100 AI + 100 real samples)
clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)  # Train on your data
```

**Then update `backend/models.py`:**

```python
def detect_ai_voice(self, audio, sample_rate):
    features = extract_features(audio, sample_rate)
    
    # Use trained classifier
    prediction = self.classifier.predict_proba([features])[0]
    
    return [
        {"label": "AI", "score": round(prediction[0], 2)},
        {"label": "REAL", "score": round(prediction[1], 2)}
    ]
```

---

## Realistic Expectations

| Approach | Time | Accuracy | Difficulty |
|----------|------|----------|------------|
| **Heuristics (current)** | 1 hour | ~50% (random) | Easy |
| **Feature + RF** | 1-2 days | 70-80% | Medium |
| **Pre-trained model** | 2-3 days | 85-95% | Medium |
| **Train from scratch** | 1-2 weeks | 90-98% | Hard |

---

## Quick Win: Feature-Based Approach

**This weekend project:**

1. **Collect ~200 samples** (100 AI, 100 real)
   - AI: Use ElevenLabs, Play.ht, TTS tools
   - Real: Record yourself, friends, YouTube clips

2. **Extract features** (code above)

3. **Train RandomForest**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Your labeled data
X = []  # Features from all audio files
y = []  # Labels: 0=real, 1=ai

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.2%}")

# Save model
import joblib
joblib.dump(clf, 'ai_voice_classifier.pkl')
```

4. **Integrate into your API** (replace heuristics with classifier)

**Result**: 70-80% accuracy in 2 days of work!

---

## Recommended Next Steps

### This Week:
1. ✅ Search Hugging Face for "audio deepfake" or "anti-spoofing" models
2. ✅ Try integrating any available pre-trained model
3. ✅ If none found, start collecting audio samples

### This Month:
1. Collect 200+ labeled samples (AI + real)
2. Implement feature extraction + RandomForest
3. Test accuracy on holdout set
4. Deploy improved model

### Long Term:
1. Download ASVspoof dataset
2. Fine-tune Wav2Vec 2.0 or similar
3. Achieve 90%+ accuracy
4. Continuously update as new AI voices emerge

---

## Important Notes

⚠️ **AI Voice Detection is Arms Race**
- New TTS systems emerge constantly
- Model needs regular retraining
- No model will be 100% accurate forever

⚠️ **Ethical Considerations**
- Don't use for surveillance without consent
- False positives can damage reputations
- Clearly communicate accuracy limitations

⚠️ **Legal Disclaimers**
- Add "For research/educational purposes"
- Warn users not to make critical decisions based on results
- Show confidence intervals, not just binary predictions

---

## Resources

**Datasets:**
- ASVspoof: https://www.asvspoof.org
- WaveFake: https://github.com/RUB-SysSec/WaveFake
- FakeAVCeleb: https://sites.google.com/view/fakeavcelebdash-lab

**Papers:**
- "ASVspoof 2021: Towards spoofed and deepfake speech detection"
- "End-to-End Anti-Spoofing with RawNet2"
- "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph"

**Tools:**
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Librosa (audio features): https://librosa.org
- PyTorch Audio: https://pytorch.org/audio

---

**TL;DR**: Your current code is a framework. To make it reliable:
1. **Quick**: Train RandomForest on 200 samples (70-80%)
2. **Better**: Use pre-trained research model (85-95%)
3. **Best**: Train on ASVspoof dataset (90-98%)

Choose based on your timeline and requirements!
