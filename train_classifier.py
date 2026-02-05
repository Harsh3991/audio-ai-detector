"""
Train a simple AI voice classifier using audio features

Usage:
1. Collect audio samples:
   - Put real voice samples in: ./training_data/real/*.wav
   - Put AI voice samples in: ./training_data/ai/*.wav

2. Run training:
   python train_classifier.py

3. Model will be saved as: ai_voice_classifier.pkl
"""

import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def extract_features(audio_path, sr=16000):
    """Extract same features as in models.py"""
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    features = []
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # Spectral features
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features.append(np.std(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    
    # RMS energy
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.append(np.mean(chroma))
    features.append(np.std(chroma))
    
    return np.array(features)


def load_dataset(data_dir="./training_data"):
    """Load audio files from directories"""
    X = []
    y = []
    
    real_dir = os.path.join(data_dir, "real")
    ai_dir = os.path.join(data_dir, "ai")
    
    print("\nLoading dataset...")
    
    # Load real voices
    if os.path.exists(real_dir):
        real_files = [f for f in os.listdir(real_dir) if f.endswith(('.wav', '.mp3'))]
        print(f"Found {len(real_files)} real voice samples")
        
        for filename in real_files:
            try:
                features = extract_features(os.path.join(real_dir, filename))
                X.append(features)
                y.append(0)  # 0 = REAL
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Load AI voices
    if os.path.exists(ai_dir):
        ai_files = [f for f in os.listdir(ai_dir) if f.endswith(('.wav', '.mp3'))]
        print(f"Found {len(ai_files)} AI voice samples")
        
        for filename in ai_files:
            try:
                features = extract_features(os.path.join(ai_dir, filename))
                X.append(features)
                y.append(1)  # 1 = AI
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return np.array(X), np.array(y)


def train_classifier():
    """Train and save the classifier"""
    print("="*60)
    print("AI Voice Classifier Training")
    print("="*60)
    
    # Load data
    X, y = load_dataset()
    
    if len(X) < 20:
        print("\n⚠ WARNING: Very few samples! Need at least 100 samples (50 each)")
        print("  Collect more data for reliable results")
        return
    
    print(f"\nDataset: {len(X)} samples ({np.sum(y==0)} real, {np.sum(y==1)} AI)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train classifier
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"\nTraining accuracy: {train_score:.2%}")
    print(f"Test accuracy: {test_score:.2%}")
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-validation: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
    
    # Detailed metrics
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'AI']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    model_path = "ai_voice_classifier.pkl"
    joblib.dump(clf, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    print("\nCopy this file to backend/ directory to use it")


if __name__ == "__main__":
    train_classifier()
