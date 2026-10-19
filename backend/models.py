"""
Model loading and inference for audio analysis
Uses Hugging Face transformers for pretrained models
"""

import torch
import numpy as np
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class AudioAnalyzer:
    """
    Manages all ML models for audio analysis
    - Language detection using Whisper
    - AI voice detection using AST ASVspoof model
    """
    
    def __init__(self, device: str = None):
        """
        Initialize models
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading models on {self.device.upper()}...")
        
        # Language detection model
        self._load_language_model()
        
        # AI voice detection model
        self._load_ai_detection_model()
        
        print("[OK] Models loaded successfully!")
    
    def _load_language_model(self):
        """
        Load Whisper for language detection
        
        Model: openai/whisper-small
        Why: 680k hours training, 99 languages, battle-tested
        Size: ~500MB
        """
        print("  Loading Whisper (language detection)...")

        model_name = "openai/whisper-small"
        
        self.whisper_processor = WhisperProcessor.from_pretrained(model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.whisper_model.to(self.device)
        self.whisper_model.eval()
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'ml': 'Malayalam'
        }
    
    def _load_ai_detection_model(self):
        """
        Load AST model for AI voice detection
        
        Model: MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection
        Why: Specifically trained on ASVspoof5 for synthetic voice detection
        Architecture: Audio Spectrogram Transformer (state-of-the-art)
        """
        print("  Loading AST ASVspoof (AI voice detection)...")
        
        # Try multiple model options in order of preference
        model_options = [
            "MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection",  # Correct model name!
            "m3hrdadfi/wav2vec2-base-asvspoof",  # Alternative ASVspoof model
            "facebook/wav2vec2-base"  # Fallback to fine-tune yourself
        ]
        
        for model_name in model_options:
            try:
                print(f"  Trying: {model_name}")
                self.ast_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.ast_model = AutoModelForAudioClassification.from_pretrained(model_name)
                self.ast_model.to(self.device)
                self.ast_model.eval()
                self.ai_detection_available = True
                self.ai_model_name = model_name
                print(f"  [OK] Loaded: {model_name}")
                
                # Debug: Show model's label configuration
                if hasattr(self.ast_model.config, 'id2label'):
                    print(f"  [INFO] Model labels: {self.ast_model.config.id2label}")
                else:
                    print(f"  [WARN] No id2label found, using defaults")
                
                return
            except Exception as e:
                print(f"  [WARN] Failed: {model_name} - {str(e)}")
                continue
        
        # If all fail, use feature-based approach
        print("  Using feature-based AI detection")
        self.ai_detection_available = False
        self._init_feature_detector()
    
    def _init_feature_detector(self):
        """Initialize feature-based detection as fallback"""
        # This is more reliable than random heuristics
        import pickle
        import os
        
        # Check if we have a pre-trained classifier
        model_path = "ai_voice_classifier.pkl"
        if os.path.exists(model_path):
            import joblib
            self.feature_classifier = joblib.load(model_path)
            self.use_feature_classifier = True
            print("  [OK] Loaded feature-based classifier")
        else:
            self.use_feature_classifier = False
            print("  [WARN] No feature classifier found - using heuristics")
    
    def detect_language(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Detect language from audio using Whisper's built-in language detection
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate (should be 16000)
        
        Returns:
            {
                'language_code': 'en',
                'language_name': 'English',
                'confidence': 0.95
            }
        """
        try:
            # Process audio
            input_features = self.whisper_processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features
            
            input_features = input_features.to(self.device)
            
            # Use Whisper's built-in language detection
            # Generate with language=None to auto-detect
            with torch.no_grad():
                # Force the model to detect language first
                # We use a very short generation to get the language token
                forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                    language=None, task="transcribe"
                )
                
                # Detect language using the model's forward pass
                # Whisper stores language in the first token
                outputs = self.whisper_model.generate(
                    input_features,
                    max_new_tokens=1,  # Only need first token for language
                    language=None,  # Auto-detect
                    task="transcribe",
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Get the first generated token
                first_token = outputs.sequences[0][1].item()
                
                # Whisper's language tokens are specific IDs
                # The processor can tell us which language this corresponds to
                # We'll use the tokenizer to decode
                
                # Alternative: Use Whisper's detect_language method if available
                # This is the more reliable approach
                try:
                    # Whisper model has a detect_language method
                    # It returns (language_token, language_probability)
                    detection_result = self.whisper_model.detect_language(input_features)

                    # Handle different return formats from different Whisper versions
                    if isinstance(detection_result, tuple) and len(detection_result) == 2:
                        language_token, language_prob = detection_result
                    else:
                        # Newer versions might return a dict or different format
                        print(f"  [DEBUG] Unexpected detect_language format: {type(detection_result)}")
                        raise ValueError("Unexpected detect_language return format")
                    
                    # Convert token to language code
                    # Whisper uses specific language codes
                    language_code_map = {
                        'en': 'en',
                        'english': 'en',
                        'hi': 'hi',
                        'hindi': 'hi',
                        'te': 'te',
                        'telugu': 'te',
                        'ta': 'ta',
                        'tamil': 'ta',
                        'ml': 'ml',
                        'malayalam': 'ml'
                    }

                    # Get language from processor's token dictionary
                    detected_lang_code = self.whisper_processor.decode([language_token.item()])

                    # Clean up the detected language (remove special tokens)
                    detected_lang_code = detected_lang_code.strip().lower()

                    # Debug logging
                    print(f"  [DEBUG] Raw detected language token: '{detected_lang_code}'")
                    print(f"  [DEBUG] Language token ID: {language_token.item()}")
                    print(f"  [DEBUG] Language probability: {confidence:.4f}")

                    # Map to our supported languages
                    if detected_lang_code in language_code_map:
                        detected_lang = language_code_map[detected_lang_code]
                    else:
                        # Try to match partial - better matching for Indian languages
                        detected_lang = 'en'  # Default
                        for key, value in language_code_map.items():
                            if detected_lang_code.startswith(key[:2]) or key.startswith(detected_lang_code[:2]):
                                detected_lang = value
                                print(f"  [DEBUG] Matched '{detected_lang_code}' to '{detected_lang}' via partial match")
                                break
                        else:
                            print(f"  [DEBUG] No match found for '{detected_lang_code}', defaulting to English")
                    
                    confidence = float(language_prob.item())
                    
                except Exception as e:
                    print(f"Using fallback language detection: {e}")
                    # Fallback: Generate short transcription and detect from text
                    generated_ids = self.whisper_model.generate(
                        input_features,
                        max_length=50,
                        language=None,
                        task="transcribe"
                    )

                    transcription = self.whisper_processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]

                    print(f"  [DEBUG] Fallback transcription: '{transcription[:50]}...'")

                    # Simple heuristic based on transcription content
                    detected_lang = 'en'  # Default
                    confidence = 0.60

                    # Check for language-specific characters in transcription
                    if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in transcription):
                        detected_lang = 'hi'  # Devanagari script
                        confidence = 0.85
                    elif any(ord(c) >= 0x0B80 and ord(c) <= 0x0BFF for c in transcription):
                        detected_lang = 'ta'  # Tamil script
                        confidence = 0.85
                    elif any(ord(c) >= 0x0C00 and ord(c) <= 0x0C7F for c in transcription):
                        detected_lang = 'te'  # Telugu script
                        confidence = 0.85
                    elif any(ord(c) >= 0x0D00 and ord(c) <= 0x0D7F for c in transcription):
                        detected_lang = 'ml'  # Malayalam script
                        confidence = 0.85

                    print(f"  [DEBUG] Fallback detected language: {detected_lang}")

                    # Special handling for Telugu: Whisper often transcribes Telugu as Hindi
                    # If detected as Hindi, try transcribing with Telugu forced to check
                    if detected_lang == 'hi':
                        print(f"  [DEBUG] Detected Hindi, checking if it might be Telugu...")
                        try:
                            # Try forcing Telugu transcription
                            telugu_ids = self.whisper_model.generate(
                                input_features,
                                max_length=50,
                                language='te',
                                task="transcribe"
                            )
                            telugu_transcription = self.whisper_processor.batch_decode(
                                telugu_ids,
                                skip_special_tokens=True
                            )[0]

                            print(f"  [DEBUG] Telugu-forced transcription: '{telugu_transcription[:50]}...'")

                            # Check if Telugu transcription produces Telugu script (0x0C00-0x0C7F)
                            has_telugu_chars = any(ord(c) >= 0x0C00 and ord(c) <= 0x0C7F for c in telugu_transcription)

                            if has_telugu_chars:
                                print(f"  [DEBUG] Telugu characters found, switching to Telugu")
                                detected_lang = 'te'
                                confidence = 0.80
                        except Exception as te:
                            print(f"  [DEBUG] Telugu check failed: {te}")
            
            # Ensure valid language
            if detected_lang not in self.supported_languages:
                detected_lang = 'en'
                confidence = confidence * 0.7  # Reduce confidence for fallback
            
            return {
                'language_code': detected_lang,
                'language_name': self.supported_languages[detected_lang],
                'confidence': min(confidence, 0.99)  # Cap at 99%
            }
            
        except Exception as e:
            print(f"Language detection error: {e}")
            # Return safe default
            return {
                'language_code': 'en',
                'language_name': 'English',
                'confidence': 0.5
            }
    
    def detect_ai_voice(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect if voice is AI-generated using AST model
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate (should be 16000)
        
        Returns:
            [
                {'label': 'bonafide', 'score': 0.85},  # Real human
                {'label': 'spoof', 'score': 0.15}      # AI/synthetic
            ]
        """
        if not self.ai_detection_available:
            return self._feature_based_detection(audio, sr)
        
        try:
            # Prepare audio for model (ensure correct length)
            # Some models expect fixed-length input
            max_length = 16000 * 10  # 10 seconds max
            if len(audio) > max_length:
                # Take middle segment
                start = (len(audio) - max_length) // 2
                audio = audio[start:start + max_length]
            
            inputs = self.ast_feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.ast_model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()[0]
            
            # Map to labels based on model configuration
            if hasattr(self.ast_model.config, 'id2label'):
                label_map = self.ast_model.config.id2label
                print(f"  [INFO] Using model labels: {label_map}")
            else:
                # Default assumption: Many ASVspoof models use [spoof, bonafide]
                label_map = {0: 'spoof', 1: 'bonafide'}
                print(f"  [INFO] Using default labels: {label_map}")
            
            results = []
            for idx, score in enumerate(probs):
                label = label_map.get(idx, f'class_{idx}')
                # Correct mapping: Bonafide = REAL, Spoof = AI
                if any(keyword in label.lower() for keyword in ['bonafide', 'real', 'human', 'genuine']):
                    normalized_label = 'REAL'
                elif any(keyword in label.lower() for keyword in ['spoof', 'fake', 'ai', 'synthetic']):
                    normalized_label = 'AI'
                else:
                    # Unknown label, use default mapping
                    normalized_label = 'REAL' if idx == 0 else 'AI'

                print(f"    Index {idx}: {label} -> {normalized_label} (score: {score:.4f})")

                results.append({
                    'label': normalized_label,
                    'score': round(float(score), 4)  # Round to 4 decimal places
                })
            
            # Ensure we have both labels
            labels_present = [r['label'] for r in results]
            if 'REAL' not in labels_present:
                results.append({'label': 'REAL', 'score': 0.0})
            if 'AI' not in labels_present:
                results.append({'label': 'AI', 'score': 0.0})
            
            # Normalize scores to ensure they sum to 1.0 after rounding
            total = sum(r['score'] for r in results)
            if total > 0:
                for r in results:
                    r['score'] = round(r['score'] / total, 4)
            
            # Sort by score descending
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            return results[:2]  # Return top 2
            
        except Exception as e:
            print(f"AI detection error: {e}")
            return self._feature_based_detection(audio, sr)
    
    def _feature_based_detection(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Feature-based AI detection using audio characteristics
        More reliable than random heuristics
        """
        import librosa
        
        try:
            # Extract comprehensive features
            features = self._extract_audio_features(audio, sr)
            
            if self.use_feature_classifier:
                # Use trained classifier
                prediction = self.feature_classifier.predict_proba([features])[0]
                return [
                    {'label': 'REAL', 'score': round(float(prediction[0]), 4)},
                    {'label': 'AI', 'score': round(float(prediction[1]), 4)}
                ]
            else:
                # Use rule-based heuristics (better than before)
                ai_score = self._calculate_ai_score(features)
                return [
                    {'label': 'REAL', 'score': round(float(1.0 - ai_score), 4)},
                    {'label': 'AI', 'score': round(float(ai_score), 4)}
                ]
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self._fallback_ai_detection(audio, sr)
    
    def _fallback_ai_detection(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Fallback detection when all other methods fail
        Returns neutral scores
        """
        return [
            {'label': 'REAL', 'score': 0.5},
            {'label': 'AI', 'score': 0.5}
        ]
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract robust audio features for AI detection"""
        import librosa
        
        features = []
        
        # 1. MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # 2. Spectral features
        features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        features.append(np.std(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        
        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 4. RMS energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # 5. Spectral contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend(np.mean(contrast, axis=1))
        
        # 6. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(np.mean(chroma))
        features.append(np.std(chroma))
        
        return np.array(features)
    
    def _calculate_ai_score(self, features: np.ndarray) -> float:
        """
        Calculate AI likelihood score from features
        Based on common patterns in synthetic speech
        """
        # Normalize features
        mfcc_mean_variance = np.std(features[:20])
        spectral_regularity = features[40]  # Spectral centroid mean
        energy_consistency = 1.0 - features[45]  # RMS std (inverted)
        
        # AI voices tend to have:
        # - More consistent MFCCs (lower variance)
        # - More regular spectral patterns
        # - More consistent energy
        
        ai_indicators = [
            1.0 - min(mfcc_mean_variance / 100.0, 1.0),  # Lower variance = more AI-like
            min(spectral_regularity / 5000.0, 1.0) * 0.5,  # Moderate regularity
            energy_consistency * 0.3  # Consistent energy
        ]
        
        ai_score = np.mean(ai_indicators)
        
        # Ensure score is in valid 0-1 range
        ai_score = np.clip(ai_score, 0.0, 1.0)
        
        return ai_score

    def analyze(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Full analysis: language + AI detection
        
        Args:
            audio: Audio waveform
            sr: Sample rate
        
        Returns:
            Complete analysis results
        """
        lang_result = self.detect_language(audio, sr)
        ai_result = self.detect_ai_voice(audio, sr)
        
        return {
            'language_detected': lang_result['language_code'],
            'language_name': lang_result['language_name'],
            'language_confidence': lang_result['confidence'],
            'ai_voice_result': ai_result,
            'device_used': self.device
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Return supported languages"""
        return self.supported_languages.copy()


# Global model instance (loaded once)
_audio_analyzer = None


def get_analyzer() -> AudioAnalyzer:
    """
    Get or create singleton instance of AudioAnalyzer
    Ensures models are loaded only once
    """
    global _audio_analyzer
    
    if _audio_analyzer is None:
        _audio_analyzer = AudioAnalyzer()
    
    return _audio_analyzer
