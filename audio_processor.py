import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    def __init__(self):
        self.target_sr = 16000
        self.max_duration = 60  # seconds
        self.min_duration = 3   # seconds
    
    def convert_to_wav(self, input_path, output_path=None):
        """Convert audio file to WAV format"""
        try:
            # Load audio file with pydub
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.target_sr)
            
            # Create output path if not provided
            if output_path is None:
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error converting audio to WAV: {str(e)}")
    
    def validate_audio_duration(self, y, sr):
        """Validate audio duration"""
        duration = len(y) / sr
        
        if duration < self.min_duration:
            raise Exception(f"Audio too short. Minimum duration: {self.min_duration} seconds")
        
        if duration > self.max_duration:
            # Truncate to max duration
            max_samples = int(self.max_duration * sr)
            y = y[:max_samples]
        
        return y, duration
    
    def extract_features(self, audio_path):
        """Extract audio features for personality analysis"""
        try:
            # Convert to WAV if needed
            wav_path = self.convert_to_wav(audio_path)
            
            # Load audio with librosa
            y, sr = librosa.load(wav_path, sr=self.target_sr)
            
            # Validate duration
            y, duration = self.validate_audio_duration(y, sr)
            
            # Extract features
            features = {}
            
            # Basic audio properties
            features['duration'] = duration
            features['sample_rate'] = sr
            
            # MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # Pitch/Fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitches = pitches[magnitudes > np.median(magnitudes)]
            pitches = pitches[pitches > 0]
            
            if len(pitches) > 0:
                features['pitch_mean'] = np.mean(pitches)
                features['pitch_std'] = np.std(pitches)
                features['pitch_range'] = np.max(pitches) - np.min(pitches)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # Zero crossing rate (indicates voicing)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # RMS Energy
            rms_energy = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms_energy)
            features['rms_std'] = np.std(rms_energy)
            
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Speaking rate estimation (rough approximation)
            # Count number of voiced segments
            frame_length = 2048
            hop_length = 512
            frames = librosa.frames_to_samples(np.arange(len(rms_energy)), hop_length=hop_length)
            voiced_frames = np.sum(rms_energy > np.mean(rms_energy) * 0.1)
            features['speaking_rate'] = (voiced_frames / len(rms_energy)) * (len(y) / sr) * 10  # rough words per minute estimation
            
            # Formants (approximation using spectral peaks)
            fft = np.abs(np.fft.fft(y))
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            # Find peaks in lower frequency range for formants
            low_freq_mask = (freqs > 0) & (freqs < 4000)
            if np.any(low_freq_mask):
                features['formant_energy'] = np.mean(fft[low_freq_mask])
            else:
                features['formant_energy'] = 0
            
            # Clean up temporary file if created
            if wav_path != audio_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
            
            return features
            
        except Exception as e:
            raise Exception(f"Error extracting audio features: {str(e)}")
