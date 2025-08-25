import numpy as np
from pydub import AudioSegment
import tempfile
import os
import warnings
import wave
from scipy import signal, fft
from scipy.io import wavfile
import io
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
            if max_samples < len(y):
                y = y[:max_samples]
        
        return y, duration
    
    def extract_mfcc_features(self, y, sr, n_mfcc=13):
        """Extract MFCC-like features using scipy"""
        # Pre-emphasis filter
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # Frame the signal
        frame_size = int(0.025 * sr)  # 25ms
        frame_stride = int(0.01 * sr)  # 10ms
        
        signal_length = len(y)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_size)) / frame_stride))
        
        pad_signal_length = int(num_frames * frame_stride + frame_size)
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(y, z)
        
        indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
                 np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_size, 1)).T
        frames = pad_signal[indices.astype(int)]
        
        # Apply window
        frames *= np.hamming(frame_size)
        
        # FFT and Power Spectrum
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
        
        # Mel Filter Banks
        nfilt = 40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((NFFT + 1) * hz_points / sr).astype(int)
        
        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])
            
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        
        # DCT
        try:
            dct_result = fft.dct(filter_banks, type=2, axis=1, norm='ortho')
            if dct_result.shape[1] >= n_mfcc:
                mfcc = dct_result[:, :n_mfcc]
            else:
                mfcc = np.zeros((filter_banks.shape[0], n_mfcc))
        except Exception as e:
            mfcc = np.zeros((filter_banks.shape[0], n_mfcc))
        
        return mfcc
    
    def extract_features(self, audio_path):
        """Extract audio features for personality analysis"""
        try:
            # Convert to WAV if needed
            wav_path = self.convert_to_wav(audio_path)
            
            # Load audio with scipy
            try:
                sr, y = wavfile.read(wav_path)
                # Convert to float and normalize
                if y.dtype == np.int16:
                    y = y.astype(np.float32) / 32767.0
                elif y.dtype == np.int32:
                    y = y.astype(np.float32) / 2147483647.0
                
                # If stereo, convert to mono
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                    
            except Exception as e:
                # Fallback: use pydub to get raw data
                audio_seg = AudioSegment.from_wav(wav_path)
                audio_seg = audio_seg.set_channels(1).set_frame_rate(self.target_sr)
                y = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32767.0
                sr = self.target_sr
            
            # Validate duration
            y, duration = self.validate_audio_duration(y, sr)
            
            # Extract features
            features = {}
            
            # Basic audio properties
            features['duration'] = duration
            features['sample_rate'] = sr
            
            # MFCC features
            try:
                mfccs = self.extract_mfcc_features(y, sr)
                features['mfcc_mean'] = np.mean(mfccs, axis=0)
                features['mfcc_std'] = np.std(mfccs, axis=0)
            except:
                features['mfcc_mean'] = np.zeros(13)
                features['mfcc_std'] = np.zeros(13)
            
            # Pitch estimation using autocorrelation
            def estimate_pitch(signal, sr, win_length=1024, hop_length=512):
                pitches = []
                win_length = int(win_length)
                hop_length = int(hop_length)
                for i in range(0, len(signal) - win_length, hop_length):
                    frame = signal[i:i + win_length]
                    # Autocorrelation
                    autocorr = np.correlate(frame, frame, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # Find the peak (excluding zero lag)
                    min_period = int(sr / 500)  # 500 Hz max
                    max_period = int(sr / 50)   # 50 Hz min
                    
                    if len(autocorr) > max_period:
                        search_range = autocorr[min_period:max_period]
                        if len(search_range) > 0:
                            peak_idx = np.argmax(search_range) + min_period
                            if autocorr[peak_idx] > 0.3 * autocorr[0]:  # Threshold for voiced
                                pitch = sr / peak_idx
                                pitches.append(pitch)
                
                return np.array(pitches)
            
            pitches = estimate_pitch(y, sr)
            if len(pitches) > 0:
                features['pitch_mean'] = np.mean(pitches)
                features['pitch_std'] = np.std(pitches)
                features['pitch_range'] = np.max(pitches) - np.min(pitches)
            else:
                features['pitch_mean'] = 200  # Default value
                features['pitch_std'] = 50
                features['pitch_range'] = 100
            
            # Spectral features using scipy
            frame_length = int(2048)
            hop_length = int(512)
            
            # Short-time Fourier transform
            f, t, Zxx = signal.stft(y, sr, nperseg=frame_length, noverlap=frame_length-hop_length)
            magnitude = np.abs(Zxx)
            
            # Spectral centroid
            spectral_centroids = np.sum(f[:, np.newaxis] * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-12)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral rolloff (85% of energy)
            cumulative_energy = np.cumsum(magnitude, axis=0)
            total_energy = cumulative_energy[-1, :]
            rolloff_point = 0.85 * total_energy
            spectral_rolloff = []
            for i in range(len(rolloff_point)):
                rolloff_idx = np.where(cumulative_energy[:, i] >= rolloff_point[i])[0]
                if len(rolloff_idx) > 0:
                    spectral_rolloff.append(f[rolloff_idx[0]])
                else:
                    spectral_rolloff.append(f[-1])
            
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum((f[:, np.newaxis] - spectral_centroids[np.newaxis, :]) ** 2 * 
                                               magnitude, axis=0) / np.sum(magnitude, axis=0))
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # Zero crossing rate
            def zero_crossing_rate(signal, frame_length=2048, hop_length=512):
                frames = []
                frame_length = int(frame_length)
                hop_length = int(hop_length)
                for i in range(0, len(signal) - frame_length, hop_length):
                    frame = signal[i:i + frame_length]
                    zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                    frames.append(zcr)
                return np.array(frames)
            
            zcr = zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # RMS Energy
            def rms_energy(signal, frame_length=2048, hop_length=512):
                frames = []
                frame_length = int(frame_length)
                hop_length = int(hop_length)
                for i in range(0, len(signal) - frame_length, hop_length):
                    frame = signal[i:i + frame_length]
                    rms = np.sqrt(np.mean(frame ** 2))
                    frames.append(rms)
                return np.array(frames)
            
            rms = rms_energy(y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Tempo estimation (basic)
            features['tempo'] = 120  # Default BPM
            
            # Speaking rate estimation
            voiced_frames = np.sum(rms > np.mean(rms) * 0.1)
            total_frames = len(rms)
            features['speaking_rate'] = (voiced_frames / total_frames) * duration * 10  # rough estimation
            
            # Formant estimation (simplified)
            fft_signal = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(fft_signal), 1/sr)
            # Focus on lower frequencies for formants
            low_freq_mask = (freqs > 0) & (freqs < 4000)
            if np.any(low_freq_mask):
                features['formant_energy'] = np.mean(np.abs(fft_signal[low_freq_mask]))
            else:
                features['formant_energy'] = 1000  # Default value
            
            # Clean up temporary file if created
            if wav_path != audio_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
            
            return features
            
        except Exception as e:
            raise Exception(f"Error extracting audio features: {str(e)}")