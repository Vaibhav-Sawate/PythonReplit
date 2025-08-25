import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import signal
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

class SpeakerDiarization:
    def __init__(self):
        self.window_length = 2.0  # seconds per window
        self.hop_length = 1.0     # seconds between windows
        self.min_speaker_duration = 3.0  # minimum seconds per speaker
        self.max_speakers = 6     # maximum number of speakers to detect
        
    def extract_speaker_features(self, y, sr, window_samples, hop_samples):
        """Extract features for speaker identification"""
        features = []
        timestamps = []
        
        for i in range(0, len(y) - window_samples, hop_samples):
            window = y[i:i + window_samples]
            timestamp = i / sr
            
            # Skip if window is too quiet (likely silence)
            if np.mean(np.abs(window)) < np.mean(np.abs(y)) * 0.1:
                continue
                
            # Extract MFCC features for speaker identification
            feature_vector = self._extract_window_features(window, sr)
            features.append(feature_vector)
            timestamps.append(timestamp)
        
        return np.array(features), np.array(timestamps)
    
    def _extract_window_features(self, window, sr):
        """Extract features from a single window for speaker identification"""
        features = []
        
        # Basic spectral features
        fft = np.abs(np.fft.fft(window))
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        
        # Only use positive frequencies
        pos_mask = freqs > 0
        fft_pos = fft[pos_mask]
        freqs_pos = freqs[pos_mask]
        
        # Spectral centroid
        if np.sum(fft_pos) > 0:
            spectral_centroid = np.sum(freqs_pos * fft_pos) / np.sum(fft_pos)
        else:
            spectral_centroid = 0
        features.append(spectral_centroid)
        
        # Spectral rolloff (85% of energy)
        cumulative_energy = np.cumsum(fft_pos)
        total_energy = cumulative_energy[-1]
        if total_energy > 0:
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            if len(rolloff_idx) > 0:
                spectral_rolloff = freqs_pos[rolloff_idx[0]]
            else:
                spectral_rolloff = freqs_pos[-1]
        else:
            spectral_rolloff = 0
        features.append(spectral_rolloff)
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * len(window))
        features.append(zcr)
        
        # RMS energy
        rms = np.sqrt(np.mean(window**2))
        features.append(rms)
        
        # Pitch estimation (simplified)
        autocorr = np.correlate(window, window, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find pitch in reasonable range (80-400 Hz)
        min_period = int(sr / 400)
        max_period = int(sr / 80)
        
        if len(autocorr) > max_period:
            search_range = autocorr[min_period:max_period]
            if len(search_range) > 0 and np.max(search_range) > 0.3 * autocorr[0]:
                peak_idx = np.argmax(search_range) + min_period
                pitch = sr / peak_idx
            else:
                pitch = 150  # Default
        else:
            pitch = 150
        features.append(pitch)
        
        # Formant estimation (simplified - first 3 formants)
        # Use linear prediction or spectral peaks
        if len(fft_pos) > 0:
            # Find peaks in spectrum for formant estimation
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(fft_pos, height=np.max(fft_pos) * 0.1)
            if len(peaks) > 0:
                # Take first few peaks as formant frequencies
                formant_freqs = freqs_pos[peaks[:3]]
                while len(formant_freqs) < 3:
                    formant_freqs = np.append(formant_freqs, 0)
                features.extend(formant_freqs[:3])
            else:
                features.extend([500, 1500, 2500])  # Default formants
        else:
            features.extend([500, 1500, 2500])
        
        # Mel-frequency cepstral coefficients (simplified)
        # Use first few coefficients
        n_mels = 13
        try:
            # Simple mel-scale approximation
            mel_filters = self._create_mel_filters(len(fft_pos), sr, n_mels)
            mel_spec = np.dot(mel_filters, fft_pos)
            mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
            log_mel_spec = np.log(mel_spec)
            
            # DCT for cepstral coefficients
            from scipy.fft import dct
            mfccs = dct(log_mel_spec, type=2, norm='ortho')[:8]  # First 8 MFCCs
            features.extend(mfccs)
        except:
            features.extend([0] * 8)  # Fallback
            
        return np.array(features)
    
    def _create_mel_filters(self, n_fft, sr, n_mels):
        """Create simple mel filter bank"""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel scale points
        low_freq_mel = 0
        high_freq_mel = hz_to_mel(sr / 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Create filter bank
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        filters = np.zeros((n_mels, n_fft))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            
            for k in range(f_m_minus, f_m):
                if bin_points[m] != bin_points[m - 1]:
                    filters[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                if bin_points[m + 1] != bin_points[m]:
                    filters[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
        
        return filters
    
    def cluster_speakers(self, features):
        """Cluster features to identify different speakers"""
        if len(features) < 2:
            return np.array([0] * len(features))
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Find optimal number of clusters
        best_n_clusters = 1
        best_score = -1
        
        for n_clusters in range(2, min(self.max_speakers + 1, len(features))):
            if n_clusters >= len(features):
                break
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_normalized)
            
            # Check if we have reasonable cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            if np.min(counts) >= 3:  # Each cluster should have at least 3 points
                try:
                    score = silhouette_score(features_normalized, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except:
                    continue
        
        # Final clustering with best parameters
        if best_n_clusters > 1:
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_normalized)
        else:
            cluster_labels = np.zeros(len(features), dtype=int)
        
        return cluster_labels
    
    def segment_by_speaker(self, y, sr, cluster_labels, timestamps):
        """Segment audio by identified speakers"""
        window_samples = int(self.window_length * sr)
        hop_samples = int(self.hop_length * sr)
        
        # Create speaker segments
        speakers = {}
        current_speaker = cluster_labels[0] if len(cluster_labels) > 0 else 0
        segment_start = 0
        
        for i, (timestamp, speaker) in enumerate(zip(timestamps, cluster_labels)):
            if speaker != current_speaker or i == len(timestamps) - 1:
                # End current segment
                segment_end = int(timestamp * sr) if i < len(timestamps) - 1 else len(y)
                
                if current_speaker not in speakers:
                    speakers[current_speaker] = []
                
                # Add segment if it's long enough
                segment_duration = (segment_end - segment_start) / sr
                if segment_duration >= self.min_speaker_duration:
                    speakers[current_speaker].append(y[segment_start:segment_end])
                
                # Start new segment
                current_speaker = speaker
                segment_start = int(timestamp * sr)
        
        # Concatenate segments for each speaker
        speaker_audio = {}
        for speaker_id, segments in speakers.items():
            if segments:  # Only include speakers with valid segments
                concatenated = np.concatenate(segments)
                if len(concatenated) / sr >= self.min_speaker_duration:  # Final duration check
                    speaker_audio[speaker_id] = concatenated
        
        return speaker_audio
    
    def diarize(self, audio_path):
        """
        Main diarization function
        
        Returns:
            dict: {speaker_id: audio_array} for each identified speaker
        """
        try:
            # Load audio
            from pydub import AudioSegment
            
            # Convert to WAV and load
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Export to temporary WAV for processing
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio.export(tmp_file.name, format='wav')
                tmp_path = tmp_file.name
            
            try:
                # Load with scipy
                sr, y = wavfile.read(tmp_path)
                if y.dtype == np.int16:
                    y = y.astype(np.float32) / 32767.0
                elif y.dtype == np.int32:
                    y = y.astype(np.float32) / 2147483647.0
            finally:
                os.unlink(tmp_path)
            
            # Check if audio is long enough for diarization
            duration = len(y) / sr
            if duration < self.min_speaker_duration * 2:  # Need at least 2 speakers worth of audio
                # Return single speaker
                return {0: y}
            
            # Extract features for speaker identification
            window_samples = int(self.window_length * sr)
            hop_samples = int(self.hop_length * sr)
            
            features, timestamps = self.extract_speaker_features(y, sr, window_samples, hop_samples)
            
            if len(features) < 4:  # Need minimum features for clustering
                return {0: y}
            
            # Cluster to identify speakers
            cluster_labels = self.cluster_speakers(features)
            
            # Segment audio by speaker
            speaker_audio = self.segment_by_speaker(y, sr, cluster_labels, timestamps)
            
            # If no valid speakers found, return original audio
            if not speaker_audio:
                return {0: y}
            
            return speaker_audio
            
        except Exception as e:
            print(f"Diarization error: {str(e)}")
            # Fallback: return single speaker
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_channels(1).set_frame_rate(16000)
                y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32767.0
                return {0: y}
            except:
                return {0: np.array([0.1, 0.1, 0.1])}  # Minimal fallback