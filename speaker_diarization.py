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
        self.window_length = 1.5  # seconds per window (shorter for better resolution)
        self.hop_length = 0.5     # seconds between windows (more overlap)
        self.min_speaker_duration = 2.0  # minimum seconds per speaker (reduced)
        self.max_speakers = 8     # maximum number of speakers to detect
        
    def extract_speaker_features(self, y, sr, window_samples, hop_samples):
        """Extract features for speaker identification"""
        features = []
        timestamps = []
        
        for i in range(0, len(y) - window_samples, hop_samples):
            window = y[i:i + window_samples]
            timestamp = i / sr
            
            # Skip if window is too quiet (likely silence) - made less restrictive
            if np.mean(np.abs(window)) < np.mean(np.abs(y)) * 0.05:
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
        """Cluster features to identify different speakers with improved sensitivity"""
        if len(features) < 2:
            return np.array([0] * len(features))
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Find optimal number of clusters with multiple methods
        best_n_clusters = 1
        best_score = -1
        cluster_scores = {}
        
        # Try different clustering approaches
        for n_clusters in range(2, min(self.max_speakers + 1, len(features)//2 + 1)):
            if n_clusters >= len(features):
                break
            
            # Try multiple initializations to find best clustering
            best_kmeans_score = -1
            best_kmeans_labels = None
            
            for init_attempt in range(5):  # Multiple random starts
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42+init_attempt, n_init=20)
                    cluster_labels = kmeans.fit_predict(features_normalized)
                    
                    # Check if we have reasonable cluster sizes (reduced minimum)
                    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                    if np.min(counts) >= 2:  # Reduced minimum cluster size
                        try:
                            sil_score = silhouette_score(features_normalized, cluster_labels)
                            
                            # Also calculate inertia-based score (within-cluster sum of squares)
                            inertia_score = 1.0 / (1.0 + kmeans.inertia_)
                            
                            # Combined score (weighted average)
                            combined_score = 0.7 * sil_score + 0.3 * inertia_score
                            
                            if combined_score > best_kmeans_score:
                                best_kmeans_score = combined_score
                                best_kmeans_labels = cluster_labels
                        except:
                            continue
                except:
                    continue
            
            if best_kmeans_labels is not None:
                cluster_scores[n_clusters] = (best_kmeans_score, best_kmeans_labels)
        
        # Select best clustering - prefer higher cluster counts if scores are close
        if cluster_scores:
            # Sort by score, but prefer more clusters if scores are within 0.1
            sorted_scores = sorted(cluster_scores.items(), key=lambda x: x[1][0], reverse=True)
            
            if len(sorted_scores) > 1:
                # If best score is close to second best, prefer more clusters
                best_score, best_labels = sorted_scores[0][1]
                for n_clusters, (score, labels) in sorted_scores[1:]:
                    if best_score - score < 0.1 and n_clusters > best_n_clusters:
                        best_n_clusters = n_clusters
                        cluster_labels = labels
                        break
                else:
                    best_n_clusters = sorted_scores[0][0]
                    cluster_labels = sorted_scores[0][1][1]
            else:
                best_n_clusters = sorted_scores[0][0]
                cluster_labels = sorted_scores[0][1][1]
            
            # Additional validation: check if speakers are actually different
            cluster_labels = self._validate_speaker_differences(features_normalized, cluster_labels)
            
        else:
            cluster_labels = np.zeros(len(features), dtype=int)
        
        return cluster_labels
    
    def _validate_speaker_differences(self, features, cluster_labels):
        """Validate that detected speakers are actually different"""
        unique_clusters = np.unique(cluster_labels)
        
        if len(unique_clusters) <= 1:
            return cluster_labels
        
        # Calculate centroid distances
        centroids = []
        for cluster_id in unique_clusters:
            cluster_features = features[cluster_labels == cluster_id]
            centroid = np.mean(cluster_features, axis=0)
            centroids.append(centroid)
        
        # Check if centroids are sufficiently different
        min_distance = float('inf')
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                min_distance = min(min_distance, distance)
        
        # If centroids are too similar, merge into single speaker
        distance_threshold = 1.5  # Adjusted threshold
        if min_distance < distance_threshold and len(unique_clusters) == 2:
            # For 2 speakers, be more strict about differences
            return np.zeros(len(cluster_labels), dtype=int)
        
        return cluster_labels
    
    def segment_by_speaker(self, y, sr, cluster_labels, timestamps):
        """Segment audio by identified speakers with improved segmentation"""
        window_samples = int(self.window_length * sr)
        hop_samples = int(self.hop_length * sr)
        
        # Apply smoothing to cluster labels to reduce noise
        cluster_labels = self._smooth_speaker_labels(cluster_labels)
        
        # Create speaker segments with better continuity
        speakers = {}
        
        if len(cluster_labels) == 0:
            return {0: y}
        
        # Track speaker changes more precisely
        for i, (timestamp, speaker) in enumerate(zip(timestamps, cluster_labels)):
            if speaker not in speakers:
                speakers[speaker] = []
            
            # Calculate sample position
            sample_start = int(timestamp * sr)
            sample_end = int((timestamp + self.window_length) * sr)
            sample_end = min(sample_end, len(y))
            
            if sample_end > sample_start:
                speakers[speaker].append(y[sample_start:sample_end])
        
        # Concatenate segments for each speaker and apply minimum duration
        speaker_audio = {}
        for speaker_id, segments in speakers.items():
            if segments:
                concatenated = np.concatenate(segments)
                duration = len(concatenated) / sr
                
                # Relaxed minimum duration check
                if duration >= self.min_speaker_duration:
                    speaker_audio[speaker_id] = concatenated
                elif len(speakers) <= 2:  # For 2 speakers, be more lenient
                    if duration >= 1.5:  # Minimum 1.5 seconds
                        speaker_audio[speaker_id] = concatenated
        
        # If no speakers meet criteria, return all audio as single speaker
        if not speaker_audio:
            speaker_audio[0] = y
        
        return speaker_audio
    
    def _smooth_speaker_labels(self, cluster_labels, window_size=3):
        """Apply smoothing to reduce noise in speaker assignments"""
        if len(cluster_labels) < window_size:
            return cluster_labels
        
        smoothed = cluster_labels.copy()
        
        for i in range(len(cluster_labels)):
            start = max(0, i - window_size // 2)
            end = min(len(cluster_labels), i + window_size // 2 + 1)
            
            window = cluster_labels[start:end]
            # Use most frequent speaker in window
            unique, counts = np.unique(window, return_counts=True)
            most_frequent = unique[np.argmax(counts)]
            smoothed[i] = most_frequent
        
        return smoothed
    
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