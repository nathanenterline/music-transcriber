import librosa
import numpy as np
import tensorflow as tf
import torch
import scipy.signal
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

class AdvancedSourceSeparation:
    """Advanced source separation using multiple techniques"""
    
    def __init__(self):
        self.demucs_model = None
        self.load_pretrained_models()
    
    def load_pretrained_models(self):
        """Load pre-trained separation models"""
        try:
            # Try to load Demucs model (requires demucs package)
            # pip install demucs
            from demucs.pretrained import get_model
            self.demucs_model = get_model('htdemucs')
            print("Loaded Demucs model successfully")
        except ImportError:
            print("Demucs not available, using fallback methods")
    
    def separate_with_demucs(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Use Demucs for high-quality source separation"""
        if self.demucs_model is None:
            return self.fallback_separation(audio, sr)
        
        try:
            from demucs.apply import apply_model
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            
            # Apply model
            with torch.no_grad():
                sources = apply_model(self.demucs_model, audio_tensor)
            
            # Convert back to numpy
            stems = {}
            stem_names = ['drums', 'bass', 'other', 'vocals']
            for i, name in enumerate(stem_names):
                stems[name] = sources[0, i].numpy()
            
            return stems
        except Exception as e:
            print(f"Demucs separation failed: {e}")
            return self.fallback_separation(audio, sr)
    
    def fallback_separation(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Fallback separation using traditional methods"""
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio, margin=3.0)
        
        # Frequency-based separation
        bass = self.isolate_bass(audio, sr)
        mid_range = self.isolate_mid_range(audio, sr)
        high_range = self.isolate_high_range(audio, sr)
        
        return {
            'harmonic': harmonic,
            'percussive': percussive,
            'bass': bass,
            'mid': mid_range,
            'high': high_range
        }
    
    def isolate_bass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Isolate bass frequencies (20-250 Hz)"""
        return self.bandpass_filter(audio, 20, 250, sr)
    
    def isolate_mid_range(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Isolate mid-range frequencies (250-4000 Hz)"""
        return self.bandpass_filter(audio, 250, 4000, sr)
    
    def isolate_high_range(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Isolate high frequencies (4000+ Hz)"""
        return self.highpass_filter(audio, 4000, sr)
    
    def bandpass_filter(self, audio: np.ndarray, low: float, high: float, sr: int) -> np.ndarray:
        """Apply bandpass filter"""
        nyquist = sr / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = scipy.signal.butter(4, [low_norm, high_norm], btype='band')
        return scipy.signal.filtfilt(b, a, audio)
    
    def highpass_filter(self, audio: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
        """Apply highpass filter"""
        nyquist = sr / 2
        cutoff_norm = cutoff / nyquist
        b, a = scipy.signal.butter(4, cutoff_norm, btype='high')
        return scipy.signal.filtfilt(b, a, audio)

class PolyphonicPitchDetection:
    """Advanced polyphonic pitch detection"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
    
    def multi_f0_estimation(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate multiple fundamental frequencies"""
        # Compute spectral features
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Apply harmonic enhancement
        enhanced_spec = self.harmonic_enhancement(magnitude)
        
        # Peak picking in frequency domain
        f0_candidates = self.spectral_peak_picking(enhanced_spec)
        
        # Temporal smoothing and tracking
        f0_tracks = self.track_f0_candidates(f0_candidates)
        
        return f0_tracks, enhanced_spec
    
    def harmonic_enhancement(self, magnitude: np.ndarray) -> np.ndarray:
        """Enhance harmonic content for better F0 detection"""
        enhanced = np.zeros_like(magnitude)
        
        for harmonic in range(1, 6):  # Consider first 5 harmonics
            # Shift and add harmonics
            shifted = np.roll(magnitude, -harmonic * 12, axis=0)  # Approximate octave shift
            enhanced += shifted / harmonic
        
        return enhanced
    
    def spectral_peak_picking(self, magnitude: np.ndarray) -> List[np.ndarray]:
        """Pick spectral peaks as F0 candidates"""
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        f0_candidates = []
        
        for frame_idx in range(magnitude.shape[1]):
            frame = magnitude[:, frame_idx]
            
            # Find peaks
            peaks, properties = scipy.signal.find_peaks(
                frame, 
                height=np.max(frame) * 0.1,  # Minimum height
                distance=10,  # Minimum distance between peaks
                prominence=np.max(frame) * 0.05
            )
            
            # Convert bin indices to frequencies
            peak_freqs = freqs[peaks]
            
            # Filter to musical range
            musical_freqs = peak_freqs[(peak_freqs >= 80) & (peak_freqs <= 2000)]
            
            f0_candidates.append(musical_freqs)
        
        return f0_candidates
    
    def track_f0_candidates(self, f0_candidates: List[np.ndarray]) -> np.ndarray:
        """Track F0 candidates across time using dynamic programming"""
        if not f0_candidates:
            return np.array([])
        
        # Simple tracking: connect nearest frequencies
        tracks = []
        current_tracks = []
        
        for frame_candidates in f0_candidates:
            new_tracks = []
            
            if not current_tracks:
                # Initialize tracks
                for freq in frame_candidates:
                    new_tracks.append([freq])
            else:
                # Connect to existing tracks
                used_candidates = set()
                
                for track in current_tracks:
                    last_freq = track[-1]
                    
                    # Find closest candidate
                    if len(frame_candidates) > 0:
                        distances = np.abs(frame_candidates - last_freq)
                        closest_idx = np.argmin(distances)
                        closest_freq = frame_candidates[closest_idx]
                        
                        # Connect if close enough (within semitone)
                        if distances[closest_idx] < last_freq * 0.06 and closest_idx not in used_candidates:
                            track.append(closest_freq)
                            new_tracks.append(track)
                            used_candidates.add(closest_idx)
                        else:
                            # End track
                            tracks.append(track)
                
                # Start new tracks for unused candidates
                for i, freq in enumerate(frame_candidates):
                    if i not in used_candidates:
                        new_tracks.append([freq])
            
            current_tracks = new_tracks
        
        # Add remaining tracks
        tracks.extend(current_tracks)
        
        # Convert to array format
        max_length = max(len(track) for track in tracks) if tracks else 0
        result = np.full((len(tracks), max_length), np.nan)
        
        for i, track in enumerate(tracks):
            result[i, :len(track)] = track
        
        return result

class RobustAudioPreprocessing:
    """Robust preprocessing for real-world audio"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply comprehensive preprocessing pipeline"""
        # Step 1: Normalize audio
        audio = self.normalize_audio(audio)
        
        # Step 2: Reduce noise
        audio = self.spectral_subtraction_denoising(audio)
        
        # Step 3: Dynamic range compression
        audio = self.soft_compression(audio)
        
        # Step 4: Spectral whitening
        audio = self.spectral_whitening(audio)
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95
        return audio
    
    def spectral_subtraction_denoising(self, audio: np.ndarray) -> np.ndarray:
        """Remove noise using spectral subtraction"""
        # Estimate noise from first 0.5 seconds
        noise_duration = min(int(0.5 * self.sr), len(audio) // 4)
        noise_sample = audio[:noise_duration]
        
        # Compute STFT
        stft = librosa.stft(audio)
        noise_stft = librosa.stft(noise_sample)
        
        # Estimate noise spectrum
        noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        
        # Apply spectral subtraction
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Subtract noise estimate
        clean_magnitude = magnitude - 2.0 * noise_magnitude
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft)
        
        return clean_audio
    
    def soft_compression(self, audio: np.ndarray, ratio: float = 4.0, 
                        threshold: float = 0.5) -> np.ndarray:
        """Apply soft compression to reduce dynamic range"""
        compressed = np.copy(audio)
        
        # Find samples above threshold
        above_threshold = np.abs(compressed) > threshold
        
        # Apply compression
        sign = np.sign(compressed[above_threshold])
        magnitude = np.abs(compressed[above_threshold])
        
        # Soft compression formula
        compressed_magnitude = (
            threshold + (magnitude - threshold) / ratio
        )
        
        compressed[above_threshold] = sign * compressed_magnitude
        
        return compressed
    
    def spectral_whitening(self, audio: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """Apply spectral whitening to reduce masking effects"""
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Compute local mean magnitude
        smoothed_magnitude = scipy.ndimage.uniform_filter(
            magnitude, size=(5, 5), mode='constant'
        )
        
        # Apply whitening
        whitened_magnitude = (
            magnitude ** alpha / 
            (smoothed_magnitude ** (alpha - 1) + 1e-10)
        )
        
        # Reconstruct
        whitened_stft = whitened_magnitude * np.exp(1j * phase)
        whitened_audio = librosa.istft(whitened_stft)
        
        return whitened_audio

class InstrumentSpecificProcessor:
    """Handle instrument-specific characteristics"""
    
    def __init__(self):
        self.guitar_processor = GuitarProcessor()
        self.piano_processor = PianoProcessor()
    
    def process_guitar(self, audio: np.ndarray, sr: int) -> Dict:
        """Process guitar-specific techniques"""
        return self.guitar_processor.analyze(audio, sr)
    
    def process_piano(self, audio: np.ndarray, sr: int) -> Dict:
        """Process piano-specific techniques"""
        return self.piano_processor.analyze(audio, sr)

class GuitarProcessor:
    """Guitar-specific processing"""
    
    def analyze(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze guitar-specific techniques"""
        # Detect string bends
        bends = self.detect_bends(audio, sr)
        
        # Detect harmonics
        harmonics = self.detect_harmonics(audio, sr)
        
        # Detect slides
        slides = self.detect_slides(audio, sr)
        
        return {
            'bends': bends,
            'harmonics': harmonics,
            'slides': slides
        }
    
    def detect_bends(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect string bends (pitch slides)"""
        # Get pitch contour with high time resolution
        pitches = librosa.yin(audio, fmin=80, fmax=2000, sr=sr)
        
        bends = []
        for i in range(1, len(pitches)):
            if not (np.isnan(pitches[i]) or np.isnan(pitches[i-1])):
                pitch_change = pitches[i] - pitches[i-1]
                # Detect significant pitch changes (bends)
                if abs(pitch_change) > 20:  # More than ~20 cents
                    bends.append({
                        'time': i * 512 / sr,  # Convert frame to time
                        'start_pitch': pitches[i-1],
                        'end_pitch': pitches[i],
                        'type': 'bend_up' if pitch_change > 0 else 'bend_down'
                    })
        
        return bends
    
    def detect_harmonics(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect natural and artificial harmonics"""
        # Harmonics have specific spectral characteristics
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        harmonics = []
        # Look for strong components at harmonic ratios
        # This is a simplified detection - real implementation would be more complex
        
        return harmonics
    
    def detect_slides(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect slides and glissandos"""
        # Similar to bends but with different duration and pitch change characteristics
        slides = []
        # Implementation would analyze pitch contours for slide patterns
        return slides

class PianoProcessor:
    """Piano-specific processing"""
    
    def analyze(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze piano-specific techniques"""
        # Detect sustain pedal usage
        sustain = self.detect_sustain_pedal(audio, sr)
        
        # Detect overlapping notes
        overlaps = self.detect_note_overlaps(audio, sr)
        
        return {
            'sustain_pedal': sustain,
            'note_overlaps': overlaps
        }
    
    def detect_sustain_pedal(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect sustain pedal usage from decay characteristics"""
        # Analyze decay rates of notes to infer pedal usage
        pedal_events = []
        # Complex implementation would analyze spectral decay
        return pedal_events
    
    def detect_note_overlaps(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect overlapping notes in polyphonic piano music"""
        overlaps = []
        # Implementation would use polyphonic transcription
        return overlaps

class IntegratedTranscriptionSystem:
    """Integrated system combining all advanced techniques"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.preprocessor = RobustAudioPreprocessing(sr)
        self.separator = AdvancedSourceSeparation()
        self.pitch_detector = PolyphonicPitchDetection(sr)
        self.instrument_processor = InstrumentSpecificProcessor()
    
    def transcribe(self, audio_path: str, target_instrument: str = 'guitar') -> Dict:
        """Complete transcription pipeline"""
        
        print("Loading audio...")
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        print("Preprocessing audio...")
        clean_audio = self.preprocessor.preprocess_audio(audio)
        
        print("Separating sources...")
        separated = self.separator.separate_with_demucs(clean_audio, self.sr)
        
        # Select appropriate stem
        if target_instrument == 'guitar':
            target_audio = separated.get('other', clean_audio)
        elif target_instrument == 'bass':
            target_audio = separated.get('bass', clean_audio)
        else:
            target_audio = clean_audio
        
        print("Detecting pitches...")
        pitches, spectrogram = self.pitch_detector.multi_f0_estimation(target_audio)
        
        print("Processing instrument-specific features...")
        if target_instrument == 'guitar':
            instrument_features = self.instrument_processor.process_guitar(target_audio, self.sr)
        elif target_instrument == 'piano':
            instrument_features = self.instrument_processor.process_piano(target_audio, self.sr)
        else:
            instrument_features = {}
        
        return {
            'pitches': pitches,
            'spectrogram': spectrogram,
            'separated_audio': separated,
            'instrument_features': instrument_features,
            'sample_rate': self.sr
        }

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = IntegratedTranscriptionSystem()
    
    # Transcribe audio file
    # results = system.transcribe('path/to/your/audio.wav', target_instrument='guitar')
    
    print("Advanced transcription system ready!")
    print("Use system.transcribe('audio_file.wav', 'guitar') to transcribe audio")
