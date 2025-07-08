import librosa
import numpy as np
import scipy.signal
from typing import List, Tuple, Dict
import tensorflow as tf
from dataclasses import dataclass
import pretty_midi

@dataclass
class Note:
    """Represents a musical note with timing and pitch information"""
    start_time: float
    end_time: float
    pitch: int  # MIDI note number
    velocity: float
    instrument: str

@dataclass
class TabNote:
    """Represents a note in tablature format"""
    fret: int
    string: int
    start_time: float
    duration: float

class AudioPreprocessor:
    """Handles audio loading and basic preprocessing"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, float]:
        """Load audio file and return waveform and duration"""
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate)
            duration = len(audio) / sr
            return audio, duration
        except Exception as e:
            raise ValueError(f"Could not load audio file: {e}")
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract relevant audio features"""
        # Short-time Fourier transform
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        
        # Chromagram (pitch classes)
        chroma = librosa.feature.chroma_stft(S=magnitude, sr=self.sample_rate)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        
        return {
            'stft': stft,
            'magnitude': magnitude,
            'mel_spectrogram': mel_spec,
            'chroma': chroma,
            'spectral_centroid': spectral_centroid
        }

class PitchDetector:
    """Detects pitches in audio using various methods"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def detect_pitches_yin(self, audio: np.ndarray, 
                          fmin: float = 80.0, fmax: float = 2000.0) -> np.ndarray:
        """Use YIN algorithm for pitch detection"""
        pitches = librosa.yin(audio, 
                             fmin=fmin, 
                             fmax=fmax, 
                             sr=self.sample_rate)
        return pitches
    
    def detect_pitches_pyin(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use probabilistic YIN for more robust pitch detection"""
        pitches, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        return pitches, voiced_flag
    
    def hz_to_midi(self, frequencies: np.ndarray) -> np.ndarray:
        """Convert frequencies to MIDI note numbers"""
        # Remove NaN values and convert
        valid_freqs = frequencies[~np.isnan(frequencies)]
        if len(valid_freqs) == 0:
            return np.array([])
        
        midi_notes = librosa.hz_to_midi(valid_freqs)
        return np.round(midi_notes).astype(int)

class OnsetDetector:
    """Detects note onsets in audio"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def detect_onsets(self, audio: np.ndarray) -> np.ndarray:
        """Detect note onset times"""
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=self.sample_rate,
            units='time'
        )
        return onset_frames

class BasicSourceSeparator:
    """Basic source separation using spectral masking"""
    
    def __init__(self):
        pass
    
    def separate_harmonic_percussive(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separate harmonic and percussive components"""
        harmonic, percussive = librosa.effects.hpss(audio)
        return harmonic, percussive
    
    def isolate_frequency_range(self, audio: np.ndarray, 
                               low_freq: float, high_freq: float,
                               sample_rate: int) -> np.ndarray:
        """Isolate specific frequency range (basic bandpass filter)"""
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        filtered = scipy.signal.filtfilt(b, a, audio)
        return filtered

class GuitarTabGenerator:
    """Generates guitar tablature from detected notes"""
    
    def __init__(self):
        # Standard guitar tuning (EADGBE) in MIDI note numbers
        self.tuning = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
        self.num_strings = 6
        self.max_fret = 24
    
    def note_to_tab(self, midi_note: int) -> List[TabNote]:
        """Convert MIDI note to possible guitar tab positions"""
        possible_positions = []
        
        for string_idx, open_note in enumerate(self.tuning):
            fret = midi_note - open_note
            if 0 <= fret <= self.max_fret:
                possible_positions.append((string_idx, fret))
        
        return possible_positions
    
    def optimize_fingering(self, notes: List[Note]) -> List[TabNote]:
        """Choose optimal fingering positions for a sequence of notes"""
        tab_notes = []
        current_position = 0  # Current fret position
        
        for note in notes:
            positions = self.note_to_tab(note.pitch)
            if not positions:
                continue
            
            # Choose position closest to current hand position
            best_position = min(positions, 
                               key=lambda pos: abs(pos[1] - current_position))
            
            string_idx, fret = best_position
            tab_note = TabNote(
                fret=fret,
                string=string_idx,
                start_time=note.start_time,
                duration=note.end_time - note.start_time
            )
            tab_notes.append(tab_note)
            current_position = fret
        
        return tab_notes
    
    def format_tab(self, tab_notes: List[TabNote], duration: float) -> str:
        """Format tab notes into ASCII tablature"""
        # Create time grid
        time_resolution = 0.1  # 100ms resolution
        num_positions = int(duration / time_resolution) + 1
        
        # Initialize tab lines
        tab_lines = ['-' * num_positions for _ in range(self.num_strings)]
        
        for tab_note in tab_notes:
            pos = int(tab_note.start_time / time_resolution)
            if pos < num_positions:
                fret_str = str(tab_note.fret) if tab_note.fret < 10 else chr(ord('A') + tab_note.fret - 10)
                tab_lines[tab_note.string] = tab_lines[tab_note.string][:pos] + fret_str + tab_lines[tab_note.string][pos+1:]
        
        # Format with string labels
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        formatted_tab = []
        for i, line in enumerate(tab_lines):
            formatted_tab.append(f"{string_names[i]}|{line}")
        
        return '\n'.join(formatted_tab)

class MusicTranscriptionSystem:
    """Main system that orchestrates the transcription process"""
    
    def __init__(self, sample_rate: int = 22050):
        self.preprocessor = AudioPreprocessor(sample_rate)
        self.pitch_detector = PitchDetector(sample_rate)
        self.onset_detector = OnsetDetector(sample_rate)
        self.source_separator = BasicSourceSeparator()
        self.guitar_tab_generator = GuitarTabGenerator()
        self.sample_rate = sample_rate
    
    def transcribe_to_guitar_tab(self, audio_filepath: str) -> str:
        """Main method to transcribe audio to guitar tablature"""
        
        # Load and preprocess audio
        print("Loading audio...")
        audio, duration = self.preprocessor.load_audio(audio_filepath)
        
        # Separate harmonic content (removes drums/percussion)
        print("Separating audio components...")
        harmonic, _ = self.source_separator.separate_harmonic_percussive(audio)
        
        # Focus on guitar frequency range (roughly 80Hz - 2000Hz)
        guitar_audio = self.source_separator.isolate_frequency_range(
            harmonic, 80, 2000, self.sample_rate
        )
        
        # Detect note onsets
        print("Detecting note onsets...")
        onsets = self.onset_detector.detect_onsets(guitar_audio)
        
        # Detect pitches
        print("Detecting pitches...")
        pitches, voiced = self.pitch_detector.detect_