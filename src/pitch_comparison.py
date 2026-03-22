import numpy as np
from typing import Optional

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Pitch extraction/comparison backend selector.
# Allowed values: "parselmouth", "librosa"
PITCH_BACKEND = "librosa"


class MidiRangeTracker:
    """Tracks the minimum and maximum MIDI values encountered during audio processing."""

    def __init__(self):
        self.min_midi = None
        self.max_midi = None
        self.valid_count = 0

    def update(self, midi_value: float) -> None:
        """Update tracker with a new MIDI value. Ignores 0, None, and non-finite values."""
        if midi_value is None or midi_value == 0 or not np.isfinite(midi_value):
            return

        if self.min_midi is None:
            self.min_midi = midi_value
            self.max_midi = midi_value
        else:
            self.min_midi = min(self.min_midi, midi_value)
            self.max_midi = max(self.max_midi, midi_value)

        self.valid_count += 1

    def get_range(self) -> Optional[tuple]:
        """Returns (min_midi, max_midi) or None if no valid values have been recorded."""
        if self.min_midi is None:
            return None
        return (self.min_midi, self.max_midi)

    def get_range_notes(self) -> str:
        """Returns a human-readable string of the detected MIDI range."""
        if self.min_midi is None:
            return "No data"

        min_note = midi_to_note_label(int(round(self.min_midi)))
        max_note = midi_to_note_label(int(round(self.max_midi)))
        return f"{min_note} - {max_note}"

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.min_midi = None
        self.max_midi = None
        self.valid_count = 0

    def has_data(self) -> bool:
        """Returns True if the tracker has recorded any valid MIDI values."""
        return self.valid_count > 0

try:
    import parselmouth

    USE_PARSELMOUTH = True
except Exception:
    USE_PARSELMOUTH = False


def midi_to_note_label(midi: int) -> str:
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave} ({midi})"


def freq_to_note(freq: float):
    if freq <= 0:
        return "-", None
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}", midi


def tuner_feedback(diff: int) -> str:
    if abs(diff) <= 0:
        return "In tune"
    if diff > 0:
        return "Sharp"
    return "Flat"


def estimate_pitch(block: np.ndarray, sr: int) -> float:
    if np.max(np.abs(block)) < 0.01:
        return 0.0

    backend = PITCH_BACKEND.strip().lower()

    if backend == "parselmouth":
        if not USE_PARSELMOUTH:
            raise RuntimeError("PITCH_BACKEND is 'parselmouth' but praat-parselmouth is not available.")
        try:
            snd = parselmouth.Sound(block.astype(np.float64), sampling_frequency=sr)
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=100.0, pitch_ceiling=500.0)
            vals = pitch.selected_array["frequency"]
            vals = vals[vals > 0]
            return float(np.median(vals)) if len(vals) else 0.0
        except Exception:
            return 0.0

    if backend == "librosa":
        try:
            import librosa

            f0 = librosa.yin(block.astype(np.float32), fmin=50, fmax=500, sr=sr, frame_length=1024, hop_length=256)
            f0 = f0[np.isfinite(f0)]
            return float(np.median(f0)) if len(f0) else 0.0
        except Exception:
            return 0.0

    raise RuntimeError("Invalid PITCH_BACKEND. Use 'parselmouth' or 'librosa'.")


def estimate_midi_range_from_audio(y: np.ndarray, sr: int):
    n_fft = 2048
    hop = 512
    if len(y) < n_fft:
        return None

    window = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fmin, fmax = 70.0, 550.0
    valid = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(valid) == 0:
        return None

    midi_vals = []
    for start in range(0, len(y) - n_fft + 1, hop):
        frame = y[start : start + n_fft].astype(np.float64)
        rms = np.sqrt(np.mean(frame * frame))
        if rms < 0.01:
            continue

        mag = np.abs(np.fft.rfft(frame * window))
        band = mag[valid]
        if len(band) == 0:
            continue

        peak_rel = int(np.argmax(band))
        peak = valid[peak_rel]
        freq = freqs[peak]
        if freq <= 0:
            continue

        midi = 69 + 12 * np.log2(freq / 440.0)
        if np.isfinite(midi):
            midi_vals.append(float(midi))

    if not midi_vals:
        return None

    arr = np.array(midi_vals, dtype=np.float64)
    low = float(np.percentile(arr, 5))
    high = float(np.percentile(arr, 95))
    if high <= low:
        high = low + 1.0
    return low, high
