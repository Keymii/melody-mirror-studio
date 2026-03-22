import os

import librosa
import numpy as np

from config import SR
from pitch_comparison import estimate_midi_range_from_audio
from vocal_extraction import extract_vocals_reference, get_demucs_cache_path


def load_audio_mono(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


def transpose_audio(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-6:
        return y.astype(np.float32, copy=False)

    y32 = y.astype(np.float32, copy=False)

    try:
        shifted = librosa.effects.pitch_shift(
            y32,
            sr=sr,
            n_steps=float(semitones),
            bins_per_octave=12,
            res_type="fft",
        )
    except Exception as exc:
        err_text = str(exc).lower()
        if "numpy.dtype size changed" not in err_text and "binary incompatibility" not in err_text:
            raise

        ratio = 2.0 ** (float(semitones) / 12.0)
        n_fft = 2048
        hop = 512

        stft = librosa.stft(y32, n_fft=n_fft, hop_length=hop)
        stretched_stft = librosa.phase_vocoder(stft, rate=(1.0 / ratio), hop_length=hop)
        stretched = librosa.istft(stretched_stft, hop_length=hop)

        src = np.linspace(0, max(0, len(stretched) - 1), num=len(y32), dtype=np.float64)
        shifted = np.interp(src, np.arange(len(stretched), dtype=np.float64), stretched.astype(np.float64))

    if len(shifted) > len(y32):
        shifted = shifted[: len(y32)]
    elif len(shifted) < len(y32):
        shifted = np.pad(shifted, (0, len(y32) - len(shifted)))

    return np.clip(shifted, -1.0, 1.0).astype(np.float32, copy=False)


def load_cached_or_legacy_audio(url: str, get_cached_wav_path_for_url_fn, legacy_path: str = "audio.wav"):
    cached_url_wav = get_cached_wav_path_for_url_fn(url) if url else ""

    if cached_url_wav and os.path.exists(cached_url_wav):
        y = load_audio_mono(cached_url_wav, sr=SR)
        return y, cached_url_wav, f"Loaded cached audio {len(y) / SR:.1f}s. Processing..."

    if os.path.exists(legacy_path):
        y = load_audio_mono(legacy_path, sr=SR)
        return y, legacy_path, None

    return None, None, None


def prepare_processed_audio(
    raw_audio_data: np.ndarray,
    semitones: float,
    use_vocals_reference: bool,
    use_vocals_playback: bool,
    downloaded_wav_path: str | None,
    axis_margin: float,
    on_demucs_progress=None,
):
    transposed = transpose_audio(raw_audio_data, SR, semitones)
    transposed = np.clip(transposed, -1.0, 1.0).astype(np.float32, copy=False)
    vocals_cache = None

    if use_vocals_reference or use_vocals_playback:
        demucs_cache_path = None
        if downloaded_wav_path:
            demucs_cache_path = get_demucs_cache_path(downloaded_wav_path)

        cache_exists = bool(demucs_cache_path and os.path.exists(demucs_cache_path))
        if on_demucs_progress and not cache_exists:
            on_demucs_progress(True)
        try:
            extracted_original = extract_vocals_reference(
                raw_audio_data,
                SR,
                cache_path=demucs_cache_path,
            )
        finally:
            if on_demucs_progress and not cache_exists:
                on_demucs_progress(False)

        vocals_cache = transpose_audio(extracted_original, SR, semitones)
        vocals_cache = np.clip(vocals_cache, -1.0, 1.0).astype(np.float32, copy=False)

    if use_vocals_reference:
        reference_audio = vocals_cache
        reference_label = "Demucs vocals only"
        axis_source_audio = vocals_cache
        axis_source_label = "extracted vocals"
    else:
        reference_audio = transposed
        reference_label = "direct file"
        axis_source_audio = transposed
        axis_source_label = "original track"

    if use_vocals_playback:
        playback_audio = vocals_cache
        playback_label = "vocals"
    else:
        playback_audio = transposed
        playback_label = "original"

    midi_range = estimate_midi_range_from_audio(axis_source_audio, SR)
    axis_text = "Axis: default range"
    axis_limits = (48.0, 84.0)

    if midi_range is not None:
        low, high = midi_range
        low_lim = low - axis_margin
        high_lim = high + axis_margin
        axis_limits = (low_lim, high_lim)
        axis_text = f"Axis ({axis_source_label}): {low_lim:.1f}..{high_lim:.1f} MIDI"

    return {
        "playback_audio": playback_audio,
        "reference_audio": reference_audio,
        "track_duration_seconds": len(playback_audio) / SR,
        "playback_label": playback_label,
        "reference_label": reference_label,
        "axis_limits": axis_limits,
        "axis_text": axis_text,
    }
