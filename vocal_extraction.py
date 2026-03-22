import os

import librosa
import numpy as np
import soundfile as sf

DEMUCS_MODEL = "htdemucs"
_DEMUCS_MODEL_CACHE = None


def _get_demucs_model():
    global _DEMUCS_MODEL_CACHE
    if _DEMUCS_MODEL_CACHE is not None:
        return _DEMUCS_MODEL_CACHE

    try:
        from demucs.pretrained import get_model
    except Exception as exc:
        raise RuntimeError(
            "Demucs is required for vocal extraction. Install it with 'pip install demucs'."
        ) from exc

    try:
        model = get_model(name=DEMUCS_MODEL)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Demucs model. Ensure internet access on first run so model weights can download."
        ) from exc

    model.cpu()
    model.eval()
    _DEMUCS_MODEL_CACHE = model
    return model


def get_demucs_cache_path(source_wav_path: str) -> str:
    root, _ = os.path.splitext(source_wav_path)
    return f"{root}.demucs_vocals.wav"


def extract_vocals_reference(y: np.ndarray, sr: int, cache_path: str | None = None) -> np.ndarray:
    if len(y) == 0:
        return y.astype(np.float32, copy=False)

    if cache_path and os.path.exists(cache_path):
        cached, _ = librosa.load(cache_path, sr=sr, mono=True)
        cached = np.clip(cached, -1.0, 1.0).astype(np.float32, copy=False)
        if len(cached) > len(y):
            cached = cached[: len(y)]
        elif len(cached) < len(y):
            cached = np.pad(cached, (0, len(y) - len(cached)))
        return cached

    try:
        import torch as th
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
    except Exception as exc:
        raise RuntimeError(
            "Demucs runtime dependencies are unavailable. Reinstall with 'pip install demucs'."
        ) from exc

    model = _get_demucs_model()

    wav = th.from_numpy(y.astype(np.float32, copy=False)).unsqueeze(0)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

    ref = wav.mean(0)
    wav = wav - ref.mean()
    scale = float(ref.std().item())
    if scale < 1e-8:
        scale = 1.0
    wav = wav / scale
    demucs_device = "cuda" if th.cuda.is_available() else "cpu"

    try:
        sources = apply_model(
            model,
            wav[None],
            device=demucs_device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
            num_workers=0,
            segment=None,
        )[0]
    except Exception as exc:
        raise RuntimeError(
            "Demucs separation failed during inference. Ensure model weights are available and memory is sufficient."
        ) from exc

    sources = sources * scale
    sources = sources + ref.mean()

    if "vocals" not in model.sources:
        raise RuntimeError("Demucs model did not expose a 'vocals' stem.")
    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx].mean(0).detach().cpu().numpy().astype(np.float32, copy=False)

    if int(model.samplerate) != int(sr):
        vocals = librosa.resample(vocals, orig_sr=int(model.samplerate), target_sr=int(sr)).astype(np.float32, copy=False)

    if len(vocals) > len(y):
        vocals = vocals[: len(y)]
    elif len(vocals) < len(y):
        vocals = np.pad(vocals, (0, len(y) - len(vocals)))

    vocals = np.clip(vocals, -1.0, 1.0).astype(np.float32, copy=False)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        sf.write(cache_path, vocals, sr, subtype="PCM_16")
    return vocals
