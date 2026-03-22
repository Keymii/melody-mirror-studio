import hashlib
import os

import yt_dlp


def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_cached_wav_path_for_url(url: str, out_dir: str = "audio_cache") -> str:
    return os.path.join(out_dir, f"{hash_url(url)}.wav")


def download_audio(url: str, out_dir: str = "audio_cache") -> tuple[str, bool]:
    os.makedirs(out_dir, exist_ok=True)
    out_wav = get_cached_wav_path_for_url(url, out_dir=out_dir)
    if os.path.exists(out_wav):
        return out_wav, True

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, f"{hash_url(url)}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out_wav):
        raise RuntimeError("Audio extraction failed. Ensure ffmpeg is installed and on PATH.")
    return out_wav, False
