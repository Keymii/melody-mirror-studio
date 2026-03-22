# MelodyMirror Studio

## Overview
MelodyMirror Studio is a desktop tool for real-time vocal practice and pitch matching. It lets you load a song from YouTube, optionally isolate vocals, transpose to a comfortable key, and compare live microphone pitch against the song reference while viewing feedback on note accuracy.

The application is designed for practical rehearsal workflows: quick song loading, playback controls, and immediate pitch-difference feedback during singing.

## Features
- Download audio from a YouTube URL with automatic local caching.
- Reuse previously cached tracks instantly when available.
- Optionally isolate vocals for a cleaner and more reliable pitch reference.
- Transpose tracks by semitones to match your preferred vocal range.
- Compare microphone pitch against the reference track in real time.
- Receive live note detection, semitone-difference, and tuning feedback.
- Control playback with responsive seek and play/pause interactions.

## Project Structure
- main.py: Minimal application entrypoint.
- gui.py: Tkinter GUI and interaction flow.
- audio_download.py: YouTube download and local cache helpers.
- vocal_extraction.py: Demucs-based vocal extraction helpers.
- audio_processing.py: Audio loading, transpose, and processing orchestration.
- microphone_engine.py: Audio stream engine and background pitch worker process.
- pitch_comparison.py: Pitch estimation and note-comparison helpers.
- config.py: Shared constants.

## Setup
### 1. Clone and enter the project
```powershell
git clone https://github.com/Keymii/melody-mirror-studio.git
cd melody-mirror-studio
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Ensure FFmpeg is installed
This project relies on FFmpeg for audio extraction via yt-dlp. Install FFmpeg and make sure it is available on your system PATH.

### 4. Run the app
```powershell
python main.py
```

## Notes on Dependencies
The requirements.txt file currently contains dependencies frozen from the development environment.

Because it was generated from a full dev environment snapshot, some packages may be unnecessary for runtime, transitive-only, or redundant for this project.

If you want to maintain a lean production dependency set, you can later curate this file to keep only the packages required by the app.

## Disclaimer
This project is intended for lawful fair-use and educational purposes.

Do not use this app to download, copy, distribute, or otherwise use copyrighted material without permission from the rights holder.

Users are solely responsible for complying with applicable laws and platform terms. The author and contributors do not endorse misuse and are not liable for unlawful use.