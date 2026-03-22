import multiprocessing as mp
import queue
import threading

import numpy as np
import sounddevice as sd

from config import BLOCK, MAX_QUEUE_CHUNKS
from pitch_comparison import estimate_pitch


def queue_put_latest(q_obj, item):
    while True:
        try:
            q_obj.put_nowait(item)
            return
        except queue.Full:
            try:
                q_obj.get_nowait()
            except queue.Empty:
                return


def pitch_worker(input_q, result_q, stop_event, sr: int):
    while not stop_event.is_set():
        try:
            packet = input_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if packet is None:
            break

        idx, mic_block, song_block = packet
        mic_pitch = estimate_pitch(mic_block, sr)
        song_pitch = estimate_pitch(song_block, sr)
        queue_put_latest(result_q, (idx, mic_pitch, song_pitch))


class AudioEngine:
    def __init__(self, song_audio: np.ndarray, reference_audio: np.ndarray, sr: int):
        self.song_audio = song_audio.astype(np.float32, copy=False)
        self.reference_audio = reference_audio.astype(np.float32, copy=False)
        self.sr = sr
        self.position = 0
        self.chunk_idx = 0
        self.running = False
        self.position_lock = threading.Lock()

        self.analysis_in = mp.Queue(maxsize=MAX_QUEUE_CHUNKS)
        self.analysis_out = mp.Queue(maxsize=MAX_QUEUE_CHUNKS)
        self.stop_event = mp.Event()
        self.worker = mp.Process(
            target=pitch_worker,
            args=(self.analysis_in, self.analysis_out, self.stop_event, self.sr),
            daemon=True,
        )
        self.stream = None

    def start(self):
        self.running = True
        self.stop_event.clear()
        self.worker.start()
        self.stream = sd.Stream(
            samplerate=self.sr,
            blocksize=BLOCK,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.running = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

        self.stop_event.set()
        queue_put_latest(self.analysis_in, None)
        if self.worker.is_alive():
            self.worker.join(timeout=1.5)
        if self.worker.is_alive():
            self.worker.terminate()

    def _callback(self, indata, outdata, frames, _time_info, status):
        if status:
            pass

        with self.position_lock:
            start = self.position
            end = start + frames
            self.position = end

        chunk = self.song_audio[start:end]
        ref_chunk = self.reference_audio[start:end]

        if len(chunk) < frames:
            pad = np.zeros(frames - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        if len(ref_chunk) < frames:
            ref_chunk = np.concatenate([ref_chunk, np.zeros(frames - len(ref_chunk), dtype=np.float32)])

        if end >= len(self.song_audio):
            self.running = False

        outdata[:, 0] = chunk
        mic_chunk = indata[:, 0].copy()

        queue_put_latest(self.analysis_in, (self.chunk_idx, mic_chunk, ref_chunk.copy()))
        self.chunk_idx += 1

    def set_position_seconds(self, seconds: float):
        target = int(max(0.0, float(seconds)) * self.sr)
        target = min(target, max(0, len(self.song_audio) - 1))
        with self.position_lock:
            self.position = target

    def get_position_seconds(self) -> float:
        with self.position_lock:
            pos = self.position
        return float(pos) / float(self.sr)

    def get_duration_seconds(self) -> float:
        return float(len(self.song_audio)) / float(self.sr)
