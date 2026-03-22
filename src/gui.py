import collections
import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.audio_download import download_audio, get_cached_wav_path_for_url
from src.audio_processing import load_cached_or_legacy_audio, load_audio_mono, prepare_processed_audio
from src.config import MAX_MIDI, MIN_MIDI, SR
from src.microphone_engine import AudioEngine
from src.pitch_comparison import freq_to_note, midi_to_note_label, tuner_feedback

matplotlib.use("TkAgg")


class SingingPracticeGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MelodyMirror Studio - Real-time Pitch Match")
        self.root.geometry("980x700")

        self.audio_data = None
        self.reference_audio_data = None
        self.raw_audio_data = None
        self.downloaded_wav_path = None
        self.engine = None
        self.mic_plot_data = collections.deque(maxlen=240)
        self.song_plot_data = collections.deque(maxlen=240)

        self.url_var = tk.StringVar(value="")
        self.cache_status_var = tk.StringVar(value="")
        self.transpose_var = tk.DoubleVar(value=0.0)
        self.axis_margin_var = tk.DoubleVar(value=6.0)
        self.use_vocals_ref_var = tk.BooleanVar(value=True)
        self.use_vocals_play_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.demucs_progress_var = tk.StringVar(value="Demucs: idle")

        self.song_note_var = tk.StringVar(value="Song: -")
        self.mic_note_var = tk.StringVar(value="Mic: -")
        self.diff_var = tk.StringVar(value="Diff: -")
        self.feedback_var = tk.StringVar(value="Feedback: -")
        self.playback_pos_var = tk.DoubleVar(value=0.0)
        self.play_pause_var = tk.StringVar(value="Play")
        self.playback_time_var = tk.StringVar(value="00:00 / 00:00")
        self.seek_dragging = False
        self.track_duration_seconds = 0.0

        self._build_ui()
        self.url_var.trace_add("write", self._on_url_change)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._check_cache_for_url()
        self._poll_results()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(frame, text="Source")
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="YouTube URL").grid(row=0, column=0, padx=6, pady=8, sticky="w")
        ttk.Entry(controls, textvariable=self.url_var, width=80).grid(row=0, column=1, columnspan=5, padx=6, pady=8, sticky="we")

        ttk.Label(controls, textvariable=self.cache_status_var, foreground="#666666", font=("Segoe UI", 9)).grid(row=1, column=1, columnspan=5, padx=6, pady=(0, 8), sticky="w")

        ttk.Checkbutton(
            controls,
            text="Use extracted vocals for pitch reference",
            variable=self.use_vocals_ref_var,
        ).grid(row=2, column=0, columnspan=3, padx=6, pady=6, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Play extracted vocals (instead of original track)",
            variable=self.use_vocals_play_var,
        ).grid(row=2, column=3, columnspan=3, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="Transpose (semitones)").grid(row=3, column=0, padx=6, pady=8, sticky="w")
        ttk.Spinbox(
            controls,
            from_=-12,
            to=12,
            increment=1,
            textvariable=self.transpose_var,
            width=8,
        ).grid(row=3, column=1, padx=6, pady=8, sticky="w")

        ttk.Label(controls, text="Axis margin (semitones)").grid(row=3, column=2, padx=6, pady=8, sticky="w")
        ttk.Spinbox(
            controls,
            from_=2,
            to=24,
            increment=1,
            textvariable=self.axis_margin_var,
            width=8,
        ).grid(row=3, column=3, padx=6, pady=8, sticky="w")

        btns = ttk.Frame(controls)
        btns.grid(row=3, column=4, columnspan=2, sticky="e", padx=6, pady=(4, 8))

        ttk.Button(btns, text="Download and Load", command=self._download_and_process_song).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(controls, textvariable=self.demucs_progress_var).grid(row=4, column=0, padx=6, pady=(0, 4), sticky="w")
        self.demucs_progress = ttk.Progressbar(controls, orient=tk.HORIZONTAL, mode="indeterminate", length=260)
        self.demucs_progress.grid(row=4, column=1, columnspan=5, padx=6, pady=(0, 8), sticky="we")

        controls.columnconfigure(1, weight=2)
        controls.columnconfigure(2, weight=0)
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(4, weight=0)
        controls.columnconfigure(5, weight=0)

        stats = ttk.LabelFrame(frame, text="Live Pitch")
        stats.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(stats, textvariable=self.song_note_var, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.mic_note_var, font=("Segoe UI", 12, "bold")).grid(row=0, column=1, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.diff_var, font=("Segoe UI", 12)).grid(row=1, column=0, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.feedback_var, font=("Segoe UI", 12)).grid(row=1, column=1, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, padx=10, pady=8, sticky="w")

        playback = ttk.LabelFrame(frame, text="Playback")
        playback.pack(fill=tk.X, pady=(0, 10))

        self.playback_slider = ttk.Scale(
            playback,
            from_=0.0,
            to=100.0,
            orient=tk.HORIZONTAL,
            variable=self.playback_pos_var,
            command=self._on_seek_changed,
        )
        self.playback_slider.pack(fill=tk.X, padx=10, pady=(8, 4))
        self.playback_slider.bind("<ButtonPress-1>", self._on_seek_press)
        self.playback_slider.bind("<ButtonRelease-1>", self._on_seek_release)

        playback_controls = ttk.Frame(playback)
        playback_controls.pack(fill=tk.X, padx=10, pady=(0, 8))

        ttk.Button(
            playback_controls,
            textvariable=self.play_pause_var,
            command=self._toggle_play_pause,
            width=10,
        ).pack(side=tk.LEFT)
        ttk.Label(playback_controls, textvariable=self.playback_time_var).pack(side=tk.RIGHT)

        fig = Figure(figsize=(9.2, 3.8), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Mic vs Song Pitch (Notes)")
        self.ax.set_ylim(48, 84)
        self.ax.set_xlim(0, 240)
        self.ax.set_xlabel("Recent frames")
        self.ax.set_ylabel("Pitch (note / MIDI number)")
        self._set_pitch_axis_limits(48, 84)
        self.mic_plot_line, = self.ax.plot([], [], color="#2f80ed", linewidth=1.8, label="Mic")
        self.song_plot_line, = self.ax.plot([], [], color="#27ae60", linewidth=1.8, alpha=0.9, label="Song")
        self.ax.legend(loc="upper right")

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def _set_pitch_axis_limits(self, low_midi: float, high_midi: float):
        low = max(MIN_MIDI, int(np.floor(low_midi)))
        high = min(MAX_MIDI, int(np.ceil(high_midi)))
        if high <= low:
            high = min(MAX_MIDI, low + 12)

        self.ax.set_ylim(low, high)
        span = high - low
        step = 1 if span <= 12 else 2 if span <= 24 else 3
        ticks = np.arange(low, high + 1, step)
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels([midi_to_note_label(int(t)) for t in ticks])

    def _set_demucs_progress(self, active: bool):
        if active:
            self.demucs_progress_var.set("Demucs: processing...")
            self.demucs_progress.start(10)
        else:
            self.demucs_progress.stop()
            self.demucs_progress_var.set("Demucs: idle")

    def _check_cache_for_url(self):
        url = self.url_var.get().strip()
        if not url:
            self.cache_status_var.set("")
            return

        cached_wav_path = get_cached_wav_path_for_url(url)
        if os.path.exists(cached_wav_path):
            file_size_mb = os.path.getsize(cached_wav_path) / (1024 * 1024)
            self.cache_status_var.set(f"Cached: {file_size_mb:.1f} MB")
        else:
            self.cache_status_var.set("Not cached")

    def _on_url_change(self, *_args):
        self._check_cache_for_url()

    def _download_and_process_song(self):
        if self.engine is not None:
            self._stop()

        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Missing URL", "Please enter a YouTube URL.")
            return

        self.status_var.set("Downloading and loading audio...")

        def worker():
            try:
                wav_path, _from_cache = download_audio(url)
                y = load_audio_mono(wav_path, sr=SR)
                self.raw_audio_data = y
                self.downloaded_wav_path = wav_path
                self.root.after(0, self._check_cache_for_url)
                self.root.after(100, self._process_song)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Download failed", str(exc)))
                self.root.after(0, lambda: self.status_var.set("Download failed"))

        threading.Thread(target=worker, daemon=True).start()

    def _process_song(self):
        if self.engine is not None:
            self._stop()

        if self.raw_audio_data is None or len(self.raw_audio_data) == 0:
            url = self.url_var.get().strip()
            loaded_audio, loaded_path, load_msg = load_cached_or_legacy_audio(url, get_cached_wav_path_for_url)
            if loaded_audio is None:
                messagebox.showerror("No downloaded audio", "No cached file found for this URL. Click Download and Load first.")
                return
            self.raw_audio_data = loaded_audio
            self.downloaded_wav_path = loaded_path
            if load_msg:
                self.status_var.set(load_msg)

        use_vocals_reference = bool(self.use_vocals_ref_var.get())
        use_vocals_playback = bool(self.use_vocals_play_var.get())
        if use_vocals_reference:
            self.status_var.set("Processing transpose + extracting vocals with Demucs...")
        else:
            self.status_var.set("Processing transpose with direct-track pitch reference...")

        axis_margin = float(self.axis_margin_var.get())

        def worker():
            try:
                result = prepare_processed_audio(
                    raw_audio_data=self.raw_audio_data,
                    semitones=float(self.transpose_var.get()),
                    use_vocals_reference=use_vocals_reference,
                    use_vocals_playback=use_vocals_playback,
                    downloaded_wav_path=self.downloaded_wav_path,
                    axis_margin=axis_margin,
                    on_demucs_progress=lambda active: self.root.after(0, lambda: self._set_demucs_progress(active)),
                )

                self.audio_data = result["playback_audio"]
                self.reference_audio_data = result["reference_audio"]
                self.track_duration_seconds = result["track_duration_seconds"]

                def update_ui_after_process():
                    self.playback_slider.configure(to=max(1.0, self.track_duration_seconds))
                    self.playback_pos_var.set(0.0)
                    self.playback_time_var.set(f"00:00 / {self._format_time(self.track_duration_seconds)}")

                    low_lim, high_lim = result["axis_limits"]
                    self._set_pitch_axis_limits(low_lim, high_lim)

                    self.status_var.set(
                        f"Processed {len(self.audio_data) / SR:.1f}s at transpose {self.transpose_var.get():+.0f}. "
                        f"Playback: {result['playback_label']}. Pitch reference: {result['reference_label']}. "
                        f"{result['axis_text']}"
                    )

                self.root.after(0, update_ui_after_process)
            except Exception as exc:
                self.root.after(0, lambda: self._set_demucs_progress(False))
                self.root.after(0, lambda: messagebox.showerror("Process failed", str(exc)))
                self.root.after(0, lambda: self.status_var.set("Process failed"))

        threading.Thread(target=worker, daemon=True).start()

    def _start(self):
        if self.audio_data is None or len(self.audio_data) == 0:
            messagebox.showerror("No processed audio", "Download and Load first.")
            return
        if self.reference_audio_data is None or len(self.reference_audio_data) == 0:
            messagebox.showerror("No vocal reference", "Process first to extract vocals for pitch comparison.")
            return
        if self.engine is not None:
            return

        self.mic_plot_data.clear()
        self.song_plot_data.clear()
        self.engine = AudioEngine(self.audio_data, self.reference_audio_data, SR)

        try:
            self.engine.start()
            seek_to = float(self.playback_pos_var.get())
            if seek_to > 0.0:
                self.engine.set_position_seconds(seek_to)
            self.play_pause_var.set("Pause")
            self.status_var.set("Running: output and mic capture are synchronized in one callback")
        except Exception as exc:
            self.engine = None
            self.play_pause_var.set("Play")
            messagebox.showerror("Audio error", str(exc))
            self.status_var.set("Audio start failed")

    def _stop(self, status_text: str = "Stopped"):
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        self.play_pause_var.set("Play")
        self.status_var.set(status_text)

    def _toggle_play_pause(self):
        if self.engine is None:
            self._start()
        else:
            self._stop(status_text="Paused")

    def _poll_results(self):
        if self.engine is not None and not self.seek_dragging:
            pos_sec = self.engine.get_position_seconds()
            self.playback_pos_var.set(min(pos_sec, max(0.0, self.track_duration_seconds)))
            self.playback_time_var.set(
                f"{self._format_time(pos_sec)} / {self._format_time(self.track_duration_seconds)}"
            )

        if self.engine is not None:
            while True:
                try:
                    _idx, mic_pitch, song_pitch = self.engine.analysis_out.get_nowait()
                except queue.Empty:
                    break

                song_note, song_midi = freq_to_note(song_pitch)
                mic_note, mic_midi = freq_to_note(mic_pitch)

                self.song_note_var.set(f"Song: {song_note} ({song_pitch:.1f} Hz)" if song_pitch > 0 else "Song: -")
                self.mic_note_var.set(f"Mic:  {mic_note} ({mic_pitch:.1f} Hz)" if mic_pitch > 0 else "Mic: -")

                if song_midi is not None and mic_midi is not None:
                    diff = mic_midi - song_midi
                    self.diff_var.set(f"Diff: {diff:+d} semitone(s)")
                    self.feedback_var.set(f"Feedback: {tuner_feedback(diff)}")
                else:
                    self.diff_var.set("Diff: -")
                    self.feedback_var.set("Feedback: -")

                self.mic_plot_data.append(float(mic_midi) if mic_midi is not None else np.nan)
                self.song_plot_data.append(float(song_midi) if song_midi is not None else np.nan)

            self._refresh_plot()

            if self.engine is not None and not self.engine.running:
                self._stop()

        self.root.after(20, self._poll_results)

    def _format_time(self, seconds: float) -> str:
        total = max(0, int(seconds))
        mins = total // 60
        secs = total % 60
        return f"{mins:02d}:{secs:02d}"

    def _on_seek_press(self, _event):
        self.seek_dragging = True

    def _on_seek_release(self, _event):
        self.seek_dragging = False
        target_sec = float(self.playback_pos_var.get())
        if self.engine is not None:
            self.engine.set_position_seconds(target_sec)
        self.playback_time_var.set(f"{self._format_time(target_sec)} / {self._format_time(self.track_duration_seconds)}")

    def _on_seek_changed(self, value):
        if not self.seek_dragging:
            return
        try:
            pos = float(value)
        except ValueError:
            return
        self.playback_time_var.set(f"{self._format_time(pos)} / {self._format_time(self.track_duration_seconds)}")

    def _refresh_plot(self):
        if len(self.mic_plot_data) == 0 and len(self.song_plot_data) == 0:
            return
        mic_y = np.array(self.mic_plot_data, dtype=np.float32)
        song_y = np.array(self.song_plot_data, dtype=np.float32)
        x_mic = np.arange(len(mic_y), dtype=np.float32)
        x_song = np.arange(len(song_y), dtype=np.float32)
        self.mic_plot_line.set_data(x_mic, mic_y)
        self.song_plot_line.set_data(x_song, song_y)
        self.ax.set_xlim(0, max(240, len(mic_y), len(song_y)))
        self.canvas.draw_idle()

    def _on_close(self):
        self._stop()
        self.root.destroy()


def run_app():
    root = tk.Tk()
    SingingPracticeGUI(root)
    root.mainloop()
