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
        self.root.geometry("1320x760")
        self.root.minsize(1160, 680)

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

        self.vocal_ranges = [
            ("Soprano", "C4", "A5", "#ffe4ec"),
            ("Mezzo-soprano", "A3", "F#5", "#ffeedd"),
            ("Alto", "G3", "E5", "#fff7cc"),
            ("Contralto", "F3", "D5", "#f6f1cc"),
            ("Tenor", "C3", "A4", "#e8f7dd"),
            ("Baritone", "A2", "F4", "#dff3f2"),
            ("Bass", "F2", "E4", "#e7ebfb"),
        ]

        self._apply_theme()
        self._build_ui()
        self.url_var.trace_add("write", self._on_url_change)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._check_cache_for_url()
        self._poll_results()

    def _apply_theme(self):
        self.colors = {
            "bg": "#f4f7fb",
            "card": "#ffffff",
            "text": "#14213d",
            "muted": "#5b6475",
            "accent": "#0f8b8d",
            "accent_active": "#0c6f71",
            "line_mic": "#2f80ed",
            "line_song": "#27ae60",
            "grid": "#e7ecf3",
        }

        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure("Card.TLabelframe", background=self.colors["card"], borderwidth=1, relief="solid")
        style.configure(
            "Card.TLabelframe.Label",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure("Card.TFrame", background=self.colors["card"])

        style.configure("AppTitle.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI Semibold", 18))
        style.configure("AppSubtitle.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Segoe UI", 10))

        style.configure("TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Meta.TLabel", background=self.colors["card"], foreground=self.colors["muted"], font=("Segoe UI", 9))
        style.configure("StatValue.TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI Semibold", 12))

        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(14, 8),
            background=self.colors["accent"],
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_active"]), ("pressed", self.colors["accent_active"])],
            foreground=[("disabled", "#d8e0ea")],
        )

        style.configure("TEntry", fieldbackground="#ffffff", padding=6)
        style.configure("TSpinbox", arrowsize=14)
        style.configure("TCheckbutton", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI", 10))

        style.configure("TProgressbar", troughcolor="#dfe6ee", background=self.colors["accent"], lightcolor=self.colors["accent"], darkcolor=self.colors["accent"])
        style.configure("Horizontal.TScale", background=self.colors["card"], troughcolor="#dfe6ee")

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=14, style="App.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(frame, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="MelodyMirror Studio", style="AppTitle.TLabel").pack(anchor="w")
        ttk.Label(header, text="Refine pitch, track notes, and rehearse with live feedback.", style="AppSubtitle.TLabel").pack(anchor="w", pady=(2, 0))

        content = ttk.Frame(frame, style="App.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(content, style="App.TFrame")
        left_panel.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

        right_panel = ttk.Frame(content, style="App.TFrame")
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(left_panel, text="Settings", style="Card.TLabelframe", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="YouTube URL").grid(row=0, column=0, padx=6, pady=8, sticky="w")
        ttk.Entry(controls, textvariable=self.url_var, width=42).grid(row=1, column=0, columnspan=4, padx=6, pady=(0, 8), sticky="we")

        ttk.Label(controls, textvariable=self.cache_status_var, style="Meta.TLabel").grid(row=2, column=0, columnspan=4, padx=6, pady=(0, 8), sticky="w")

        ttk.Checkbutton(
            controls,
            text="Use extracted vocals for pitch reference",
            variable=self.use_vocals_ref_var,
        ).grid(row=3, column=0, columnspan=4, padx=6, pady=6, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Play extracted vocals (instead of original track)",
            variable=self.use_vocals_play_var,
        ).grid(row=4, column=0, columnspan=4, padx=6, pady=6, sticky="w")

        ttk.Label(controls, text="Transpose (semitones)").grid(row=5, column=0, padx=6, pady=(10, 6), sticky="w")
        ttk.Spinbox(
            controls,
            from_=-12,
            to=12,
            increment=1,
            textvariable=self.transpose_var,
            width=8,
        ).grid(row=5, column=1, padx=6, pady=(10, 6), sticky="w")

        ttk.Label(controls, text="Axis margin").grid(row=5, column=2, padx=6, pady=(10, 6), sticky="w")
        ttk.Spinbox(
            controls,
            from_=2,
            to=24,
            increment=1,
            textvariable=self.axis_margin_var,
            width=8,
        ).grid(row=5, column=3, padx=6, pady=(10, 6), sticky="w")

        btns = ttk.Frame(controls)
        btns.grid(row=6, column=0, columnspan=4, sticky="w", padx=6, pady=(10, 8))

        ttk.Button(btns, text="Load Audio", style="Accent.TButton", command=self._download_and_process_song).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(controls, textvariable=self.demucs_progress_var, style="Meta.TLabel").grid(row=7, column=0, columnspan=4, padx=6, pady=(0, 4), sticky="w")
        self.demucs_progress = ttk.Progressbar(controls, orient=tk.HORIZONTAL, mode="indeterminate", length=260)
        self.demucs_progress.grid(row=8, column=0, columnspan=4, padx=6, pady=(0, 8), sticky="we")

        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=0)
        controls.columnconfigure(2, weight=0)
        controls.columnconfigure(3, weight=0)

        stats = ttk.LabelFrame(left_panel, text="Live Pitch", style="Card.TLabelframe", padding=10)
        stats.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(stats, textvariable=self.song_note_var, style="StatValue.TLabel").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.mic_note_var, style="StatValue.TLabel").grid(row=0, column=1, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.diff_var, style="StatValue.TLabel").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.feedback_var, style="StatValue.TLabel").grid(row=1, column=1, padx=10, pady=8, sticky="w")
        ttk.Label(stats, textvariable=self.status_var, style="Meta.TLabel").grid(row=2, column=0, columnspan=2, padx=10, pady=8, sticky="w")
        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)

        playback = ttk.LabelFrame(right_panel, text="Playback", style="Card.TLabelframe", padding=10)
        playback.grid(row=0, column=0, sticky="ew", pady=(0, 10))

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
            style="Accent.TButton",
            width=10,
        ).pack(side=tk.LEFT)
        ttk.Label(playback_controls, textvariable=self.playback_time_var, style="Meta.TLabel").pack(side=tk.RIGHT)

        plot_card = ttk.LabelFrame(right_panel, text="Pitch Plot", style="Card.TLabelframe", padding=8)
        plot_card.grid(row=1, column=0, sticky="nsew")
        plot_card.rowconfigure(0, weight=1)
        plot_card.columnconfigure(0, weight=1)

        fig = Figure(figsize=(9.2, 4.8), dpi=100)
        fig.patch.set_facecolor(self.colors["card"])
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("#fbfcfe")
        self.ax.set_title("Mic vs Song Pitch (Notes)")
        self.ax.set_ylim(48, 84)
        self.ax.set_xlim(0, 240)
        self.ax.set_xlabel("Recent frames")
        self.ax.set_ylabel("Pitch (note / MIDI number)")
        self.ax.grid(axis="y", color=self.colors["grid"], linewidth=0.9)
        self._add_vocal_range_background()
        for side in ("top", "right"):
            self.ax.spines[side].set_visible(False)
        self.ax.spines["left"].set_color("#c9d5e3")
        self.ax.spines["bottom"].set_color("#c9d5e3")
        self._set_pitch_axis_limits(48, 84)
        self.mic_plot_line, = self.ax.plot([], [], color=self.colors["line_mic"], linewidth=2.0, label="Mic")
        self.song_plot_line, = self.ax.plot([], [], color=self.colors["line_song"], linewidth=2.0, alpha=0.95, label="Song")
        self.ax.legend(loc="upper right")

        canvas = FigureCanvasTkAgg(fig, master=plot_card)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.canvas = canvas

    def _note_to_midi(self, note: str) -> int:
        note_map = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
        }

        if len(note) < 2:
            raise ValueError(f"Invalid note format: {note}")

        if len(note) >= 3 and note[1] == "#":
            pitch = note[:2]
            octave = int(note[2:])
        else:
            pitch = note[0]
            octave = int(note[1:])

        return (octave + 1) * 12 + note_map[pitch]

    def _add_vocal_range_background(self):
        for name, low_note, high_note, color in self.vocal_ranges:
            low_midi = self._note_to_midi(low_note)
            high_midi = self._note_to_midi(high_note)
            self.ax.axhspan(low_midi, high_midi, color=color, alpha=0.45, zorder=0)
            self.ax.text(
                0.01,
                (low_midi + high_midi) / 2,
                name,
                transform=self.ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=8,
                color="#5b6475",
                alpha=0.9,
                zorder=1,
            )

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
                        f"Processed {len(self.audio_data) / SR:.1f}s at transpose {self.transpose_var.get():+.0f}.\n"
                        f"Playback: {result['playback_label']}. Pitch reference: {result['reference_label']}. \n"
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

                self.mic_plot_data.append(self._plot_midi_value(mic_midi))
                self.song_plot_data.append(self._plot_midi_value(song_midi))

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

    def _plot_midi_value(self, midi_value):
        if midi_value is None:
            return np.nan
        val = float(midi_value)
        if not np.isfinite(val):
            return np.nan
        if val < MIN_MIDI or val > MAX_MIDI:
            return np.nan
        return val

    def _refresh_plot(self):
        if len(self.mic_plot_data) == 0 and len(self.song_plot_data) == 0:
            return

        # Keep unvoiced/invalid frames as blank gaps on the plot.
        mic_y = np.array(self.mic_plot_data, dtype=np.float64)
        song_y = np.array(self.song_plot_data, dtype=np.float64)
        x_mic = np.arange(len(mic_y), dtype=np.float64)
        x_song = np.arange(len(song_y), dtype=np.float64)

        mic_masked = np.ma.masked_invalid(mic_y)
        song_masked = np.ma.masked_invalid(song_y)

        self.mic_plot_line.set_data(x_mic, mic_masked)
        self.song_plot_line.set_data(x_song, song_masked)
        self.ax.set_xlim(0, max(240, len(mic_y), len(song_y)))
        self.canvas.draw_idle()

    def _on_close(self):
        self._stop()
        self.root.destroy()


def run_app():
    root = tk.Tk()
    SingingPracticeGUI(root)
    root.mainloop()
