"""Interfaz local V1 para Screen Watcher."""

from __future__ import annotations

from dataclasses import asdict
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from watcher import Config, ROOT, WatcherWorker


class ScreenWatcherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Screen Watcher")
        self.minsize(880, 560)
        # Ventana amplia normal: conserva la composición completa sin maximizar.
        width = int(self.winfo_screenwidth() * 0.92)
        height = int(self.winfo_screenheight() * 0.90)
        x = max(20, (self.winfo_screenwidth() - width) // 2)
        y = max(20, (self.winfo_screenheight() - height) // 2)
        self._initial_geometry = f"{width}x{height}+{x}+{y}"
        self.state("normal")
        self.geometry(self._initial_geometry)
        self.configure(bg="#15181d")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.messages: Queue = Queue()
        self.worker: WatcherWorker | None = None
        self.worker_thread: Thread | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.restart_requested = False
        self.settings_window: tk.Toplevel | None = None
        self.fields: dict[str, tk.StringVar] = {}
        self.config_value = Config.load(ROOT / "config.json")
        self._configure_style()
        self._build()
        self.after(80, self._drain_messages)
        self.after_idle(self._apply_initial_window_size)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _apply_initial_window_size(self) -> None:
        """Evita que Windows restaure un estado maximizado de ejecuciones previas."""
        self.state("normal")
        self.geometry(self._initial_geometry)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#15181d")
        style.configure("Panel.TFrame", background="#20242b")
        style.configure("TLabel", background="#20242b", foreground="#e8edf2")
        style.configure("Title.TLabel", background="#15181d", foreground="#f5f7fa", font=("Segoe UI", 15, "bold"))
        style.configure("Status.TLabel", background="#15181d", foreground="#6ee7a1", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", background="#20242b", foreground="#9ca8b8", font=("Segoe UI", 9, "bold"))
        style.configure("Value.TLabel", background="#20242b", foreground="#ffffff", font=("Segoe UI", 11))
        style.configure("TButton", padding=7)
        style.configure("Treeview", background="#20242b", fieldbackground="#20242b", foreground="#e8edf2", rowheight=26)
        style.configure("Treeview.Heading", background="#303844", foreground="#dce5ef")

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        ttk.Label(header, text="SCREEN WATCHER", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Settings", command=self._open_settings).pack(side="right")
        self.status = ttk.Label(header, text="● STOPPED", style="Status.TLabel")
        self.status.pack(side="right", padx=14)
        self.latency = ttk.Label(header, text="— s", style="Status.TLabel")
        self.latency.pack(side="right", padx=18)

        main = ttk.Frame(self)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        preview_panel = ttk.Frame(main, style="Panel.TFrame")
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.preview_label = ttk.Label(preview_panel, text="La captura aparecerá al iniciar", anchor="center", style="Value.TLabel")
        self.preview_label.pack(fill="both", expand=True, padx=12, pady=12)

        side = ttk.Frame(main, style="Panel.TFrame")
        side.grid(row=0, column=1, sticky="nsew")
        ttk.Label(side, text="CURRENT", style="Header.TLabel").pack(anchor="w", padx=14, pady=(14, 8))
        self.app_value = self._detail(side, "App", "—")
        self.activity_value = self._detail(side, "Activity", "—")
        self.summary_value = self._detail(side, "Summary", "—")
        self.change_value = self._detail(side, "Visual change", "—")
        self.model_value = self._detail(side, "Model", self.config_value.model_name.rsplit("/", 1)[-1])
        self.start_button = ttk.Button(side, text="Start watching", command=self._toggle)
        self.start_button.pack(fill="x", padx=14, pady=(24, 14))
        timeline_panel = ttk.Frame(self, style="Panel.TFrame")
        timeline_panel.grid(row=2, column=0, sticky="ew", padx=18, pady=(10, 18), ipady=8)
        ttk.Label(timeline_panel, text="TIMELINE", style="Header.TLabel").pack(anchor="w", padx=12, pady=(0, 5))
        self.timeline = ttk.Treeview(timeline_panel, columns=("time", "app", "activity", "summary"), show="headings", height=4)
        for name, text, width in (("time", "Time", 70), ("app", "App", 180), ("activity", "Activity", 120), ("summary", "Summary", 700)):
            self.timeline.heading(name, text=text)
            self.timeline.column(name, width=width, anchor="w")
        self.timeline.pack(fill="both", expand=True, padx=10)

        # Solo la fila central crece; cabecera y timeline tienen altura estable.
        main.grid(row=1, column=0, sticky="nsew", padx=18)

    def _detail(self, parent, label: str, initial: str) -> ttk.Label:
        ttk.Label(parent, text=label, style="Header.TLabel").pack(anchor="w", padx=14, pady=(7, 0))
        value = ttk.Label(parent, text=initial, style="Value.TLabel", wraplength=240)
        value.pack(anchor="w", padx=14)
        return value

    def _load_config_fields(self) -> None:
        values = asdict(self.config_value)
        for key, variable in self.fields.items():
            value = values[key]
            variable.set(",".join(map(str, value)) if isinstance(value, tuple) else str(value))

    def _open_settings(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.focus_set()
            return
        window = tk.Toplevel(self)
        self.settings_window = window
        window.title("Screen Watcher settings")
        window.configure(bg="#20242b")
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()
        panel = ttk.Frame(window, style="Panel.TFrame", padding=18)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="CONFIGURATION", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        keys = (
            "model_name", "target_size", "max_new_tokens", "capture_interval_seconds",
            "mean_difference_threshold", "pixel_difference_threshold",
            "changed_ratio_threshold", "confirmations_required",
        )
        self.fields = {key: tk.StringVar() for key in keys}
        self._load_config_fields()
        options = {
            "model_name": ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
            "target_size": ("854,480", "1024,576", "1280,720", "1920,1080"),
            "max_new_tokens": ("32", "64", "128", "256", "512"),
            "capture_interval_seconds": ("1", "2", "5", "10"),
            "confirmations_required": ("1", "2", "3", "4"),
        }
        labels = (
            ("model_name", "Model"), ("target_size", "Resolution"), ("max_new_tokens", "Max tokens"),
            ("capture_interval_seconds", "Capture interval (s)"), ("confirmations_required", "Confirmations"),
            ("mean_difference_threshold", "Mean threshold"), ("pixel_difference_threshold", "Pixel threshold"),
            ("changed_ratio_threshold", "Change ratio"),
        )
        for row, (key, label) in enumerate(labels, start=1):
            ttk.Label(panel, text=label, style="Header.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 14), pady=5)
            if key in options:
                control = ttk.Combobox(panel, textvariable=self.fields[key], values=options[key], state="readonly", width=32)
            else:
                control = ttk.Spinbox(panel, textvariable=self.fields[key], width=34)
            control.grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Button(panel, text="Apply and restart", command=self._apply).grid(row=len(labels) + 1, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        window.protocol("WM_DELETE_WINDOW", window.destroy)

    def _toggle(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.start_button.configure(text="Stopping…", state="disabled")
        else:
            self._start()

    def _start(self) -> None:
        self.worker = WatcherWorker(self.config_value, self.messages.put)
        self.worker_thread = Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        self.start_button.configure(text="Stop watching")

    def _apply(self) -> None:
        try:
            width, height = (int(part.strip()) for part in self.fields["target_size"].get().split(","))
            self.config_value = Config(
                **{**asdict(self.config_value), "model_name": self.fields["model_name"].get().strip(),
                   "capture_interval_seconds": float(self.fields["capture_interval_seconds"].get()),
                   "target_size": (width, height), "max_new_tokens": int(self.fields["max_new_tokens"].get()),
                   "mean_difference_threshold": float(self.fields["mean_difference_threshold"].get()),
                   "pixel_difference_threshold": int(self.fields["pixel_difference_threshold"].get()),
                   "changed_ratio_threshold": float(self.fields["changed_ratio_threshold"].get()),
                   "confirmations_required": int(self.fields["confirmations_required"].get())}
            )
            self.config_value.save(ROOT / "config.json")
        except (ValueError, TypeError) as error:
            messagebox.showerror("Invalid configuration", f"Review the values: {error}")
            return
        self.model_value.configure(text=self.config_value.model_name.rsplit("/", 1)[-1])
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.destroy()
        self.settings_window = None
        if self.worker is not None:
            self.worker.stop()
            self.start_button.configure(text="Restarting…", state="disabled")
            self.restart_requested = True
        else:
            self._start()

    def _drain_messages(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                self._handle(message)
        except Empty:
            pass
        self.after(80, self._drain_messages)

    def _handle(self, message: dict) -> None:
        kind = message["type"]
        if kind == "status":
            value = message["value"]
            self.status.configure(text=f"● {value}")
            if value == "STOPPED":
                self.worker = None
                if self.restart_requested:
                    self.restart_requested = False
                    self.start_button.configure(state="normal")
                    self._start()
                else:
                    self.start_button.configure(text="Start watching", state="normal")
        elif kind == "latency":
            self.latency.configure(text=f"{message['seconds']:.1f} s")
        elif kind == "frame":
            image = Image.fromarray(message["frame"])
            # PhotoImage impone su tamaño solicitado al panel: limitar su altura
            # reserva espacio permanente para la timeline en pantallas bajas.
            image.thumbnail((680, 330), Image.Resampling.LANCZOS)
            self.preview = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview, text="")
        elif kind == "measurement":
            measurement = message["measurement"]
            self.change_value.configure(text=f"{measurement.changed_ratio:.1%} ({measurement.mean_difference:.1f})")
        elif kind == "state":
            state = message["state"]
            self.app_value.configure(text=state.app)
            self.activity_value.configure(text=state.activity)
            self.summary_value.configure(text=state.summary)
        elif kind == "event":
            event = message["event"]
            self.timeline.insert("", "end", values=(event.timestamp.strftime("%H:%M"), event.state.app, event.state.activity, event.state.summary))
            self.timeline.yview_moveto(1)
        elif kind == "error":
            self.status.configure(text="● ERROR")
            messagebox.showerror("Screen Watcher", message["message"])

    def _close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.destroy()


if __name__ == "__main__":
    ScreenWatcherApp().mainloop()
