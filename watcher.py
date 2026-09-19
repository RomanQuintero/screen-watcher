"""Screen Watcher: observa localmente y registra cambios semánticos."""

from __future__ import annotations

import json
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

import dxcam
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from activity import ActivityEvent, ActivityState, EventHistory, normalize_activity
from change_detector import VisualChangeDetector


ROOT = Path(__file__).parent
PROMPT = '''Analiza la captura de pantalla y responde solo con un objeto JSON válido, sin Markdown.
El objeto tiene, en este orden, las claves app, activity y summary.
app: nombre canónico de la aplicación o sitio principal visible.
activity: elige exactamente una categoría de esta lista: development, terminal, web, media,
communication, document, design, gaming, idle, other.
summary: descripción humana en español de un máximo de 12 palabras.
No inventes detalles ni incluyas datos sensibles.'''


@dataclass(frozen=True)
class Config:
    model_name: str
    capture_interval_seconds: float
    target_size: tuple[int, int]
    max_new_tokens: int
    thumbnail_size: tuple[int, int]
    mean_difference_threshold: float
    pixel_difference_threshold: int
    changed_ratio_threshold: float
    confirmations_required: int
    warmup_seconds: float

    @classmethod
    def load(cls, path: Path) -> "Config":
        values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            **{
                **values,
                "target_size": tuple(values["target_size"]),
                "thumbnail_size": tuple(values["thumbnail_size"]),
            }
        )

    def save(self, path: Path) -> None:
        values = {
            **self.__dict__,
            "target_size": list(self.target_size),
            "thumbnail_size": list(self.thumbnail_size),
        }
        path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class VlmClient:
    def __init__(self, config: Config) -> None:
        print(f"Cargando {config.model_name}…")
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.target_size = config.target_size
        self.max_new_tokens = config.max_new_tokens

    def infer(self, frame) -> ActivityState:
        image = Image.fromarray(frame)
        image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}
        ]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        # Qwen devuelve entrada + continuación: decodificamos únicamente los tokens nuevos.
        answer = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        state = parse_activity(answer)
        return state


def parse_activity(answer: str) -> ActivityState:
    """Convierte la salida del modelo en un estado seguro y pequeño."""
    text = extract_json(answer)
    try:
        data = json.loads(text)
        app = str(data["app"]).strip()[:80]
        activity = normalize_activity(str(data["activity"]))
        summary = str(data.get("summary", "Respuesta sin resumen")).strip()[:240]
        if app and activity:
            return ActivityState(app, activity, summary or "Respuesta sin resumen")
    except (json.JSONDecodeError, KeyError, TypeError):
        partial = parse_truncated_activity(text)
        if partial is not None:
            return partial
    return ActivityState("Desconocida", "other", answer.replace("\n", " ")[:240] or "Sin respuesta")


def extract_json(answer: str) -> str:
    """Quita el bloque Markdown opcional que algunos VLM añaden al JSON."""
    text = answer.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else text


def parse_truncated_activity(text: str) -> ActivityState | None:
    """Recupera app/actividad si el límite cortó el cierre del JSON o el summary."""
    def field(name: str) -> str | None:
        match = re.search(rf'"{name}"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        if not match:
            return None
        try:
            return json.loads(f'"{match.group(1)}"').strip()
        except json.JSONDecodeError:
            return None

    app = field("app")
    activity = field("activity")
    if not app or not activity:
        return None
    summary = field("summary")
    if not summary:
        unfinished = re.search(r'"summary"\s*:\s*"(.*)$', text, flags=re.DOTALL)
        summary = unfinished.group(1).strip() if unfinished else "Respuesta truncada"
    return ActivityState(app[:80], normalize_activity(activity), (summary or "Respuesta truncada")[:240])


def record_state(history: EventHistory, state: ActivityState) -> tuple[ActivityEvent | None, str]:
    """Registra una decisión y devuelve el evento y su motivo para la interfaz."""
    previous = history.last_state
    event = history.add_if_new(state)
    if event:
        reason = "no había estado anterior" if previous is None else "app o actividad cambiaron"
        return event, reason
    reason = "misma identidad semántica"
    return None, reason


class WatcherWorker:
    """Bucle del watcher independiente de la consola o de la interfaz gráfica."""

    def __init__(self, config: Config, emit: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.config = config
        self.emit = emit or (lambda _message: None)
        self.stop_requested = Event()

    def stop(self) -> None:
        self.stop_requested.set()

    def run(self) -> None:
        camera = None
        client = None
        try:
            self.emit({"type": "status", "value": "Cargando modelo…"})
            camera = dxcam.create()
            client = VlmClient(self.config)
            detector = VisualChangeDetector(
                self.config.thumbnail_size,
                self.config.mean_difference_threshold,
                self.config.pixel_difference_threshold,
                self.config.changed_ratio_threshold,
            )
            history = EventHistory()
            confirmations = 0
            initial_state_logged = False
            self.emit({"type": "status", "value": "WATCHING"})
            time.sleep(self.config.warmup_seconds)
            while not self.stop_requested.is_set():
                started = time.monotonic()
                frame = camera.grab()
                if frame is None:
                    self.emit({"type": "status", "value": "Esperando captura…"})
                else:
                    measurement = detector.measure(frame)
                    self.emit({"type": "frame", "frame": frame.copy()})
                    self.emit({"type": "measurement", "measurement": measurement})
                    if not initial_state_logged:
                        initial_state_logged = True
                        state = client.infer(frame)
                        event, reason = record_state(history, state)
                        self.emit({"type": "state", "state": state, "reason": reason})
                        if event:
                            self.emit({"type": "event", "event": event})
                    else:
                        confirmations = confirmations + 1 if measurement.significant else 0
                        if confirmations >= self.config.confirmations_required:
                            confirmations = 0
                            state = client.infer(frame)
                            event, reason = record_state(history, state)
                            self.emit({"type": "state", "state": state, "reason": reason})
                            if event:
                                self.emit({"type": "event", "event": event})
                elapsed = time.monotonic() - started
                self.emit({"type": "latency", "seconds": elapsed})
                self.stop_requested.wait(max(0, self.config.capture_interval_seconds - elapsed))
        except Exception as error:
            self.emit({"type": "error", "message": str(error)})
        finally:
            if camera is not None:
                camera.stop()
            del client
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.emit({"type": "status", "value": "STOPPED"})


def run(config: Config) -> None:
    worker = WatcherWorker(config)
    signal.signal(signal.SIGINT, lambda *_unused: worker.stop())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_unused: worker.stop())
    print("Observando. Pulsa Ctrl+C para detener.")
    worker.run()
    print("Screen Watcher detenido. Recursos liberados.")


if __name__ == "__main__":
    run(Config.load(ROOT / "config.json"))
