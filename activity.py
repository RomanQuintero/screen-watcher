"""Representación y almacenamiento en memoria del historial semántico."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


ACTIVITY_CATEGORIES = frozenset({
    "development", "terminal", "web", "media", "communication", "document",
    "design", "gaming", "idle", "other",
})


def normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def normalize_app(value: str) -> str:
    """Normalización deliberadamente simple: espacios, mayúsculas y alias comunes."""
    normalized = normalize_text(value)
    aliases = {
        "vs code": "visual studio code",
        "vscode": "visual studio code",
        "powershell": "windows powershell",
    }
    return aliases.get(normalized, normalized)


def normalize_activity(value: str) -> str:
    category = normalize_text(value)
    return category if category in ACTIVITY_CATEGORIES else "other"


@dataclass(frozen=True)
class ActivityState:
    app: str
    activity: str
    summary: str

    @property
    def identity(self) -> tuple[str, str]:
        return (normalize_app(self.app), normalize_activity(self.activity))


@dataclass(frozen=True)
class ActivityEvent:
    timestamp: datetime
    state: ActivityState

    def display(self) -> str:
        return f"{self.timestamp:%H:%M}  {self.state.app:<12} {self.state.summary}"


class EventHistory:
    def __init__(self) -> None:
        self.events: list[ActivityEvent] = []
        self._last_state: ActivityState | None = None

    @property
    def last_state(self) -> ActivityState | None:
        """Último estado aceptado, para diagnosticar la decisión de evento."""
        return self._last_state

    def add_if_new(self, state: ActivityState) -> ActivityEvent | None:
        if self._last_state is not None and state.identity == self._last_state.identity:
            return None
        event = ActivityEvent(datetime.now(), state)
        self.events.append(event)
        self._last_state = state
        return event
