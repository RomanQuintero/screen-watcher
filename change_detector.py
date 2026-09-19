"""Detección de cambio visual barata para decidir cuándo consultar el VLM."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ChangeMeasurement:
    mean_difference: float
    changed_ratio: float
    significant: bool


class VisualChangeDetector:
    """Compara miniaturas en gris; no usa modelos ni conserva capturas completas."""

    def __init__(
        self,
        thumbnail_size: tuple[int, int],
        mean_threshold: float,
        pixel_threshold: int,
        changed_ratio_threshold: float,
    ) -> None:
        self.thumbnail_size = thumbnail_size
        self.mean_threshold = mean_threshold
        self.pixel_threshold = pixel_threshold
        self.changed_ratio_threshold = changed_ratio_threshold
        self._previous: np.ndarray | None = None

    def measure(self, frame: np.ndarray) -> ChangeMeasurement:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        current = cv2.resize(gray, self.thumbnail_size, interpolation=cv2.INTER_AREA)
        if self._previous is None:
            self._previous = current
            return ChangeMeasurement(0.0, 0.0, False)

        difference = cv2.absdiff(current, self._previous)
        mean_difference = float(difference.mean())
        changed_ratio = float((difference >= self.pixel_threshold).mean())
        self._previous = current
        significant = (
            mean_difference >= self.mean_threshold
            and changed_ratio >= self.changed_ratio_threshold
        )
        return ChangeMeasurement(mean_difference, changed_ratio, significant)
