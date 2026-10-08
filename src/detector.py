from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceBox:
    """Representa uma área retangular detectada na imagem."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FaceDetector:
    """Detector de rostos baseado em Haar Cascade do OpenCV.

    É uma solução local, leve e sem envio da fotografia para serviços externos.
    """

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(cascade_path)
        if self._classifier.empty():
            raise RuntimeError("Não foi possível carregar o detector de rostos do OpenCV.")

    def detect(self, image_rgb: np.ndarray) -> List[FaceBox]:
        if image_rgb is None or image_rgb.size == 0:
            return []

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        height, width = gray.shape[:2]
        min_side = max(24, min(width, height) // 20)

        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        result = [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        result.sort(key=lambda f: (f.y, f.x))
        return result


def draw_face_boxes(
    image_rgb: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    selected: List[bool] | None = None,
) -> np.ndarray:
    """Desenha caixas e números nos rostos sem alterar a imagem original."""

    output = image_rgb.copy()
    for index, (x, y, w, h) in enumerate(faces):
        is_selected = True if selected is None or index >= len(selected) else selected[index]
        color = (13, 127, 118) if is_selected else (145, 155, 163)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
        status = "DESFOCAR" if is_selected else "VISIVEL"
        label = f"Rosto {index + 1} - {status}"
        text_y = max(24, y - 8)
        cv2.putText(
            output,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
