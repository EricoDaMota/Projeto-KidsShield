from __future__ import annotations

from io import BytesIO
from typing import Iterable, Tuple

import cv2
import numpy as np
from PIL import Image

FaceTuple = Tuple[int, int, int, int]


def clamp_box(box: FaceTuple, image_shape: tuple[int, ...]) -> FaceTuple:
    x, y, w, h = [int(v) for v in box]
    height, width = image_shape[:2]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def expand_box(box: FaceTuple, image_shape: tuple[int, ...], margin_percent: int = 8) -> FaceTuple:
    x, y, w, h = clamp_box(box, image_shape)
    height, width = image_shape[:2]
    mx = int(w * margin_percent / 100)
    my = int(h * margin_percent / 100)
    x2 = max(0, x - mx)
    y2 = max(0, y - my)
    right = min(width, x + w + mx)
    bottom = min(height, y + h + my)
    return x2, y2, max(1, right - x2), max(1, bottom - y2)


def _odd_kernel_size(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def apply_blur(
    image_rgb: np.ndarray,
    boxes: Iterable[FaceTuple],
    strength: int = 35,
    margin_percent: int = 8,
) -> np.ndarray:
    """Aplica Gaussian Blur somente às regiões informadas."""

    output = image_rgb.copy()
    for box in boxes:
        x, y, w, h = expand_box(box, output.shape, margin_percent)
        roi = output[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        # O kernel é proporcional ao tamanho do rosto e limitado para estabilidade.
        base = max(9, min(w, h) * max(10, strength) // 100)
        kernel = _odd_kernel_size(min(base, 151))
        blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        # Duas passagens aumentam a proteção sem depender de serviço externo.
        blurred = cv2.GaussianBlur(blurred, (kernel, kernel), 0)
        output[y : y + h, x : x + w] = blurred

    return output


def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
