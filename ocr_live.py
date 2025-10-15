# -*- coding: utf-8 -*-
"""
src/ocr_live.py — небольшой модуль для OCR/предобработки кадра.
Его задача: безопасно вырезать ROI, подготовить изображение и вернуть распознанный текст.

Использование:
    from src.ocr_live import read_roi, preprocess_for_text, ocr_text, ocr_digits

    roi_img = read_roi(frame, (x1, y1, x2, y2))
    prep = preprocess_for_text(roi_img)
    text = ocr_text(prep)

Зависимости:
    pip install opencv-python pillow pytesseract numpy
    + установленный Tesseract OCR (в Windows обычно: C:\Program Files\Tesseract-OCR\tesseract.exe)
Настройка пути Tesseract (если нужно):
    setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
или пропишите путь в TESSERACT_CMD внизу.
"""

from __future__ import annotations
import os
import re
from typing import Tuple, Optional

import cv2
import numpy as np
import pytesseract


# ==== Настройки Tesseract ====
# Можно задать путь через переменную окружения TESSERACT_CMD
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
# Языки по умолчанию: рус+англ (цифры/латиница/кириллица)
_DEFAULT_LANG = "rus+eng"


# ==== Утилиты для ROI ====
def read_roi(frame: np.ndarray, roi_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Безопасно вырезает ROI из кадра.
    roi_xyxy = (x1, y1, x2, y2), где x2/y2 НЕ включительно (как в numpy-slice логике).

    Поднимает ValueError, если frame=None или выход за границы.
    """
    if frame is None:
        raise ValueError("read_roi: frame is None (кадр не получен)")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_xyxy

    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(
            f"read_roi: ROI {roi_xyxy} вне границ кадра {w}x{h}. "
            "Проверь координаты ROI."
        )

    return frame[y1:y2, x1:x2].copy()


# ==== Предобработка ====
def preprocess_for_text(img_bgr: np.ndarray) -> np.ndarray:
    """
    Предобработка под обычный текст: серый -> шумоподавление -> контраст -> бинаризация.
    Подходит для стэков/подписей/рейз-сайзов.
    Возвращает одноканальное изображение (uint8), готовое для OCR.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Лёгкая фильтрация шума без размытия острых краёв цифр
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Локальный контраст (CLAHE) улучшает читаемость бледного текста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Адаптивная бинаризация — более устойчива к фоновой неравномерности
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 10
    )

    # Небольшое "закрытие", чтобы склеить разорванные штрихи
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return bin_img


def preprocess_for_digits(img_bgr: np.ndarray) -> np.ndarray:
    """
    Предобработка, заточенная под цифры/короткие числа (BB, рейз-сайзы, пороги N).
    Чуть жёстче бинаризация.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Усиление контраста
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Жёсткий порог Отсу
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Тонкое открытие/закрытие
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


# ==== OCR-вызовы ====
def _base_tesseract(img: np.ndarray, lang: str, psm: int, whitelist: Optional[str]) -> str:
    """
    Общий вызов Tesseract. psm=6 — построчный, 7 — одна строка, 8 — одно слово.
    whitelist — ограничение алфавита, ускоряет и повышает точность.
    """
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        # tessedit_char_whitelist корректнее задавать через -c
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    txt = pytesseract.image_to_string(img, lang=lang, config=config)
    return txt.strip()


def ocr_text(img_bin_or_gray: np.ndarray, lang: str = _DEFAULT_LANG, psm: int = 6) -> str:
    """
    OCR для общего текста (позиции, подписи таблиц, кнопки, статусы).
    psm=6 (по умолчанию): "assume a single uniform block of text".
    """
    return _base_tesseract(img_bin_or_gray, lang=lang, psm=psm, whitelist=None)


def ocr_digits(img_bin_or_gray: np.ndarray, psm: int = 7) -> str:
    """
    OCR только цифр и разделителей. Полезно для BB/сайзов/порогов в мини-квадратах.
    psm=7: "treat the image as a single text line".
    """
    whitelist = "0123456789.,:+-xX/%"
    return _base_tesseract(img_bin_or_gray, lang="eng", psm=psm, whitelist=whitelist)


# ==== Парсеры текста в числа/BB ====
_FLOAT_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def extract_number(text: str) -> Optional[float]:
    """
    Достаёт первое число из строки и приводит к float (запятая -> точка).
    Возвращает None, если числа нет.
    """
    if not text:
        return None
    m = _FLOAT_RE.search(text)
    if not m:
        return None
    s = m.group(0).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def round_to_half(x: float) -> float:
    """Округление до ближайших 0.5 (… 10.0, 10.5, 11.0, …)."""
    return round(x * 2) / 2.0


# ==== Вспомогательное (цвет/пиксель) ====
def sample_cell_color(img_bgr: np.ndarray, margin: int = 2) -> Tuple[int, int, int]:
    """
    Примерно оценивает «базовый» цвет ячейки: усредняет центральную область.
    Возвращает BGR-кортеж (int, int, int).
    """
    h, w = img_bgr.shape[:2]
    x1, y1 = margin, margin
    x2, y2 = max(margin + 1, w - margin), max(margin + 1, h - margin)
    patch = img_bgr[y1:y2, x1:x2]
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    b, g, r = [int(round(v)) for v in mean_bgr.tolist()]
    return (b, g, r)


def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    """Конвертация BGR (OpenCV) -> HEX-строка вида #RRGGBB."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# ==== Пример безопасного конвейера (функции-обёртки) ====
def ocr_roi_text(frame: np.ndarray, roi_xyxy: Tuple[int, int, int, int]) -> str:
    """
    Быстрый путь: вырезать ROI → подготовить → OCR.
    Для обычного текста.
    """
    roi = read_roi(frame, roi_xyxy)
    prep = preprocess_for_text(roi)
    return ocr_text(prep)


def ocr_roi_digits(frame: np.ndarray, roi_xyxy: Tuple[int, int, int, int]) -> Optional[float]:
    """
    Быстрый путь: вырезать ROI → подготовить под цифры → OCR → float.
    """
    roi = read_roi(frame, roi_xyxy)
    prep = preprocess_for_digits(roi)
    raw = ocr_digits(prep)
    return extract_number(raw)
