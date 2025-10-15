# src/state_from_screen.py — версия без dxcam, с mss
import time
import numpy as np
from mss import mss

from .profiles import PROFILES, ACTIVE
from .parsers import parse_stack_bb, normalize_hand
from .ocr_live import (
    preprocess_for_text,
    preprocess_for_digits,
    ocr_text,
    ocr_digits,
)

P = PROFILES[ACTIVE]
_sct = sct = mss()  # один экземпляр на модуль

# ---------- утилиты ----------

def _clamp_box(box):
    """box = (x1,y1,x2,y2) -> корректируем и считаем width/height"""
    x1, y1, x2, y2 = box
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    w = max(1, int(x2 - x1))
    h = max(1, int(y2 - y1))
    left, top = max(0, int(x1)), max(0, int(y1))
    return {"left": left, "top": top, "width": w, "height": h}

def grab_region(_, box):
    """Захват через mss; параметр cam игнорируем для совместимости с app.py"""
    region = _clamp_box(box)
    shot = _sct.grab(region)  # BGRA
    frame = np.array(shot)[:, :, :3].copy()  # BGR
    return frame

def robust_digits(img) -> float:
    """OCR чисел с защитой от падений: пусто -> 0.0"""
    try:
        if img is None or (hasattr(img, "size") and img.size == 0):
            return 0.0
        txt = ocr_digits(preprocess_for_digits(img))
        val = parse_stack_bb(txt)
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0

def robust_hand(img):
    """OCR руки с защитой: пусто -> ('??', None)"""
    try:
        if img is None or (hasattr(img, "size") and img.size == 0):
            return ("??", None)
        txt = ocr_text(preprocess_for_text(img))
        out = normalize_hand(txt)
        return out if out else ("??", None)
    except Exception:
        return ("??", None)

# ---------- основной конструктор состояния ----------

def build_state(cam_ignored):
    # Снимаем ROI из профиля
    img_h   = grab_region(None, P["ROI_STACK_HERO"])
    img_l   = grab_region(None, P["ROI_STACK_LEFT"])
    img_r   = grab_region(None, P["ROI_STACK_RIGHT"])
    img_pot = grab_region(None, P["ROI_POT"])
    img_hand= grab_region(None, P["ROI_HAND"])

    # OCR c защитой
    s_h  = robust_digits(img_h)
    s_l  = robust_digits(img_l)
    s_r  = robust_digits(img_r)
    pot  = robust_digits(img_pot)
    hand = robust_hand(img_hand)

    return {
        "mode": "3max",        # TODO: авто-детект позже
        "stage": "preflop",
        "hero_pos": "BB",      # TODO: авто-детект позже
        "stacks_bb": {
            "hero": s_h,
            "vill_left": s_l,
            "vill_right": s_r
        },
        "pot_bb": pot,
        "hero_hand": hand
    }
