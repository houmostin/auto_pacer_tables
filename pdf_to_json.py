# -*- coding: utf-8 -*-
"""
Интерактивный парсер мини-таблиц из PDF в единый JSON.

Что делает:
  - Открывает PDF, рендерит страницы.
  - По твоему клику мышкой выбирает:
      1) ROI заголовка (строка с "HU SB (vs fish)" и/или "R 2,5x - 2x")
      2) ROI сетки 13x13.
  - По каждой клетке сетки определяет базовый цвет -> действие
    (fold/limp|call/raise/push) и ищет мини-квадратики (push/call/fold/raise) с порогом N.
  - Пытается OCR-ить спот и рейз-сайз из заголовка.
  - Сохраняет в JSON (дописывает).

Запуск (из корня проекта):
    python -m src.pdf_to_json --pdf "vs_fish_final_19.03 (2).pdf" --out tables_from_pdf.json

Управление:
  - Стрелки ←/→ : переключение страниц PDF
  - Клавиша H   : выбрать ROI заголовка (клик-даун, клик-ап, потом S)
  - Клавиша G   : выбрать ROI сетки (клик-даун, клик-ап, потом S)
  - Клавиша A   : добавить текущую мини-таблицу в JSON
  - Клавиша N   : начать разметку следующей мини-таблицы (на этой же странице или после стрелки)
  - Клавиша Q/Esc: выход без сохранения текущего ROI
  - Клавиша ?   : подсказка горячих клавиш

Зависимости:
    pip install pymupdf opencv-python pillow numpy pytesseract
"""

from __future__ import annotations
import os, re, json, argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import fitz  # PyMuPDF
import cv2
import numpy as np

# --- гибкий импорт наших OCR-утилит ---
try:
    from .ocr_live import (
        preprocess_for_text, preprocess_for_digits,
        ocr_text, ocr_digits, extract_number,
        sample_cell_color, bgr_to_hex
    )
except Exception:
    from ocr_live import (
        preprocess_for_text, preprocess_for_digits,
        ocr_text, ocr_digits, extract_number,
        sample_cell_color, bgr_to_hex
    )

# ===== Палитра (твои HEX) =====
HEX_ORANGE = "#ffdc6d"  # limp/call
HEX_BLUE   = "#71daff"  # raise
HEX_GREEN  = "#05ff76"  # push
HEX_GRAY   = "#f2f2f2"  # fold

COLOR_TOL = 18.0  # чувствительность классификации цвета в LAB

COLOR_TO_ACTION = {
    HEX_ORANGE: "call_or_limp",
    HEX_BLUE:   "raise",
    HEX_GREEN:  "push",
    HEX_GRAY:   "fold",
}

def hex_to_bgr(hx: str) -> Tuple[int,int,int]:
    hx = hx.strip().lstrip("#")
    return (int(hx[4:6],16), int(hx[2:4],16), int(hx[0:2],16))

def bgr_to_lab(bgr: Tuple[int,int,int]) -> np.ndarray:
    arr = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    return lab[0,0,:].astype(np.float32)

LAB_TARGETS = [(hx, bgr_to_lab(hex_to_bgr(hx))) for hx in [HEX_ORANGE, HEX_BLUE, HEX_GREEN, HEX_GRAY]]

def classify_action_by_color(bgr: Tuple[int,int,int]) -> Optional[str]:
    lab = bgr_to_lab(bgr)
    best, dmin = None, 1e9
    for hx, lab_ref in LAB_TARGETS:
        d = float(np.linalg.norm(lab - lab_ref))
        if d < dmin:
            dmin, best = d, hx
    if dmin <= COLOR_TOL:
        return COLOR_TO_ACTION.get(best)
    return None  # например красный/шум — игнор

# ===== Геометрия сетки =====
RANKS = list("AKQJT98765432")

@dataclass
class ROI:
    x1:int; y1:int; x2:int; y2:int

@dataclass
class GridSpec:
    rows:int = 13
    cols:int = 13
    inner_margin:int = 4  # не брать линии сетки

def crop(img: np.ndarray, roi: ROI) -> np.ndarray:
    h,w = img.shape[:2]
    x1 = max(0, min(w, roi.x1)); x2 = max(0, min(w, roi.x2))
    y1 = max(0, min(h, roi.y1)); y2 = max(0, min(h, roi.y2))
    if x2<=x1 or y2<=y1: raise ValueError(f"Bad ROI: {roi}")
    return img[y1:y2, x1:x2].copy()

def iter_cells(grid_roi: ROI, spec: GridSpec):
    total_w = grid_roi.x2 - grid_roi.x1
    total_h = grid_roi.y2 - grid_roi.y1
    cw = total_w / spec.cols
    ch = total_h / spec.rows
    for r in range(spec.rows):
        for c in range(spec.cols):
            x1 = int(grid_roi.x1 + c*cw) + spec.inner_margin
            y1 = int(grid_roi.y1 + r*ch) + spec.inner_margin
            x2 = int(grid_roi.x1 + (c+1)*cw) - spec.inner_margin
            y2 = int(grid_roi.y1 + (r+1)*ch) - spec.inner_margin
            yield (r,c), ROI(x1,y1,x2,y2)

def cell_to_hand_key(r:int, c:int) -> str:
    r1, r2 = RANKS[r], RANKS[c]
    if r==c:  return f"{r1}{r2}"     # пары
    if c<r:   return f"{r1}{r2}o"    # слева -> offsuit
    return f"{r1}{r2}s"              # справа -> suited

# ===== Рендер PDF =====
def render_pdf_pages(pdf_path: str, zoom: float = 2.5) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    imgs = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
    doc.close()
    return imgs

# ===== OCR шапки (spot и raise) =====
def parse_header(header_bgr: np.ndarray) -> Tuple[str, Dict[str,float]]:
    prep = preprocess_for_text(header_bgr)
    t = ocr_text(prep).upper().replace(",", ".")
    spot_key = "UNKNOWN"

    # Базовые шаблоны (дополним по ходу)
    if "HU" in t and "SB" in t and "VS FISH" in t:
        spot_key = "HU_SB"
    if "HEADS-UP BB VS OPUSH" in t:
        spot_key = "HU_BB_vs_OPUSH"
    if "HU BB VS MR" in t:
        spot_key = "HU_BB_vs_MR"
    if "HU BB VS LIMP" in t:
        spot_key = "HU_BB_vs_SB_LIMP"
    if "3MAX" in t and "BB VS SB LIMP" in t:
        spot_key = "3max_BB_vs_SB_LIMP"
    if "3MAX SB VS BTN MR" in t:
        spot_key = "3max_SB_vs_BTN_MR"
    if "3MAX BB VS BTN MR" in t:
        spot_key = "3max_BB_vs_BTN_MR"
    if "3MAX BB VS BTN LIMP" in t:
        spot_key = "3max_BB_vs_BTN_LIMP"
    if "3MAX BB VS BTN PUSH" in t:
        spot_key = "3max_BB_vs_BTN_PUSH"

    # raise-size: R 2.5x - 2x (или одна цифра)
    min_x = max_x = default_x = None
    rx = re.findall(r"R\s*([0-9]+(?:\.[0-9]+)?)\s*[XХ]", t)
    if rx:
        vals = [float(x) for x in rx]
        if len(vals)==1:
            min_x = max_x = default_x = vals[0]
        else:
            min_x, max_x = min(vals), max(vals)
            default_x = min_x
    raise_size = {
        "min_x": float(min_x) if min_x is not None else 2.0,
        "max_x": float(max_x) if max_x is not None else (float(min_x) if min_x is not None else 2.0),
        "default_x": float(default_x) if default_x is not None else 2.0
    }
    return spot_key, raise_size

# ===== Поиск мини-квадратиков в клетке =====
MINISQ_MIN_AREA_FRAC = 0.006
MINISQ_MAX_AREA_FRAC = 0.15
MINISQ_DIGITS_PSM = 7

def find_minisquares(cell_bgr: np.ndarray) -> List[Tuple[str, float]]:
    h,w = cell_bgr.shape[:2]
    area = float(h*w)
    out: List[Tuple[str,float]] = []

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[...,2], 230, 255, cv2.THRESH_BINARY_INV)
    k = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x,y,w0,h0 = cv2.boundingRect(cnt)
        a = (w0*h0)/area
        if a < MINISQ_MIN_AREA_FRAC or a > MINISQ_MAX_AREA_FRAC:
            continue
        minisq = cell_bgr[y:y+h0, x:x+w0]

        # цвет мини-квадрата
        color_hex = bgr_to_hex(sample_cell_color(minisq))
        act = classify_action_by_color(hex_to_bgr(color_hex))
        if act is None:
            center = minisq[h0//4:3*h0//4, w0//4:3*w0//4]
            color_hex = bgr_to_hex(sample_cell_color(center))
            act = classify_action_by_color(hex_to_bgr(color_hex))
        if act is None:
            continue

        # OCR цифры N
        prep = preprocess_for_digits(minisq)
        raw = ocr_digits(prep, psm=MINISQ_DIGITS_PSM)
        n = extract_number(raw)
        if n is None:
            continue
        out.append((act, float(n)))

    out.sort(key=lambda t: t[1])
    return out

# ===== Интерактив: выбор ROI и навигация по страницам =====
@dataclass
class PickerState:
    header: Optional[ROI] = None
    grid: Optional[ROI] = None

def draw_help(img):
    overlay = img.copy()
    lines = [
        "Controls: ←/→ page | H=pick Header | G=pick Grid | A=append JSON | N=next table | ?=help | Q/Esc=quit",
        "After H/G: drag with mouse, release, then press 'S' to confirm (R=reset)."
    ]
    y = 30
    for t in lines:
        cv2.putText(overlay, t, (30,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, t, (30,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
        y += 30
    return overlay

def pick_roi(img: np.ndarray, title: str) -> ROI:
    pts = []
    view = img.copy()
    def on_mouse(event, x, y, flags, param):
        nonlocal pts, view
        if event == cv2.EVENT_LBUTTONDOWN:
            pts = [(x,y)]
        elif event == cv2.EVENT_LBUTTONUP:
            pts.append((x,y))
            view = img.copy()
            cv2.rectangle(view, pts[0], pts[1], (0,255,0), 2)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s') and len(pts)==2:
            break
        if key == ord('r'):
            view = img.copy(); pts = []
        if key == 27 or key == ord('q'):
            cv2.destroyWindow(title); raise SystemExit("Cancelled.")
    cv2.destroyWindow(title)
    (x1,y1),(x2,y2) = pts
    x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
    return ROI(x1,y1,x2,y2)

def parse_table(img: np.ndarray, header_roi: ROI, grid_roi: ROI) -> Dict:
    header_bgr = crop(img, header_roi)
    spot_key, raise_size = parse_header(header_bgr)

    grid_spec = GridSpec()
    grid: Dict[str, Dict] = {}
    for (r,c), cell_roi in iter_cells(grid_roi, grid_spec):
        cell = crop(img, cell_roi)
        base_hex = bgr_to_hex(sample_cell_color(cell))
        base_act = classify_action_by_color(hex_to_bgr(base_hex)) or "none"
        overrides = []
        for act, n in find_minisquares(cell):
            overrides.append({"act": act, "n": float(n)})
        hand = cell_to_hand_key(r,c)
        grid[hand] = {"base": base_act, "overrides": overrides}

    return {
        "spot_key": spot_key,
        "vs_profile": "fish",
        "raise_size": raise_size,
        "grid": grid
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Путь к PDF с таблицами")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "tables_from_pdf.json"),
                    help="Куда сохранять JSON (будет дополняться)")
    ap.add_argument("--zoom", type=float, default=2.5, help="Рендер масштаб (2.0-3.0 ок)")
    args = ap.parse_args()

    pages = render_pdf_pages(args.pdf, zoom=args.zoom)
    page_idx = 0
    pick = PickerState()

    while True:
        base = pages[page_idx].copy()
        view = draw_help(base)

        # Нарисуем выбранные ROI
        if pick.header is not None:
            cv2.rectangle(view, (pick.header.x1, pick.header.y1), (pick.header.x2, pick.header.y2), (0,255,0), 2)
        if pick.grid is not None:
            cv2.rectangle(view, (pick.grid.x1, pick.grid.y1), (pick.grid.x2, pick.grid.y2), (255,0,0), 2)

        cv2.putText(view, f"Page {page_idx+1}/{len(pages)}", (30, view.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(view, f"Page {page_idx+1}/{len(pages)}", (30, view.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("PDF Tables Parser", view)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q')):  # Esc/Q
            break
        elif key == ord('?'):
            pass  # help уже нарисован
        elif key == 81:  # ←
            page_idx = (page_idx - 1) % len(pages)
            pick = PickerState()
        elif key == 83:  # →
            page_idx = (page_idx + 1) % len(pages)
            pick = PickerState()
        elif key == ord('h'):
            pick.header = pick_roi(pages[page_idx], "Выбери HEADER ROI и нажми 's'")
        elif key == ord('g'):
            pick.grid = pick_roi(pages[page_idx], "Выбери GRID ROI и нажми 's'")
        elif key == ord('n'):
            pick = PickerState()
        elif key == ord('a'):
            if pick.header is None or pick.grid is None:
                print("Сначала укажи H (header) и G (grid).")
                continue
            entry = parse_table(pages[page_idx], pick.header, pick.grid)
            # записываем/дополняем JSON
            if os.path.exists(args.out):
                with open(args.out, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "tables" not in data or not isinstance(data["tables"], list):
                    data = {"tables": []}
            else:
                data = {"tables": []}
            data["tables"].append(entry)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Добавлена таблица: {entry.get('spot_key')}  →  {args.out}")
            pick = PickerState()  # можно сразу размечать следующую
        # иначе — ждём следующие клавиши

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
