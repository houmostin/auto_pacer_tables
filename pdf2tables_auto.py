# -*- coding: utf-8 -*-
"""
Автоматический парсер: PDF -> tables_from_pdf.json

Запуск из корня:
  python -m src.pdf2tables_auto --pdf "vs_fish_final_19.03 (2).pdf" --out tables_from_pdf.json

Опции:
  --pages "1,2,4"   только эти страницы (1-based); по умолчанию все
  --dpi 300         DPI рендера страниц (по умолчанию 300)

Ограничения: ожидает типовую верстку (шапка ярко-зелёная, сетка 13x13 с чёрными линиями).
"""

from __future__ import annotations
import re, os, json, argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Если tesseract не в PATH, раскомментируй и укажи путь:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Эталонные цвета (из твоих правил)
HEX_ORANGE = "#ffdc6d"  # limp/call
HEX_BLUE   = "#71daff"  # raise
HEX_GREEN  = "#05ff76"  # push (также используется для мини-квадрата push)
HEX_GRAY   = "#f2f2f2"  # fold

COLOR_TO_ACTION = {
    HEX_ORANGE: "call_or_limp",
    HEX_BLUE:   "raise",
    HEX_GREEN:  "push",
    HEX_GRAY:   "fold",
}

LAB_TOL = 65.0  # допуск по цвету (Lab), подобран «пожирнее», чтобы работать на PDF/PNG

RANKS = list("AKQJT98765432")


def hex_to_bgr(hx: str) -> Tuple[int,int,int]:
    hx = hx.strip().lstrip("#")
    return (int(hx[4:6],16), int(hx[2:4],16), int(hx[0:2],16))

def bgr_to_lab(bgr: Tuple[int,int,int]) -> np.ndarray:
    arr = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    return lab[0,0,:].astype(np.float32)

LAB_TARGETS = {
    HEX_ORANGE: bgr_to_lab(hex_to_bgr(HEX_ORANGE)),
    HEX_BLUE:   bgr_to_lab(hex_to_bgr(HEX_BLUE)),
    HEX_GREEN:  bgr_to_lab(hex_to_bgr(HEX_GREEN)),
    HEX_GRAY:   bgr_to_lab(hex_to_bgr(HEX_GRAY)),
}

def classify_action_by_bgr(bgr: Tuple[int,int,int]) -> Optional[str]:
    lab = bgr_to_lab(bgr)
    best, dmin = None, 1e9
    for hx, ref in LAB_TARGETS.items():
        d = float(np.linalg.norm(lab - ref))
        if d < dmin: dmin, best = d, hx
    if dmin <= LAB_TOL:
        return COLOR_TO_ACTION[best]
    return None


def render_page(pdf: fitz.Document, pno: int, dpi: int=300) -> np.ndarray:
    page = pdf.load_page(pno)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    # PyMuPDF: RGB -> OpenCV BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def find_grid_rois(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Ищем большие прямоугольники-сетки 13x13 по линиям.
    Возвращает список ROI (x1,y1,x2,y2).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # усилим линии
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)

    # выделим вертикали/горизонтали морфологией
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_h, iterations=1)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_v, iterations=1)
    grid = cv2.bitwise_or(hor, ver)

    # контуры кандидатов
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    H, W = gray.shape
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < W*0.25 or h < H*0.25:
            continue  # мелочь отсекаем
        # Немного расширим, чтобы захватить целиком
        pad = 6
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(W, x+w+pad); y2 = min(H, y+h+pad)
        rois.append((x1,y1,x2,y2))

    # сортируем сверху-вниз, слева-направо
    rois.sort(key=lambda r: (r[1], r[0]))
    return rois


def iter_cells(roi: Tuple[int,int,int,int], rows=13, cols=13, inner=6):
    x1,y1,x2,y2 = roi
    W = x2-x1; H = y2-y1
    cw = W/cols; ch = H/rows
    for r in range(rows):
        for c in range(cols):
            cx1 = int(x1 + c*cw) + inner
            cy1 = int(y1 + r*ch) + inner
            cx2 = int(x1 + (c+1)*cw) - inner
            cy2 = int(y1 + (r+1)*ch) - inner
            yield (r,c), (cx1,cy1,cx2,cy2)


def hand_key(r:int, c:int) -> str:
    a,b = RANKS[r], RANKS[c]
    if r==c:  return f"{a}{b}"
    if c<r:   return f"{a}{b}o"
    return f"{a}{b}s"


def mean_bgr(img: np.ndarray) -> Tuple[int,int,int]:
    m = img.reshape(-1,3).mean(axis=0)
    return int(m[0]), int(m[1]), int(m[2])


def ocr_digits(img_gray: np.ndarray) -> Optional[float]:
    """OCR одной-двух цифр (N)."""
    # усиление
    g = cv2.resize(img_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cfg = r"--psm 7 -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(Image.fromarray(th), config=cfg)
    txt = re.sub(r"[^\d]", "", txt or "")
    if not txt:
        return None
    try:
        return float(int(txt))
    except Exception:
        return None


def detect_overrides(cell: np.ndarray) -> List[Dict[str,Any]]:
    """
    Ищем мини-квадраты: зелёный (push N), оранжевый (call/limp N).
    Смотрим два угла сверху (левый/правый) и центр вверху.
    """
    h,w = cell.shape[:2]
    patches = [
        ("left",  cell[0:int(h*0.35), 0:int(w*0.35)]),
        ("right", cell[0:int(h*0.35), int(w*0.65):w]),
        ("top",   cell[0:int(h*0.28), int(w*0.35):int(w*0.65)]),
    ]
    results = []
    for name, p in patches:
        bgr = mean_bgr(p)
        act = classify_action_by_bgr(bgr)
        if act in ("push","call_or_limp"):
            gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
            n = ocr_digits(gray)
            if n is not None:
                results.append({"act": act, "n": float(n)})
    # Упорядочим по N (как в твоём правиле)
    results.sort(key=lambda t: t["n"])
    # де-дупликация по act (берём минимум N для каждого типа)
    dedup: Dict[str, Dict[str,Any]] = {}
    for r in results:
        if r["act"] not in dedup or r["n"] < dedup[r["act"]]["n"]:
            dedup[r["act"]] = r
    return list(dedup.values())


def parse_header_spot_and_raise(header_img: np.ndarray) -> Tuple[str, Dict[str,float]]:
    """
    OCR заголовка в зелёной шапке: из текста делаем spot_key, вытаскиваем "R 2,5x - 2x" и т.п.
    """
    rgb = cv2.cvtColor(header_img, cv2.COLOR_BGR2RGB)
    txt = pytesseract.image_to_string(Image.fromarray(rgb), config="--psm 6")
    txt = (txt or "").replace("\n"," ").strip()

    # spot_key
    norm = txt.upper()
    # Примеры: "HU BB VS LIMP (VS FISH)  R 2,5X - 2X"
    # делаем HU_BB_vs_LIMP
    spot = norm
    spot = re.sub(r"\(VS\s*FISH\)", "", spot)
    spot = spot.replace("HEADS-UP", "HU")
    spot = spot.replace("HEADS UP", "HU")
    spot = re.sub(r"\s+VS\s+", "_vs_", spot)
    spot = re.sub(r"\s+", "_", spot)
    # оставим только интересную часть после возможного префикса таблицы
    # найдём HU/3MAX как маркер начала
    m = re.search(r"(HU|3MAX)[_\w]*", spot)
    if m:
        spot_key = m.group(0)
    else:
        spot_key = "UNKNOWN"

    # raise size
    # Ищем шаблоны вида "R 2,5X - 2X" или "R 2X" или просто "2,3X - 2X"
    rs = {"min_x": 2.0, "max_x": 2.0, "default_x": 2.0}
    t = norm.replace(",", ".")
    mm = re.search(r"R?\s*([0-9]+(?:\.[0-9]+)?)\s*X(?:\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*X)?", t)
    if mm:
        a = float(mm.group(1))
        b = float(mm.group(2)) if mm.group(2) else None
        if b is None:
            rs = {"min_x": a, "max_x": a, "default_x": a}
        else:
            x1, x2 = min(a,b), max(a,b)
            rs = {"min_x": x1, "max_x": x2, "default_x": x1}

    return spot_key, rs


def split_header_and_grid(full_roi: Tuple[int,int,int,int]) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """
    Разделяем общий ROI на верхнюю шапку и саму сетку (примерно).
    У таблиц шапка довольно высокая — возьмём 12–15% сверху.
    """
    x1,y1,x2,y2 = full_roi
    H = y2 - y1
    h_hdr = int(H * 0.14)
    header = (x1, y1, x2, y1 + h_hdr)
    grid   = (x1, y1 + h_hdr, x2, y2)
    return header, grid


def extract_table(img: np.ndarray, roi: Tuple[int,int,int,int]) -> Dict[str,Any]:
    header_roi, grid_roi = split_header_and_grid(roi)
    header_img = img[header_roi[1]:header_roi[3], header_roi[0]:header_roi[2]]
    spot_key, raise_size = parse_header_spot_and_raise(header_img)

    grid_dict: Dict[str, Dict[str,Any]] = {}

    for (r,c), (x1,y1,x2,y2) in iter_cells(grid_roi, 13, 13, inner=6):
        cell = img[y1:y2, x1:x2]
        # базовый цвет ячейки = средний цвет центральной части
        h, w = cell.shape[:2]
        cx1, cx2 = int(w*0.25), int(w*0.75)
        cy1, cy2 = int(h*0.25), int(h*0.75)
        patch = cell[cy1:cy2, cx1:cx2]
        bgr = mean_bgr(patch)
        base = classify_action_by_bgr(bgr) or "none"

        overrides = detect_overrides(cell)

        grid_dict[hand_key(r,c)] = {"base": base, "overrides": overrides}

    return {
        "spot_key": spot_key,
        "vs_profile": "fish",
        "raise_size": raise_size,
        "grid": grid_dict,
    }


def run(pdf_path: str, out_json: str, pages: Optional[List[int]], dpi: int):
    doc = fitz.open(pdf_path)
    page_ids = pages or list(range(1, doc.page_count+1))

    results = {"tables": []}

    for pno1 in page_ids:
        pno0 = pno1 - 1
        img = render_page(doc, pno0, dpi=dpi)
        rois = find_grid_rois(img)
        if not rois:
            print(f"[WARN] Страница {pno1}: сетки не найдены")
            continue
        print(f"[INFO] Страница {pno1}: найдено таблиц: {len(rois)}")
        for i, roi in enumerate(rois, 1):
            try:
                table = extract_table(img, roi)
                results["tables"].append(table)
                print(f"  [+] Таблица {i}: {table['spot_key']}  RAISE={table['raise_size']}")
            except Exception as e:
                print(f"  [ERR] Таблица {i}: {e}")

    # Если файл уже есть — аккуратно аппендим
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and "tables" in prev:
                prev["tables"].extend(results["tables"])
                results = prev
        except Exception:
            pass

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Сохранено {len(results['tables'])} таблиц в {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="путь к исходному PDF")
    ap.add_argument("--out", default="tables_from_pdf.json", help="куда сохранить JSON")
    ap.add_argument("--pages", default="", help="напр. '1,3,5' (1-based); по умолчанию все")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    pages = None
    if args.pages.strip():
        pages = [int(x) for x in re.split(r"[,\s]+", args.pages.strip()) if x]

    run(args.pdf, args.out, pages, dpi=args.dpi)
