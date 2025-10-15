# -*- coding: utf-8 -*-
"""
Ручной конструктор мини-таблицы с автозаливкой по цветам.
Ключевые клавиши:
  G   — выбрать ROI сетки (обвести, затем 's')
  H   — задать spot_key и raise size
  A   — автозаливка по цветам
  C   — калибровка цветов (кликни по 4 клеткам: оранжевый, синий, зелёный, серый)
  V   — вкл/выкл полупрозрачную заливку ячеек (для наглядности)
  ←↑→↓, SPACE / F L R P, O, D, X, S — как раньше
Запуск:
  python -m src.manual_table_creator --img "C:\...\hu_bb_vs_limp.png" --out tables_from_pdf.json
"""

from __future__ import annotations
import os, json, argparse, re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

# --- эталонные HEX (из твоего описания) ---
HEX_ORANGE = "#ffdc6d"  # limp/call
HEX_BLUE   = "#71daff"  # raise
HEX_GREEN  = "#05ff76"  # push
HEX_GRAY   = "#f2f2f2"  # fold

COLOR_TO_ACTION = {
    HEX_ORANGE: "call_or_limp",
    HEX_BLUE:   "raise",
    HEX_GREEN:  "push",
    HEX_GRAY:   "fold",
}

# БОЛЕЕ МЯГКИЙ допуск по цвету
COLOR_TOL = 35.0

RANKS = list("AKQJT98765432")

# -------------------- utils --------------------

@dataclass
class ROI:
    x1:int; y1:int; x2:int; y2:int

@dataclass
class GridSpec:
    rows:int=13; cols:int=13; inner:int=4

def clamp_roi(img, roi: ROI) -> ROI:
    h,w = img.shape[:2]
    x1 = max(0, min(w, roi.x1)); x2 = max(0, min(w, roi.x2))
    y1 = max(0, min(h, roi.y1)); y2 = max(0, min(h, roi.y2))
    if x2<=x1: x2=x1+1
    if y2<=y1: y2=y1+1
    return ROI(x1,y1,x2,y2)

def crop(img, roi: ROI):
    r = clamp_roi(img, roi)
    return img[r.y1:r.y2, r.x1:r.x2].copy()

def iter_cells(grid_roi: ROI, spec: GridSpec):
    W = grid_roi.x2 - grid_roi.x1
    H = grid_roi.y2 - grid_roi.y1
    cw = W/spec.cols
    ch = H/spec.rows
    for r in range(spec.rows):
        for c in range(spec.cols):
            x1 = int(grid_roi.x1 + c*cw) + spec.inner
            y1 = int(grid_roi.y1 + r*ch) + spec.inner
            x2 = int(grid_roi.x1 + (c+1)*cw) - spec.inner
            y2 = int(grid_roi.y1 + (r+1)*ch) - spec.inner
            yield (r,c), ROI(x1,y1,x2,y2)

def cell_hand_key(r:int, c:int) -> str:
    r1, r2 = RANKS[r], RANKS[c]
    if r==c:  return f"{r1}{r2}"     # пары
    if c<r:   return f"{r1}{r2}o"    # слева — оффсьют
    return f"{r1}{r2}s"              # справа — суитед

def hex_to_bgr(hx: str) -> Tuple[int,int,int]:
    hx = hx.strip().lstrip("#")
    return (int(hx[4:6],16), int(hx[2:4],16), int(hx[0:2],16))

def bgr_to_lab(bgr: Tuple[int,int,int]) -> np.ndarray:
    arr = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    return lab[0,0,:].astype(np.float32)

# ТЕКУЩИЕ целевые цвета в Lab (можно перекалибровать клавишей C)
LAB_TARGETS = {
    HEX_ORANGE: bgr_to_lab(hex_to_bgr(HEX_ORANGE)),
    HEX_BLUE:   bgr_to_lab(hex_to_bgr(HEX_BLUE)),
    HEX_GREEN:  bgr_to_lab(hex_to_bgr(HEX_GREEN)),
    HEX_GRAY:   bgr_to_lab(hex_to_bgr(HEX_GRAY)),
}

def classify_action_by_color(bgr: Tuple[int,int,int]) -> Optional[str]:
    lab = bgr_to_lab(bgr)
    best_hx, dmin = None, 1e9
    for hx, lab_ref in LAB_TARGETS.items():
        d = float(np.linalg.norm(lab - lab_ref))
        if d < dmin:
            dmin, best_hx = d, hx
    if dmin <= COLOR_TOL:
        return COLOR_TO_ACTION.get(best_hx)
    return None

def sample_center_color(cell_bgr) -> Tuple[int,int,int]:
    h,w = cell_bgr.shape[:2]
    cx1, cx2 = w//4, 3*w//4
    cy1, cy2 = h//4, 3*h//4
    patch = cell_bgr[cy1:cy2, cx1:cx2]
    mean = patch.reshape(-1,3).mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))

def imread_unicode(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

# -------------------- редактор --------------------

class Editor:
    def __init__(self, img, grid_roi: ROI):
        self.img = img
        self.grid_roi = clamp_roi(img, grid_roi)
        self.spec = GridSpec()
        self.sel = [0,0]  # r,c
        self.grid: Dict[str, Dict] = { cell_hand_key(r,c): {"base":"none","overrides":[]} 
                                       for r in range(self.spec.rows)
                                       for c in range(self.spec.cols) }
        self.spot_key = "UNKNOWN"
        self.raise_size = {"min_x":2.0, "max_x":2.0, "default_x":2.0}
        self.fill_overlay = True  # заливка включена по умолчанию

    def _draw_cell(self, view, roi: ROI, base: str, selected=False):
        # цвета отрисовки (BGR)
        color_map = {
            "none": (60,60,60),
            "fold": (180,180,180),
            "call_or_limp": (200,200,0),
            "raise": (0,165,255),   # оранжевый
            "push": (60,200,60),
        }
        col = color_map.get(base, (60,60,60))

        if self.fill_overlay:
            overlay = view.copy()
            cv2.rectangle(overlay, (roi.x1,roi.y1), (roi.x2,roi.y2), col, thickness=-1)
            cv2.addWeighted(overlay, 0.28, view, 0.72, 0, dst=view)
            cv2.rectangle(view, (roi.x1,roi.y1), (roi.x2,roi.y2), col, 2)
        else:
            cv2.rectangle(view, (roi.x1,roi.y1), (roi.x2,roi.y2), col, 2)

        if selected:
            cv2.rectangle(view, (roi.x1,roi.y1), (roi.x2,roi.y2), (0,0,255), 3)

    def draw(self):
        view = self.img.copy()
        for (r,c), roi in iter_cells(self.grid_roi, self.spec):
            k = cell_hand_key(r,c)
            base = self.grid[k]["base"]
            self._draw_cell(view, roi, base, selected=False)

        # подсветка выделенной
        (r,c), roi = list(iter_cells(self.grid_roi, self.spec))[self.sel[0]*13 + self.sel[1]]
        self._draw_cell(view, roi, self.grid[self.current_key()]["base"], selected=True)

        # панель сверху
        panel = np.zeros((120, view.shape[1], 3), dtype=np.uint8)
        msgs = [
            f"SPOT={self.spot_key} | RAISE: {self.raise_size} | FILL={'ON' if self.fill_overlay else 'OFF'}",
            "G pick GRID | H header | A auto | C calibrate | V fill on/off | S save | Q/Esc quit",
            "Arrows move | SPACE cycle | F/L/R/P set base | O add override | D pop | X clear"
        ]
        y=25
        for t in msgs:
            cv2.putText(panel, t, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            y+=30

        return np.vstack([panel, view])

    def current_key(self) -> str:
        return cell_hand_key(self.sel[0], self.sel[1])

    def cycle_base(self):
        order = ["none","fold","call_or_limp","raise","push"]
        k = self.current_key()
        cur = self.grid[k]["base"]
        self.grid[k]["base"] = order[(order.index(cur)+1)%len(order)]

    def set_base(self, act: str):
        if act not in ("fold","call_or_limp","raise","push","none"): return
        self.grid[self.current_key()]["base"] = act

    def add_override(self, act: str, n: float):
        if act not in ("fold","call_or_limp","raise","push"): return
        item = {"act":act, "n": float(n)}
        cell = self.grid[self.current_key()]
        cell["overrides"].append(item)
        cell["overrides"].sort(key=lambda t: t["n"])

    def pop_override(self):
        cell = self.grid[self.current_key()]
        if cell["overrides"]:
            cell["overrides"].pop()

    def clear_overrides(self):
        self.grid[self.current_key()]["overrides"] = []

    def auto_from_colors(self):
        counts = {"fold":0, "call_or_limp":0, "raise":0, "push":0, "none":0}
        for (r,c), roi in iter_cells(self.grid_roi, self.spec):
            cell = crop(self.img, roi)
            bgr = sample_center_color(cell)
            act = classify_action_by_color(bgr) or "none"
            k = cell_hand_key(r,c)
            self.grid[k]["base"] = act
            counts[act] = counts.get(act,0) + 1
        print("[AUTO] counts:", counts)

    def set_header(self, spot_key: str, raise_spec: str):
        self.spot_key = spot_key.strip()
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[-–]\s*([0-9]+(?:\.[0-9]+)?))?\s*$", raise_spec)
        if m:
            a = float(m.group(1)); b = m.group(2)
            if b is None:
                self.raise_size = {"min_x":a, "max_x":a, "default_x":a}
            else:
                x1, x2 = min(a,float(b)), max(a,float(b))
                self.raise_size = {"min_x":x1, "max_x":x2, "default_x":x1}
        else:
            self.raise_size = {"min_x":2.0, "max_x":2.0, "default_x":2.0}

# -------------------- ROI picker --------------------

class RoiPicker:
    def __init__(self, img):
        self.img = img
        self.pts = []
        self.view = img.copy()
        self.title = "Pick GRID ROI (drag LMB, then 's' to confirm, 'r' reset)"

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts = [(x,y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.pts.append((x,y))
            self.view = self.img.copy()
            cv2.rectangle(self.view, self.pts[0], self.pts[1], (0,255,0), 2)

    def pick(self) -> ROI:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.on_mouse)
        while True:
            cv2.imshow(self.title, self.view)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s') and len(self.pts)==2:
                break
            if key == ord('r'):
                self.pts = []; self.view = self.img.copy()
            if key in (27, ord('q')):
                cv2.destroyWindow(self.title); raise SystemExit("Cancelled.")
        cv2.destroyWindow(self.title)
        (x1,y1),(x2,y2) = self.pts
        x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
        return ROI(x1,y1,x2,y2)

# -------------------- калибровка цветов --------------------

def _click_get_bgr(win_name, img) -> Tuple[int,int,int]:
    got = {"bgr": None}
    def cb(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b,g,r = img[y, x].tolist()
            got["bgr"] = (int(b),int(g),int(r))
    cv2.setMouseCallback(win_name, cb)
    while True:
        cv2.imshow(win_name, img)
        k = cv2.waitKey(10) & 0xFF
        if got["bgr"] is not None or k in (27, ord('q')):
            break
    cv2.setMouseCallback(win_name, lambda *a: None)
    return got["bgr"]

def calibrate_colors(img, grid_roi: ROI):
    global LAB_TARGETS
    view = img.copy()
    cv2.rectangle(view, (grid_roi.x1,grid_roi.y1), (grid_roi.x2,grid_roi.y2), (0,255,0), 2)

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    steps = [
        (HEX_ORANGE, "Оранжевый (LIMP/CALL) — кликни по клетке"),
        (HEX_BLUE,   "Синий (RAISE) — кликни по клетке"),
        (HEX_GREEN,  "Зелёный (PUSH) — кликни по клетке"),
        (HEX_GRAY,   "Серый (FOLD) — кликни по клетке"),
    ]
    new_targets = {}
    for hx, msg in steps:
        tmp = view.copy()
        cv2.putText(tmp, msg, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(tmp, msg, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        bgr = _click_get_bgr("Calibrate", tmp)
        if bgr is not None:
            new_targets[hx] = bgr_to_lab(bgr)
    cv2.destroyWindow("Calibrate")
    if new_targets:
        LAB_TARGETS.update(new_targets)
        print("[CALIB] обновлены эталоны для:", [k for k in new_targets.keys()])
    else:
        print("[CALIB] пропущено")

# -------------------- основной цикл --------------------

def run(img_path: str, out_json: str):
    img = imread_unicode(img_path)
    if img is None:
        raise SystemExit(f"Не удалось открыть изображение: {img_path}")

    print("Нажми G в окне, обведи сетку 13×13 и нажми 's'.")

    grid_picker = RoiPicker(img)
    grid_roi: Optional[ROI] = None
    editor: Optional[Editor] = None

    while True:
        base = img.copy()
        cv2.putText(base, "Keys: G pick GRID | H header | A auto | C calibrate | V fill | S save | Q quit", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(base, "Keys: G pick GRID | H header | A auto | C calibrate | V fill | S save | Q quit", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        if grid_roi:
            ed_view = editor.draw()
            cv2.imshow("Manual Table Editor", ed_view)
        else:
            cv2.imshow("Manual Table Editor", base)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('g'):
            grid_roi = grid_picker.pick()
            editor = Editor(img, grid_roi)
        elif editor is None:
            continue
        elif key == ord('h'):
            spot = input("spot_key (например: HU_BB_vs_LIMP): ").strip()
            raise_spec = input("raise size (например '2.5-2' или '2'): ").strip()
            editor.set_header(spot, raise_spec)
        elif key == ord('a'):
            editor.auto_from_colors()
        elif key == ord('c'):
            calibrate_colors(img, editor.grid_roi)
        elif key == ord('v'):
            editor.fill_overlay = not editor.fill_overlay
        elif key == ord('s'):
            entry = {
                "spot_key": editor.spot_key or "UNKNOWN",
                "vs_profile": "fish",
                "raise_size": editor.raise_size,
                "grid": editor.grid
            }
            if os.path.exists(out_json):
                try:
                    with open(out_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {"tables":[]}
            else:
                data = {"tables":[]}
            data.setdefault("tables", []).append(entry)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Добавлено: {entry['spot_key']}  →  {out_json}")
        else:
            r,c = editor.sel
            if key == 81:   editor.sel[1] = max(0, c-1)       # ←
            elif key == 83: editor.sel[1] = min(12, c+1)      # →
            elif key == 82: editor.sel[0] = max(0, r-1)       # ↑
            elif key == 84: editor.sel[0] = min(12, r+1)      # ↓
            elif key == 32: editor.cycle_base()               # SPACE
            elif key == ord('f'): editor.set_base("fold")
            elif key == ord('l'): editor.set_base("call_or_limp")
            elif key == ord('r'): editor.set_base("raise")
            elif key == ord('p'): editor.set_base("push")
            elif key == ord('x'): editor.clear_overrides()
            elif key == ord('d'): editor.pop_override()
            elif key == ord('o'):
                typ = input("override act [push/call_or_limp/fold/raise]: ").strip().lower()
                n = float(input("N (порог, читается как <= N+0.5): ").replace(",", "."))
                editor.add_override(typ, n)

    cv2.destroyAllWindows()

# -------------------- CLI --------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="PNG/JPG скрин одной мини-таблицы")
    ap.add_argument("--out", default="tables_from_pdf.json", help="куда сохранять общий JSON")
    args = ap.parse_args()
    run(args.img, args.out)
