# auto_parse_vs_fish.py (v2: multi-detector, debug dumps, validation)
# deps: pip install opencv-python-headless pillow numpy pytesseract pymupdf
# system: tesseract-ocr in PATH (or pass --tesspath "C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

import sys, os, json, re, argparse, math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2
import pytesseract

# ============ CONFIG ============
COLOR_MAP_RGB = {
    "fold":        (242, 242, 242),  # #f2f2f2
    "limp_call":   (255, 220, 109),  # #ffdc6d
    "raise":       (113, 218, 255),  # #71daff
    "push":        (5,   255, 118),  # #05ff76
}
LAB_DIST_THRESH = 20.0         # чуть свободнее, чем было
RENDER_SCALE_DEFAULT = 3.0
OCR_DIGITS_CFG = r'--psm 7 -l eng -c tessedit_char_whitelist=0123456789,.'

# детекторные режимы
DETECT_MODES = ("auto","lines","green","palette")

# ============ COLOR UTILS ============
def rgb2lab(rgb: Tuple[int,int,int]) -> np.ndarray:
    rgb_arr = np.array([[rgb]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
    return lab[0,0,:].astype(np.float32)

LAB_REF = {k: rgb2lab(v) for k,v in COLOR_MAP_RGB.items()}

def lab_distance(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    return float(np.linalg.norm(lab_a - lab_b))

def classify_cell_color(rgb: Tuple[int,int,int]) -> Optional[str]:
    lab = rgb2lab(rgb)
    best = None
    best_d = 1e9
    for k, ref in LAB_REF.items():
        d = lab_distance(lab, ref)
        if d < best_d:
            best_d, best = d, k
    return best if best_d <= LAB_DIST_THRESH else None

def avg_cell_color(img_rgb: np.ndarray) -> Tuple[int,int,int]:
    h, w, _ = img_rgb.shape
    pad_y = max(2, h//6)
    pad_x = max(2, w//6)
    crop = img_rgb[pad_y:h-pad_y, pad_x:w-pad_x]
    if crop.size == 0:
        crop = img_rgb
    mean = crop.reshape(-1,3).mean(0)
    return tuple(int(round(x)) for x in mean)

# ============ LOW-LEVEL ============
def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw

def ensure_dir(d: Optional[str]):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

# ============ ROI DETECTORS ============

def find_tables_by_lines(page_rgb: np.ndarray, debug_dir: Optional[str], page_idx:int) -> List[Tuple[int,int,int,int]]:
    gray = to_gray(page_rgb)
    bw = binarize_otsu(gray)
    inv = cv2.bitwise_not(bw)

    h, w = inv.shape
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h//60)))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w//60), 1))
    vert = cv2.dilate(cv2.erode(inv, vert_kernel, 1), vert_kernel, 1)
    hori = cv2.dilate(cv2.erode(inv, hori_kernel, 1), hori_kernel, 1)
    grid = cv2.bitwise_and(vert, hori)
    grid = cv2.dilate(grid, np.ones((3,3), np.uint8), 1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"p{page_idx:02d}_lines_grid.png"), grid)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        area = wc*hc
        if area < 20000 or wc < 200 or hc < 200:  # пороги стали мягче
            continue
        rois.append((x,y,wc,hc))
    rois.sort(key=lambda r: (r[1]//50, r[0]))
    return rois

def _palette_mask(page_rgb: np.ndarray, thr: float = LAB_DIST_THRESH+2.0) -> np.ndarray:
    """Пиксели, близкие к любой из 4 эталонных заливок ячеек."""
    lab = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    acc = None
    for ref in LAB_REF.values():
        d = np.linalg.norm(lab - ref.reshape(1,1,3), axis=2)
        m = (d <= thr).astype(np.uint8)
        acc = m if acc is None else np.maximum(acc, m)
    return (acc*255).astype(np.uint8)

def find_tables_by_palette(page_rgb: np.ndarray, debug_dir: Optional[str], page_idx:int) -> List[Tuple[int,int,int,int]]:
    mask = _palette_mask(page_rgb)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8), 2)
    mask = cv2.medianBlur(mask, 5)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"p{page_idx:02d}_palette_mask.png"), mask)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois=[]
    H,W = mask.shape
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        area = wc*hc
        if area < 40000 or wc < 260 or hc < 260:
            continue
        # отсечь странные сверхширокие блоки (целая полоса)
        if wc > 0.95*W and hc < 0.25*H:
            continue
        rois.append((x,y,wc,hc))
    rois.sort(key=lambda r:(r[1]//50, r[0]))
    return rois

def find_tables_by_green_headers(page_rgb: np.ndarray, debug_dir: Optional[str], page_idx:int) -> List[Tuple[int,int,int,int]]:
    """Ищем зелёные шапки (цвет близок к push #05ff76), затем ниже ищем палитровый блок."""
    lab = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = LAB_REF["push"]
    d = np.linalg.norm(lab - ref.reshape(1,1,3), axis=2)
    mask_g = (d <= (LAB_DIST_THRESH+5)).astype(np.uint8)*255
    # сгладим узкие разрывы
    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_CLOSE, np.ones((21,3),np.uint8), 1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"p{page_idx:02d}_green_mask.png"), mask_g)

    H,W = mask_g.shape
    # найдём горизонтальные полосы
    ysum = mask_g.sum(axis=1)
    rows = np.where(ysum > 0.25*255*W)[0]  # полоса, занимающая значимую ширину
    if len(rows)==0:
        return []

    # сгруппируем соседние y в кластеры(полосы)
    bands=[]
    start=rows[0]
    prev=rows[0]
    for y in rows[1:]:
        if y - prev <= 4:
            prev = y
        else:
            bands.append((start, prev))
            start = y; prev = y
    bands.append((start, prev))

    # для каждой полосы посмотрим ниже «палитровую» область и возьмём её bbox
    rois=[]
    pal = _palette_mask(page_rgb)
    pal = cv2.morphologyEx(pal, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), 1)

    for (y0,y1) in bands:
        y_top = min(y1+8, H-1)
        y_bot = min(y_top + int(0.6*H), H-1)  # не уходим слишком вниз
        sub = pal[y_top:y_bot, :]
        contours,_ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,wc,hc = cv2.boundingRect(cnt)
            xg, yg, wg, hg = x, y_top + y, wc, hc
            area = wg*hg
            if area < 35000 or wg < 260 or hg < 260:
                continue
            rois.append((xg, yg, wg, hg))

    rois.sort(key=lambda r:(r[1]//50, r[0]))
    # склеим/удалим сильные пересечения
    filtered=[]
    for r in rois:
        if all(not _overlap_more_than(r, q, 0.75) for q in filtered):
            filtered.append(r)
    if debug_dir:
        dbg = page_rgb.copy()
        for i,(x,y,w,h) in enumerate(filtered):
            cv2.rectangle(dbg, (x,y), (x+w, y+h), (255,0,0), 3)
            cv2.putText(dbg, f"roi{i}", (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imwrite(os.path.join(debug_dir, f"p{page_idx:02d}_green_rois.png"), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
    return filtered

# ============ OCR HELPERS ============
def ocr_text(img_rgb: np.ndarray, psm: int = 6, lang: str = "eng") -> str:
    cfg = f"--psm {psm} -l {lang}"
    return pytesseract.image_to_string(img_rgb, config=cfg).strip()

def ocr_digits(img_rgb: np.ndarray) -> Optional[int]:
    gray = to_gray(img_rgb)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(bw, config=OCR_DIGITS_CFG)
    txt = txt.strip().replace(' ', '')
    m = re.search(r'(\d{1,2})', txt)
    if not m:
        return None
    try:
        return int(m.group(1))
    except:
        return None

# ============ HEADER PARSING ============
def parse_header_get_meta(header_rgb: np.ndarray) -> Dict[str, Any]:
    # попытаемся с двумя высотами OCR: как есть и дополнительно слегка увеличить
    def parse_once(img):
        txt = ocr_text(img, psm=6, lang="eng+rus").upper()
        vs = "FISH" if "FISH" in txt else None
        spot = re.sub(r'\(.*?\)', '', txt)
        spot = spot.replace('—',' ').replace('-', ' ')
        spot = re.sub(r'R\s*\d+(\.\d+)?X.*$', '', spot)
        spot = re.sub(r'[^A-Z0-9 ]', ' ', spot)
        spot = re.sub(r'\s+', ' ', spot).strip()
        tokens = [t for t in spot.split(' ') if t]
        if tokens and tokens[0] == '3':
            tokens = ['3MAX'] + tokens[1:]
        spot_key = "_".join(tokens[:6]) if tokens else "UNKNOWN"
        nums = re.findall(r'(\d+(?:[.,]\d+)?)\s*X', txt)
        nums = [float(x.replace(',','.')) for x in nums]
        raise_size = None
        if nums:
            if len(nums) == 1:
                raise_size = {"min_x":nums[0], "max_x":nums[0], "default_x":nums[0]}
            else:
                mn, mx = min(nums), max(nums)
                raise_size = {"min_x":mn, "max_x":mx, "default_x":mx}
        return {"spot_key": spot_key, "vs_profile": vs or "fish", "raise_size": raise_size}
    meta = parse_once(header_rgb)
    if meta["spot_key"] == "UNKNOWN":
        # попробуем слегка увеличить контраст
        hdr = cv2.cvtColor(header_rgb, cv2.COLOR_RGB2GRAY)
        hdr = cv2.convertScaleAbs(hdr, alpha=1.6, beta=10)
        hdr = cv2.cvtColor(hdr, cv2.COLOR_GRAY2RGB)
        meta = parse_once(hdr)
    return meta

# ============ OVERRIDES DETECTOR ============
def find_overrides_in_cell(cell_rgb: np.ndarray) -> List[Dict[str, Any]]:
    h,w,_ = cell_rgb.shape
    hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    inv = cv2.bitwise_not(v)
    inv = cv2.medianBlur(inv, 3)
    _, bw = cv2.threshold(inv, 230, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), 1)
    contours,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overrides = []
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        area = wc*hc
        if area < (h*w)*0.012 or area > (h*w)*0.32:
            continue
        pad = 2
        x0 = max(0, x-pad); y0=max(0,y-pad)
        x1 = min(w, x+wc+pad); y1=min(h, y+hc+pad)
        chip = cell_rgb[y0:y1, x0:x1]
        if chip.size == 0: 
            continue
        chip_lab = cv2.cvtColor(chip, cv2.COLOR_RGB2LAB)
        mask = chip_lab[:,:,0] > 30
        if mask.sum() < 10:
            continue
        avg_rgb = tuple(int(round(v)) for v in chip[mask].reshape(-1,3).mean(0))
        act_key = classify_cell_color(avg_rgb)
        if act_key is None:
            continue
        n = ocr_digits(chip)
        if n is None:
            continue
        overrides.append({"act": act_key, "n": n})

    overrides.sort(key=lambda d: d["n"])
    uniq, seen = [], set()
    for ov in overrides:
        key=(ov["act"], ov["n"])
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(ov)
    return uniq

# ============ 13x13 GRID KEYS ============
RANKS = list("AKQJT98765432")
ACTIONS = {"fold","limp_call","raise","push"}

def hand_key(r_i: int, c_i: int) -> str:
    r = RANKS[r_i]; c = RANKS[c_i]
    if r_i == c_i:
        return r+r
    return (c+r+"o") if c_i < r_i else (r+c+"s")

def all_169_keys() -> List[str]:
    return [hand_key(ri,ci) for ri in range(13) for ci in range(13)]

VALID_169_SET = set(all_169_keys())

# ============ TABLE EXTRACT ============
def extract_table(page_rgb: np.ndarray, roi: Tuple[int,int,int,int], debug_dir: Optional[str], idx:int) -> Dict[str, Any]:
    x,y,w,h = roi
    tbl = page_rgb[y:y+h, x:x+w]
    if tbl.size == 0:
        raise ValueError("Empty ROI")

    # шапка (эвристика)
    head_h = max(int(h*0.20), 60)
    head_h = min(head_h, h-180) if h>260 else min(head_h, h//4)
    header_rgb = tbl[:head_h]
    grid_rgb = tbl[head_h:]

    meta = parse_header_get_meta(header_rgb)

    gh, gw, _ = grid_rgb.shape
    cell_w = gw / 13.0
    cell_h = gh / 13.0

    grid: Dict[str, Dict[str, Any]] = {}
    for ri in range(13):
        for ci in range(13):
            x0 = int(round(ci*cell_w)); x1 = int(round((ci+1)*cell_w))
            y0 = int(round(ri*cell_h)); y1 = int(round((ri+1)*cell_h))
            cell = grid_rgb[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            base = classify_cell_color(avg_cell_color(cell)) or "fold"
            overrides = find_overrides_in_cell(cell)
            grid[hand_key(ri, ci)] = {"base": base, "overrides": overrides}

    if debug_dir:
        # сохранить сам ROI
        cv2.imwrite(os.path.join(debug_dir, f"roi_{idx:03d}.png"), cv2.cvtColor(tbl, cv2.COLOR_RGB2BGR))
    return {
        "spot_key": meta["spot_key"],
        "vs_profile": meta["vs_profile"],
        "raise_size": meta["raise_size"],
        "grid": grid
    }

# ============ PAGE RENDER ============
def render_pdf_pages(pdf_path: str, scale: float = RENDER_SCALE_DEFAULT) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    pages_rgb=[]
    for i in range(len(doc)):
        page = doc.load_page(i)
        mtx = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mtx, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages_rgb.append(np.array(img))
    return pages_rgb

# ============ PARSE PDF ============
def _overlap_more_than(a, b, thr=0.8):
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ax1, ay1 = ax+aw, ay+ah
    bx1, by1 = bx+bw, by+bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    union = aw*ah + bw*bh - inter
    return (inter/union >= thr) if union else False

def _grid_quality_ok(grid: Dict[str, Any]) -> bool:
    non_empty = sum(1 for v in grid.values() if v["base"] in ("raise","push","limp_call"))
    return non_empty >= 10

def detect_rois(page_rgb: np.ndarray, mode:str, debug_dir: Optional[str], page_idx:int) -> List[Tuple[int,int,int,int]]:
    if mode == "lines":
        return find_tables_by_lines(page_rgb, debug_dir, page_idx)
    if mode == "green":
        return find_tables_by_green_headers(page_rgb, debug_dir, page_idx)
    if mode == "palette":
        return find_tables_by_palette(page_rgb, debug_dir, page_idx)
    # auto: каскад
    for m in ("lines","green","palette"):
        rois = detect_rois(page_rgb, m, debug_dir, page_idx)
        if rois:
            return rois
    return []

def parse_pdf_to_json(pdf_path: str, out_path: str, scale: float, mode:str, debug_dir: Optional[str]) -> Dict[str, Any]:
    ensure_dir(debug_dir)
    pages = render_pdf_pages(pdf_path, scale)
    all_tables = []
    roi_count = 0
    for p_idx, page_rgb in enumerate(pages):
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"p{p_idx:02d}.png"), cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR))
        rois = detect_rois(page_rgb, mode, debug_dir, p_idx)
        # удалим дубли
        filtered=[]
        for r in sorted(rois, key=lambda r:(r[1],r[0])):
            if all(not _overlap_more_than(r, q, 0.75) for q in filtered):
                filtered.append(r)
        for roi in filtered:
            try:
                tbl = extract_table(page_rgb, roi, debug_dir, roi_count)
                roi_count += 1
                if _grid_quality_ok(tbl["grid"]):
                    all_tables.append(tbl)
            except Exception:
                pass
    result = {"tables": all_tables}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result

# ============ VALIDATION CLI ============
def validate_json(path: str) -> int:
    if not os.path.isfile(path):
        print(f"[validate] файл не найден: {path}")
        return 1
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[validate] не удалось прочитать JSON: {e}")
        return 1

    tables = data.get("tables", [])
    if not isinstance(tables, list) or not tables:
        print("[validate] нет массива tables или он пуст.")
        return 1

    valid = {hand_key(ri,ci) for ri in range(13) for ci in range(13)}
    rc = 0
    for idx, t in enumerate(tables):
        spot = t.get("spot_key") or f"table_{idx}"
        grid = t.get("grid")
        if not isinstance(grid, dict):
            print(f"[{spot}] grid отсутствует или не dict")
            rc = 1
            continue
        keys_set = set(grid.keys())
        missing = valid - keys_set
        extra   = keys_set - valid
        if missing or extra:
            print(f"[{spot}] ❌ размер grid={len(keys_set)} (ожидалось 169)")
            if missing:
                print(f"   отсутствуют: {sorted(list(missing))[:10]}{' ...' if len(missing)>10 else ''}")
            if extra:
                print(f"   лишние: {sorted(list(extra))[:10]}{' ...' if len(extra)>10 else ''}")
            rc = 1
        else:
            print(f"[{spot}] ✅ 169 рук")
        # базовые проверки значений
        ACTIONS = {"fold","limp_call","raise","push"}
        for hk, cell in grid.items():
            if not isinstance(cell, dict):
                print(f"   [{spot}] {hk}: cell не dict"); rc = 1; continue
            base = cell.get("base")
            if base not in ACTIONS:
                print(f"   [{spot}] {hk}: base \"{base}\" вне допустимых {ACTIONS}"); rc = 1
            ovs = cell.get("overrides", [])
            if not isinstance(ovs, list):
                print(f"   [{spot}] {hk}: overrides не list"); rc = 1; continue
            for ov in ovs:
                if not isinstance(ov, dict):
                    print(f"   [{spot}] {hk}: override не dict"); rc = 1; continue
                act = ov.get("act"); n = ov.get("n")
                if act not in ACTIONS or not isinstance(n, int):
                    print(f"   [{spot}] {hk}: override некорректен (act={act}, n={n})"); rc = 1
    if rc == 0:
        print("[validate] ✅ всё корректно")
    else:
        print("[validate] ❌ найдены ошибки")
    return rc

# ============ MAIN ============
def main():
    parser = argparse.ArgumentParser(description="Парсер PDF vs_fish -> JSON + валидация 169 рук")
    parser.add_argument("pdf", nargs="?", help="Путь к PDF, напр. 'vs_fish_final_19.03 (2).pdf'")
    parser.add_argument("--out", default="tables_from_pdf.json", help="Файл результата JSON")
    parser.add_argument("--scale", type=float, default=RENDER_SCALE_DEFAULT, help="Масштаб рендера PDF (напр. 4.0)")
    parser.add_argument("--mode", choices=DETECT_MODES, default="auto", help="Способ поиска таблиц (auto/lines/green/palette)")
    parser.add_argument("--debug-dumps", dest="debug_dir", help="Папка для PNG отладочных масок и ROI")
    parser.add_argument("--validate", action="store_true", help="После парсинга проверить JSON (169 ключей в каждой таблице)")
    parser.add_argument("--validate-json", dest="validate_json", help="Проверить уже готовый JSON и выйти")
    parser.add_argument("--tesspath", help="Путь к tesseract.exe (Windows), напр. 'C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe'")
    args = parser.parse_args()

    if args.validate_json:
        sys.exit(validate_json(args.validate_json))

    if args.tesspath:
        pytesseract.pytesseract.tesseract_cmd = args.tesspath

    if not args.pdf:
        print("Укажи PDF или используй --validate-json <file.json>")
        sys.exit(1)

    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        print(f"Не найден файл: {pdf_path}")
        sys.exit(1)

    print(f"[+] Рендер PDF: {pdf_path} (scale={args.scale}, mode={args.mode})")
    data = parse_pdf_to_json(pdf_path, args.out, args.scale, args.mode, args.debug_dir)
    print(f"[+] Готово. Таблиц: {len(data['tables'])}. JSON: {args.out}")

    if args.validate:
        code = validate_json(args.out)
        sys.exit(code)

if __name__ == "__main__":
    main()
