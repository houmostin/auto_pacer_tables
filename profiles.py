# src/profiles.py
PROFILES = {
    "my_laptop_1920x1080": {
        "CAP_REGION": [400, 200, 1520, 900],     # окно тренажера
        "ROI_STACK_HERO":  [740, 820,  900, 860],
        "ROI_STACK_LEFT":  [460, 320,  600, 360],
        "ROI_STACK_RIGHT": [1300,320, 1440, 360],
        "ROI_POT":         [880, 520, 1030, 555],
        "ROI_HAND":        [860, 770, 1010, 815],
    }
}
ACTIVE = "my_laptop_1920x1080"

# src/parsers.py
import re

def parse_stack_bb(txt: str):
    # "14.5 BB" | "14 BB" | "14,5 BB"
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*B+ ?B?", txt, re.I)
    return float(m.group(1).replace(",", ".")) if m else None

RANK = "AKQJT98765432"
def normalize_hand(txt: str):
    t = txt.upper().replace(" ", "")
    m = re.findall(r"[AKQJT2-9]", t)
    if len(m) >= 2:
        r1, r2 = m[0], m[1]
        if r1 == r2: return f"{r1}{r2}"
        # пока считаем offsuit, позже добавим масти => s/o
        return f"{r1}{r2}o"
    return None

