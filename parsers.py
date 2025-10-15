# -*- coding: utf-8 -*-
"""
src/parsers.py — парсеры текста (стэки, руки) и утилиты по эффективному стеку.
Зависит от src.ocr_live (extract_number, round_to_half) — не переписываем логику дважды.
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, Iterable

from .ocr_live import extract_number, round_to_half

# ====== Парсер стэка (в BB) ======

def parse_stack_bb(text: str) -> Optional[float]:
    """
    Извлекает число из OCR-строки и приводит к float (BB).
    Возвращает None, если ничего похожего на число не найдено.
    Пример входа: "11.3 BB", "20bb", "  7,5  ", "Stack: 12".
    """
    if not text:
        return None
    val = extract_number(text)
    return val


# ====== Парсер руки (нормализация к виду AA / K7s / Q9o) ======

_RANKS = set(list("AKQJT98765432"))

# Варианты символов, которые иногда «ломает» OCR:
_SUBS = {
    "О": "O",   # русская О -> латинская O (иногда встретится в "Qo")
    "о": "o",
    "Х": "X",   # русская Х -> X (на сайзингах бывает)
    "х": "x",
    "С": "C",   # safety, чтобы не спутать 's'/'o'
    "с": "c",
    "l": "1",   # частая ошибка OCR, но для рангов не используется
}

def _cleanup(s: str) -> str:
    s = s.strip()
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    s = s.replace(" ", "")
    return s

def normalize_hand(raw: str) -> Optional[Tuple[str, Optional[bool]]]:
    """
    Приводит строку с рукой к нормализованному формату.

    Возвращает:
        - для пар: ("AA", None)
        - для непарных: ("K7", True)  -> suited (s)
                         ("Q9", False) -> offsuit (o)
        - None, если распознать нельзя.

    Правила:
      - Принимаем форматы: "AA", "K7", "K7s", "Q9o", "T9s", "jt", "a2o" и т.п.
      - Ранги всегда в верхнем регистре, 'T' = Ten.
      - Если суффикса нет:
           * в логике сетки чартов дальше решается по положению ячейки (лево/право от диагонали),
             но на уровне нормализации возвращаем suited=None (не знаем).
         Исключение: если явно "s" или "o" — выставляем True/False.
    """
    if not raw:
        return None

    s = _cleanup(raw).upper()

    # Быстрый кейс: пара, ровно 2 символа и одинаковые
    if len(s) == 2 and s[0] == s[1] and s[0] in _RANKS:
        return (s, None)  # пара, suited неприменим

    # Убираем возможные неалфавитные хвосты (например, лишние знаки)
    s = re.sub(r"[^AKQJT98765432SO]", "", s)

    # Попытка распознать формат XY[s|o]?
    m = re.match(r"^([AKQJT98765432])([AKQJT98765432])([SO])?$", s)
    if not m:
        return None

    r1, r2, suit_flag = m.group(1), m.group(2), m.group(3)

    # Пара?
    if r1 == r2:
        return (r1 + r2, None)

    # Непарная: определяем suited, если явно указан
    suited: Optional[bool]
    if suit_flag == "S":
        suited = True
    elif suit_flag == "O":
        suited = False
    else:
        suited = None  # неизвестно на этапе OCR (вычислим позднее по сетке)

    return (r1 + r2, suited)


def hand_key(rank2: str, suited: Optional[bool]) -> str:
    """
    Собирает ключ руки для обращения к сетке:
      - Пара: "AA"
      - Непара: "K7s" / "K7o" / "K7" (если suited неизвестен)
    """
    if len(rank2) == 2 and rank2[0] == rank2[1]:
        return rank2  # пара
    if suited is True:
        return f"{rank2}s"
    if suited is False:
        return f"{rank2}o"
    return rank2  # неизвестно — позволим внешней логике решить по позиции в сетке


# ====== Эффективный стек ======

def eff_stack_hu(hero_bb: float, opp_bb: float) -> float:
    """
    Эффективный стек для HU: минимум из двух, округление до 0.5.
    """
    return round_to_half(min(hero_bb, opp_bb))

def eff_stack_3max_three(btn_bb: float, sb_bb: float, bb_bb: float) -> float:
    """
    Эффективный стек для 3-max, когда в раздаче трое:
    медиана из трёх, затем округление до 0.5.
    """
    vals = sorted([btn_bb, sb_bb, bb_bb])
    median = vals[1]
    return round_to_half(median)

def eff_stack_3max_two(a_bb: float, b_bb: float) -> float:
    """
    Когда из троих кто-то уже сфолдил и осталось двое, считаем как HU.
    """
    return eff_stack_hu(a_bb, b_bb)


# ====== Вспомогательные проверки ======

def is_valid_rank2(s: str) -> bool:
    return bool(re.fullmatch(r"[AKQJT98765432]{2}", s))

def players_median_rounded(stacks: Iterable[float]) -> float:
    """
    Медиана произвольного итерируемого набора из трёх чисел с округлением до 0.5.
    Удобно для тестов.
    """
    vals = sorted(list(stacks))
    if len(vals) != 3:
        raise ValueError("players_median_rounded ожидает 3 значения")
    return round_to_half(vals[1])
