# -*- coding: utf-8 -*-
"""
Резолвер префлоп-решения по разметке из tables_from_pdf.json.

Структура JSON (как сохраняет manual_table_creator.py):
{
  "tables": [
    {
      "spot_key": "HU_BB_vs_LIMP",
      "vs_profile": "fish",
      "raise_size": {"min_x":2.5, "max_x":2.0, "default_x":2.5},
      "grid": {
        "AA": {"base":"push","overrides":[]},
        "K7s":{"base":"raise","overrides":[{"act":"push","n":10},{"act":"call_or_limp","n":12}]},
        ...
      }
    },
    ...
  ]
}
"""

from __future__ import annotations
import json
import os
from functools import lru_cache
from typing import Dict, Any, Optional

RANKS = "AKQJT98765432"

# ---------- загрузка базы ----------

@lru_cache(maxsize=1)
def _load_tables(path: str = "tables_from_pdf.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"tables": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "tables" not in data:
        return {"tables": []}
    return data

def _get_table(spot_key: str) -> Optional[Dict[str, Any]]:
    data = _load_tables()
    for t in data.get("tables", []):
        if t.get("spot_key") == spot_key:
            return t
    return None

# ---------- утилиты ----------

def _round_half(x: float) -> float:
    # округление до ближайших 0.5
    return round(x * 2.0) / 2.0

def _median3(a: float, b: float, c: float) -> float:
    trio = sorted([a, b, c])
    return trio[1]

def _effective_stack(state: Dict[str, Any]) -> float:
    """
    HU: min(герой, оппонент)
    3-max (трое в игре): медиана трёх, потом округление до 0.5
    Если кто-то сфолдил (позже появится флаг) — применим HU-правило.
    """
    mode = (state.get("mode") or "").upper()
    sbbs = state.get("stacks_bb", {}) or {}
    h = float(sbbs.get("hero") or 0.0)
    l = float(sbbs.get("vill_left") or 0.0)
    r = float(sbbs.get("vill_right") or 0.0)

    if mode == "HU":
        opp = r if r > 0 else l
        eff = min(h, opp if opp > 0 else h)
    else:
        eff = _median3(h, l, r)

    return _round_half(eff)

def _normalize_hand_key(hero_hand) -> Optional[str]:
    """
    Превращает распознанную руку в ключ вида:
    - 'TT' для пар
    - 'K7s'/'K7o' для неподпарных.
    При неизвестном суите попробуем 's' затем 'o'.
    """
    if hero_hand is None:
        return None

    # варианты входа: "K7s" / ("K7","s") / ("TT", None) / ["K7","o"]
    if isinstance(hero_hand, str):
        key = hero_hand.strip().upper()
        if len(key) in (2,3):
            return key
        return None

    if isinstance(hero_hand, (list, tuple)) and hero_hand:
        a = hero_hand[0]
        b = hero_hand[1] if len(hero_hand) > 1 else None
        if isinstance(a, str):
            rank = a.strip().upper()
            if len(rank) == 2 and rank[0] == rank[1]:
                return rank  # пары
            if len(rank) == 2:
                if isinstance(b, str) and b:
                    sfx = b.strip().lower()[0]
                    if sfx in ("s", "o"):
                        return rank + sfx
                # неизвестный суит – попробуем s, затем o
                return None  # пусть решится ниже
    return None

def _decide_action_label(act: str, spot_key: str, state: Dict[str, Any], raise_size: Dict[str, float]) -> str:
    act = (act or "").lower()
    if act == "push":
        return "PUSH"
    if act == "fold":
        return "FOLD"
    if act == "raise":
        x = raise_size.get("default_x") or raise_size.get("min_x") or 2.0
        # формат CAPS как просил
        return f"RAISE {x:g}X"
    if act == "call_or_limp":
        spot = (spot_key or "").lower()
        pos = (state.get("hero_pos") or "").upper()
        # Heuristics:
        #  - BB vs LIMP -> CHECK
        if "vs_limp" in spot and "BB" in pos:
            return "CHECK"
        #  - SB or BTN открывается -> LIMP
        if (" sb" in f" {pos}") or ("_SB" in spot and "vs" not in spot):
            return "LIMP"
        #  - против MR/PUSH -> CALL
        if "vs_mr" in spot or "vs_3x" in spot or "opush" in spot or "vs push" in spot or "push" in spot:
            return "CALL"
        # запасной вариант
        return "CALL"
    return "FOLD"

# ---------- ядро выбора клетки ----------

def _apply_overrides(eff: float, base: str, overrides: list) -> str:
    """
    overrides отсортированы по N по возрастанию.
    Берём первый, у которого eff <= N + 0.5 (граница принадлежит квадрату).
    Если ни один не подошёл — базовый цвет.
    """
    if not overrides:
        return base
    for ov in sorted(overrides, key=lambda t: float(t.get("n", 0.0))):
        n = float(ov.get("n", 0.0))
        if eff <= (n + 0.5):
            return ov.get("act") or base
    return base

def _hand_variants(hand_key: Optional[str]) -> list[str]:
    """
    Возвращает список ключей, которые попробуем по очереди,
    если суит неизвестен.
    """
    if hand_key:
        return [hand_key]
    # неизвестный суит, попробуем оба: s, o.
    # В реальности сюда попадём, если OCR не смог определить su/off.
    return []

def _lookup_cell(table: Dict[str, Any], hand_key: Optional[str]) -> Optional[Dict[str, Any]]:
    grid = table.get("grid") or {}
    # 1) точное совпадение
    if hand_key and hand_key in grid:
        return grid[hand_key]
    # 2) если суит не ясен: попробуем 's' затем 'o'
    if hand_key and len(hand_key) == 2:  # например "K7" без суффикса
        for sfx in ("s", "o"):
            if hand_key + sfx in grid:
                return grid[hand_key + sfx]
    # 3) если распознали только непарные ранги (без суита) — попробуем обе
    if not hand_key:
        return None
    return None

# ---------- публичный API ----------

def lookup_decision(state: Dict[str, Any]) -> str:
    """
    Возвращает одно CAPS-действие: PUSH / CALL / CHECK / LIMP / RAISE 2X / FOLD.
    """
    spot_key = state.get("spot_key") or ""
    t = _get_table(spot_key)
    if not t:
        return "FOLD"  # нет таблицы — безопасный дефолт

    eff = _effective_stack(state)  # уже округлён до 0.5

    # распознанная рука
    raw_hand = state.get("hero_hand")
    hand_key = _normalize_hand_key(raw_hand)
    cell = _lookup_cell(t, hand_key)

    # если суит неясен и мы получили 'K7' без суффикса — попробуем 's'/'o'
    if cell is None and isinstance(raw_hand, (list, tuple)) and isinstance(raw_hand[0], str):
        base_pair = raw_hand[0].strip().upper()
        if len(base_pair) == 2 and base_pair[0] != base_pair[1]:
            for sfx in ("s", "o"):
                hk = base_pair + sfx
                if hk in (t.get("grid") or {}):
                    cell = t["grid"][hk]
                    break

    if not cell:
        return "FOLD"

    base = (cell.get("base") or "fold").lower()
    overrides = cell.get("overrides") or []
    chosen = _apply_overrides(eff, base, overrides)

    return _decide_action_label(chosen, spot_key, state, t.get("raise_size") or {})
