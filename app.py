# -*- coding: utf-8 -*-
"""
Main app loop for the poker trainer.

Запуск из корня проекта:
    python -m src.app
"""

# --- Force Qt platform plugins path on Windows (PyQt5) ---
import os, sys
from pathlib import Path

def _ensure_qt_plugins():
    if sys.platform != "win32":
        return
    root = Path(sys.prefix) / "Lib" / "site-packages" / "PyQt5"
    # Возможные раскладки: PyQt5/Qt5/... или PyQt5/Qt/...
    candidates = [
        root / "Qt5" / "plugins" / "platforms",
        root / "Qt"  / "plugins" / "platforms",
    ]
    for plat in candidates:
        if plat.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plat)
            # Иногда ещё нужны dll из bin — добавим
            for bindir in [root / "Qt5" / "bin", root / "Qt" / "bin"]:
                if bindir.exists():
                    try:
                        os.add_dll_directory(str(bindir))  # Py3.8+
                    except Exception:
                        os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
            break

_ensure_qt_plugins()

# --- Imports after Qt fix ---
import json
import time
import sqlite3
from datetime import datetime

from PyQt5 import QtWidgets

from .panel import Panel
from .state_from_screen import build_state
from .decide import lookup_decision


DB_PATH = "trainer_live.db"


def _init_db(path: str):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS logs(
               ts TEXT,
               state_json TEXT,
               model_action TEXT
           )"""
    )
    conn.commit()
    return conn


def _log(conn, state_obj, action: str):
    try:
        ts = datetime.now().isoformat(timespec="seconds")
        payload = json.dumps(state_obj, ensure_ascii=False)
        conn.execute("INSERT INTO logs(ts, state_json, model_action) VALUES(?,?,?)", (ts, payload, action))
        conn.commit()
    except Exception:
        # логирование — best-effort; не валим приложение
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = Panel()
    ui.show()

    conn = _init_db(DB_PATH)

    # cam не нужен: state_from_screen сам использует mss внутри
    cam = None

    try:
        while True:
            # 1) читаем состояние со стола
            s = build_state(cam)

            # ----- ВРЕМЕННЫЙ ХАРДКОД для теста твоей таблицы -----
            # пока авто-детекта спота нет — фиксируем вручную
            s["spot_key"] = os.environ.get("OVERRIDE_SPOT", "HU_BB_vs_LIMP")
            s["mode"] = "HU"
            s["hero_pos"] = "BB"
            # ------------------------------------------------------

            # 2) показываем сырое состояние в панели
            ui.show_state(
                json.dumps(
                    {
                        "hand": s.get("hero_hand"),
                        "bb": s.get("stacks_bb", {}).get("hero"),
                        "pos": s.get("hero_pos"),
                    },
                    ensure_ascii=False,
                )
            )

            # 3) решаем действие по базе таблиц
            try:
                action = lookup_decision(s)
            except Exception as e:
                action = f"ERROR: {type(e).__name__}: {e}"

            # 4) показываем ответ и логируем
            ui.show_answer(action)
            _log(conn, s, action)

            # 5) даём Qt обработать события + лёгкий троттлинг
            app.processEvents()
            time.sleep(0.7)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
