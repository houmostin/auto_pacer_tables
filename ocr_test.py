import time
import dxcam
import cv2
import numpy as np
import pytesseract

# === НАСТРОЙКИ ===
MONITOR = 0
# Стартовая область захвата (НЕ весь экран!), подгони под окно тренажёра.
# Формат: (x1, y1, x2, y2)
CAP_REGION = [400, 200, 1520, 900]

# ROI внутри CAP_REGION (где текст, который читаем). Подгони позже.
# Например, блок стека/BB:
ROI = [700, 600, 1000, 660]  # координаты в СИСТЕМНЫХ координатах экрана

LANG = "rus+eng"
CFG_TXT = r"--oem 1 --psm 7"  # одиночная строка
CFG_NUM = r"--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.,Bb"

def preprocess(img, for_digits=False):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Увеличим, чтобы Tesseract легче читал мелкий шрифт
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.bilateralFilter(g, 5, 60, 60)
    g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 5)
    # Иногда лучше инвертировать (белый текст на тёмном фоне)
    # g = cv2.bitwise_not(g)
    return g

def ocr_text(img, digits=False):
    cfg = CFG_NUM if digits else CFG_TXT
    return pytesseract.image_to_string(img, lang=LANG, config=cfg).strip()

def main():
    cam = dxcam.create(output_idx=MONITOR)
    print("[i] Q/ESC — выход, R — сменить ROI, C — сменить CAP_REGION, V — показать разовый предпросмотр.")
    last_out = ""

    while True:
        # Захватываем ТОЛЬКО область CAP_REGION — так окно превью не попадёт обратно в кадр
        frame = cam.grab(region=tuple(CAP_REGION))
        if frame is None:
            print("[warn] frame is None — жду поток...")
            time.sleep(0.05)
            continue

        # Вырезаем ROI по абсолютным координатам экрана:
        x1, y1, x2, y2 = ROI
        cx1, cy1, cx2, cy2 = CAP_REGION
        # Пересчёт ROI в локальные координаты кадра
        rx1, ry1 = x1 - cx1, y1 - cy1
        rx2, ry2 = x2 - cx1, y2 - cy1
        if rx1 < 0 or ry1 < 0 or rx2 > frame.shape[1] or ry2 > frame.shape[0]:
            print("[err] ROI выходит за границы CAP_REGION. Нажми R и введи корректные координаты.")
            time.sleep(0.3)
            continue

        roi_img = frame[ry1:ry2, rx1:rx2]
        if roi_img.size == 0:
            time.sleep(0.05)
            continue

        g = preprocess(roi_img)
        text = ocr_text(g, digits=False)  # для цифр поставь digits=True

        # Печатаем только изменения и не очень короткий мусор
        if text and (len(text) > 2) and (text != last_out):
            print("OCR:", repr(text))
            last_out = text

        # Обработка клавиш без имshow:
        k = cv2.waitKey(1) & 0xFF  # будет всегда -1, но оставим для ESC
        if k in (27, ord('q')):
            break
# пример вызова
text_stack = ocr_text(preprocess(roi_stack), digits=True)
text_hand  = ocr_text(preprocess(roi_hand),  digits=False)


        # Прочитаем ввод из консоли, если он есть (не блокируя основной цикл)
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getwch().lower()
            if ch == 'q' or ch == '\x1b':
                break
            elif ch == 'r':
                try:
                    print("Введите ROI (x1 y1 x2 y2) в системных координатах экрана:")
                    raw = input("ROI> ").strip()
                    nx1, ny1, nx2, ny2 = map(int, raw.split())
                    ROI[:] = [nx1, ny1, nx2, ny2]
                    print("Новый ROI:", ROI)
                except Exception as e:
                    print("Неверный ввод:", e)
            elif ch == 'c':
                try:
                    print("Введите CAP_REGION (x1 y1 x2 y2) — прямоугольник захвата:")
                    raw = input("CAP> ").strip()
                    nx1, ny1, nx2, ny2 = map(int, raw.split())
                    CAP_REGION[:] = [nx1, ny1, nx2, ny2]
                    print("Новый CAP_REGION:", CAP_REGION)
                except Exception as e:
                    print("Неверный ввод:", e)
            elif ch == 'v':
                # Разовый предпросмотр (окно появится, но нужный прямоугольник захвата его не увидит)
                preview = frame.copy()
                cv2.rectangle(preview,
                              (rx1, ry1), (rx2, ry2),
                              (0, 255, 0), 2)
                cv2.imshow("Preview (закрой окно)", preview)
                cv2.waitKey(0)
                cv2.destroyWindow("Preview")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
