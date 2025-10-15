import pdfplumber, pandas as pd, re, sys

pdf_path = r".\data\vs_fish_final_19.03 (2).pdf"
rows = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for t in tables:
            # здесь под конкретный формат твоего PDF:
            # ожидаем колонки: Hand, 8-10BB, 10-12BB, 12-15BB, ...
            header = [c.strip() if c else "" for c in t[0]]
            if "Hand" not in header: continue
            for row in t[1:]:
                hand = row[0].strip()
                for i, col in enumerate(header[1:], start=1):
                    cell = (row[i] or "").strip().lower()
                    if not cell: continue
                    action = "push" if "push" in cell or "all-in" in cell else "fold"
                    # вытащим диапазон BB из названия колонки
                    m = re.search(r"(\d+)[—\-](\d+)\s*bb", col.lower())
                    if not m: continue
                    stack_min, stack_max = int(m.group(1)), int(m.group(2))
                    rows.append(dict(mode="3max", pos_hero="BB", hand=hand,
                                     action=action, stack_min=stack_min, stack_max=stack_max))
df = pd.DataFrame(rows)
df.to_csv(r".\data\charts\preflop.csv", index=False)
print("Saved rows:", len(df))
