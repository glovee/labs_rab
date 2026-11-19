# -*- coding: utf-8 -*-
"""Таблица 1 отчёта: результаты внешних итераций.
Создаёт table1.csv (для Excel) и table1.md и выводит таблицу на экран."""
import csv
from penalty import load_results

R = load_results()
cols = ["k", "r", "x1", "x2", "f(x)", "max(0,g1)", "F(x,r)", "итер. BFGS"]
rows = [[h["k"], f"{h['r']:.0e}", f"{h['x1']:.6f}", f"{h['x2']:.6f}", f"{h['f']:.6f}",
         f"{h['viol']:.2e}", f"{h['F']:.6f}", h["nit"]] for h in R["hist"]]

with open("table1.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh, delimiter=";"); w.writerow(cols); w.writerows(rows)

md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
open("table1.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
print("Сохранено: table1.csv, table1.md")
