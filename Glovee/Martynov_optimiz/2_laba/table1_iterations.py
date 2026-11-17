"""Таблица 1: ход случайного поиска с золотым сечением (R0 = 2, M = 20, зерно 0).
Показаны выбранные принятые шаги (в таблице не все 44).
Запуск: python table1_iterations.py   (рядом должен лежать rastrigin_rs.py)
Результат: вывод в консоль и файл table1_iterations.csv (открывается в Excel)
"""
import csv
import numpy as np

from rastrigin_rs import random_search

X0 = (2.5, 2.5)   # начальная точка
R0 = 2.0          # начальный радиус поиска
M = 20            # число неудач подряд до уменьшения радиуса
SEED = 0          # зерно датчика случайных чисел

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def ru(x, d=4):
    """Число с десятичной запятой и настоящим минусом."""
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def sci(x, d=2):
    """Число вида −1,66·10⁻⁹."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    sign = "−" if x < 0 else ""
    return sign + f"{abs(x) / 10 ** e:.{d}f}".replace(".", ",") + "·10" + str(e).translate(SUP)


def fmt_x(x):
    return ru(x, 4) if abs(x) >= 1e-3 else sci(x)


def fmt_f(f):
    return ru(f, 4) if f >= 0.01 else sci(f)


res = random_search(x0=X0, R0=R0, M=M, q=0.5, golden=True, rng=np.random.default_rng(SEED))
log = res["log"]                                  # (итерация, x1, x2, J, R, вычислений J)
STEPS = [0, 1, 2, 3, 4, 5, 9, 14, 15, 19, 21, 22, 25, 31, 38, len(log) - 1]   # номера принятых шагов

header = ["шаг №", "итерация k", "x1", "x2", "J", "R", "вычислений J"]
rows = []
for s in STEPS:
    k, x1, x2, f, R, ne = log[s]
    rows.append([s, k, fmt_x(x1), fmt_x(x2), fmt_f(f), f"{R:g}".replace(".", ","), ne])

with open("table1_iterations.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(rows)

widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
print(f"\nвсего принятых шагов: {len(log) - 1}, итераций: {res['n_iter']}, вычислений J: {res['n_eval']}")
print("сохранено: table1_iterations.csv")
