"""Таблица 1: ход итераций градиентного спуска (h = 0,01).
Запуск: python table1_iterations.py   (рядом должен лежать himmelblau_gd.py)
Результат: вывод в консоль и файл table1_iterations.csv (открывается в Excel)
"""
import csv
import numpy as np

from main import J, grad_J, gradient_descent

X0 = (0.0, 0.0)   # начальная точка
H = 0.01          # шаг

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def ru(x, d=4):
    """Число с десятичной запятой и настоящим минусом."""
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def sci(x, d=2):
    """Число вида 1,03·10⁻³."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** e:.{d}f}".replace(".", ",") + "·10" + str(e).translate(SUP)


def fmt(x):
    return ru(x, 4) if abs(x) >= 0.01 else sci(x)


x_fin, n_it, hist = gradient_descent(X0, h=H)

header = ["k", "x1", "x2", "J(x^k)", "||grad J(x^k)||"]
rows = []
for k in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, n_it]:   # выбранные итерации
    p = hist[k]
    rows.append([k, ru(p[0], 6), ru(p[1], 6), fmt(J(p)), fmt(np.linalg.norm(grad_J(p)))])

# csv (utf-8-sig, чтобы Excel не ломал кириллицу и спецсимволы)
with open("table1_iterations.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(rows)

# вывод в консоль
widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
print("\nсохранено: table1_iterations.csv")
