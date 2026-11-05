"""Таблица 2: зависимость результата от шага h (максимум 200000 итераций на каждый шаг).
Заодно считает теоретический порог сходимости по собственным числам матрицы вторых
производных в минимуме (3; 2): H = [[74, 20], [20, 34]].
Запуск: python table2_step.py   (рядом должен лежать himmelblau_gd.py)
Результат: вывод в консоль и файл table2_step.csv (открывается в Excel)
"""
import csv
import numpy as np

from main import grad_J, gradient_descent

X0 = (0.0, 0.0)          # начальная точка
MAX_ITER = 200000        # ограничение числа итераций
STEPS = [0.001, 0.005, 0.01, 0.0185, 0.02, 0.024, 0.025, 0.03, 0.04]

header = ["h", "число итераций", "результат"]
rows = []
for h in STEPS:
    with np.errstate(all="ignore"):
        x, k, _ = gradient_descent(X0, h=h, max_iter=MAX_ITER)
    h_str = str(h).replace(".", ",")
    if np.all(np.isfinite(x)) and np.linalg.norm(grad_J(x)) < 1e-8:
        rows.append([h_str, k, "сходится к (3; 2)"])
    elif np.all(np.isfinite(x)):
        rows.append([h_str, "–", "не сходится (колеблется)"])
    else:
        rows.append([h_str, "–", "расходится"])

with open("table2_step.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(rows)

widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))

# теоретический порог сходимости
lam = np.linalg.eigvalsh(np.array([[74.0, 20.0], [20.0, 34.0]]))
print(f"\nλmin = {lam[0]:.4f}, λmax = {lam[1]:.4f}")
print(f"порог сходимости h < 2/λmax = {2 / lam[1]:.4f}")
print(f"оптимальный шаг 2/(λmin+λmax) = {2 / lam.sum():.4f}")
print(f"множитель убывания при h = 0,01: 1 - h·λmin = {1 - 0.01 * lam[0]:.4f}")
print("\nсохранено: table2_step.csv")
