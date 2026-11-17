"""Таблица 4: обычный случайный поиск (фиксированный шаг, M = 300), 200 запусков в строке.
Проверяется, важно ли, чтобы среди радиусов R0, q R0, q^2 R0, ... встретилось значение около 1
(расстояние между соседними минимумами функции Растригина равно 1).
Успех - найденная точка ближе 1e-3 к глобальному минимуму (0; 0).
Запуск: python table4_radius_sequence.py   (рядом должен лежать rastrigin_rs.py)
Результат: вывод в консоль и файл table4_radius_sequence.csv (открывается в Excel)
"""
import csv

from rastrigin_rs import run_statistics

N_RUNS = 200
M = 300
CASES = [  # (R0, q, текстовая запись q, последовательность радиусов)
    (2.0, 1 / 2, "1/2", "2; 1; 0,5; ..."),
    (3.0, 1 / 2, "1/2", "3; 1,5; 0,75; ..."),
    (3.0, 1 / 3, "1/3", "3; 1; 0,33; ..."),
    (1.5, 1 / 2, "1/2", "1,5; 0,75; 0,38; ..."),
    (1.5, 2 / 3, "2/3", "1,5; 1; 0,67; ..."),
]

header = ["R0", "q", "радиусы поиска", "доля успеха, %", "медиана вычислений J"]
rows = []
for R0, q, q_txt, seq in CASES:
    p, ev, _ = run_statistics(N_RUNS, R0=R0, M=M, q=q, golden=False)
    rows.append([f"{R0:g}".replace(".", ","), q_txt, seq, f"{100 * p:g}".replace(".", ","), int(ev)])

with open("table4_radius_sequence.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(rows)

widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
print("\nсохранено: table4_radius_sequence.csv")
