"""Таблица 3: влияние числа неудач M до уменьшения радиуса (R0 = 2, q = 1/2), 200 запусков в строке.
Успех - найденная точка ближе 1e-3 к глобальному минимуму (0; 0).
Запуск: python table3_M.py   (рядом должен лежать rastrigin_rs.py)
Результат: вывод в консоль и файл table3_M.csv (открывается в Excel)
"""
import csv

from rastrigin_rs import run_statistics

N_RUNS = 200
R0 = 2.0
CASES = [("золотое сечение", True, [5, 10, 20, 40, 80]),
         ("обычный поиск", False, [20, 100, 300, 600])]

header = ["метод", "M", "доля успеха, %", "медиана вычислений J", "среднее найденное J"]
rows = []
for name, golden, Ms in CASES:
    for M in Ms:
        p, ev, fmean = run_statistics(N_RUNS, R0=R0, M=M, q=0.5, golden=golden)
        rows.append([name, M, f"{100 * p:g}".replace(".", ","), int(ev),
                     f"{fmean:.3f}".replace(".", ",")])

with open("table3_M.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(rows)

widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
print("\nсохранено: table3_M.csv")
