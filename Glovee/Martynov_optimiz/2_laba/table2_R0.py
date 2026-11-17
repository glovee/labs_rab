"""Таблица 2: влияние начального радиуса поиска R0 (q = 1/2), 200 запусков в каждой строке.
Сравниваются золотое сечение (M = 20) и обычный поиск с фиксированным шагом (M = 300).
Успех - найденная точка ближе 1e-3 к глобальному минимуму (0; 0).
Запуск: python table2_R0.py   (рядом должен лежать rastrigin_rs.py)
Результат: вывод в консоль и файл table2_R0.csv (открывается в Excel)
"""
import csv

from rastrigin_rs import run_statistics

N_RUNS = 200
R0_LIST = [0.5, 1.0, 2.0, 3.0, 4.0]

header = ["R0",
          "золотое сечение (M=20): доля успеха, %", "золотое сечение: медиана вычислений J",
          "обычный поиск (M=300): доля успеха, %", "обычный поиск: медиана вычислений J"]
rows = []
for R0 in R0_LIST:
    p_g, ev_g, _ = run_statistics(N_RUNS, R0=R0, M=20, q=0.5, golden=True)
    p_p, ev_p, _ = run_statistics(N_RUNS, R0=R0, M=300, q=0.5, golden=False)
    rows.append([f"{R0:g}".replace(".", ","),
                 f"{100 * p_g:g}".replace(".", ","), int(ev_g),
                 f"{100 * p_p:g}".replace(".", ","), int(ev_p)])

with open("table2_R0.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(rows)

widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
for r in [header] + rows:
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
print("\nсохранено: table2_R0.csv")
