"""Рисунок 4: значение функции и радиус поиска по итерациям.
Запуск: python fig4_convergence.py   (рядом должен лежать rastrigin_rs.py)
Результат: fig4_convergence.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rastrigin_rs import random_search

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (2.5, 2.5)   # начальная точка
R0 = 2.0          # начальный радиус поиска
M = 20            # число неудач подряд до уменьшения радиуса
SEED = 0          # зерно датчика случайных чисел
FLOOR = 1e-16     # J около нуля считается с погрешностью округления ~1e-15

res = random_search(x0=X0, R0=R0, M=M, q=0.5, golden=True, rng=np.random.default_rng(SEED))
ks = np.arange(len(res["f_hist"]))
Jv = np.maximum(res["f_hist"], FLOOR)
Rv = res["R_hist"]
acc = np.array([row[0] for row in res["log"]])      # номера итераций с принятым шагом

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))

axs[0].semilogy(ks, Jv, "-", color="tab:blue", lw=1.8)
axs[0].plot(acc, Jv[acc], "o", ms=3.5, color="tab:red", label="принятые шаги")
axs[0].set_xlabel("номер итерации $k$")
axs[0].set_ylabel(r"$J(x^k)$")
axs[0].set_title("Лучшее найденное значение функции")
axs[0].legend()
axs[0].grid(True, which="both", alpha=0.3)

axs[1].semilogy(ks, Rv, "-", color="tab:green", lw=1.8)
axs[1].set_xlabel("номер итерации $k$")
axs[1].set_ylabel(r"радиус поиска $R$")
axs[1].set_title("Радиус поиска (уменьшается после $M$ неудач подряд)")
axs[1].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("fig4_convergence.png", dpi=150)
print(f"сохранено: fig4_convergence.png  (итераций: {res['n_iter']}, принятых шагов: {len(acc) - 1})")
