"""Рисунок 1: функция Растригина (карта значений) и траектория случайного поиска.
Запуск: python fig1_contour.py   (рядом должен лежать rastrigin_rs.py)
Результат: fig1_contour.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rastrigin_rs import J, random_search

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (2.5, 2.5)   # начальная точка
R0 = 2.0          # начальный радиус поиска
M = 20            # число неудач подряд до уменьшения радиуса
SEED = 0          # зерно датчика случайных чисел (для воспроизводимости)

res = random_search(x0=X0, R0=R0, M=M, q=0.5, golden=True, rng=np.random.default_rng(SEED))
path = res["path"]

L = 3.3
g = np.linspace(-L, L, 800)
X1, X2 = np.meshgrid(g, g)
Z = J((X1, X2))

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(Z, extent=[-L, L, -L, L], origin="lower", cmap="viridis")
cb = fig.colorbar(im, ax=ax)
cb.set_label(r"$J(x_1, x_2)$")

ax.plot(path[:, 0], path[:, 1], "-o", color="red", lw=1.6, ms=4,
        label="траектория случайного поиска (принятые шаги)", zorder=5)
ax.scatter(*X0, s=260, c="white", edgecolors="black", linewidths=2,
           zorder=6, label="старт (2,5; 2,5)")
ax.scatter(*res["x"], s=420, c="white", edgecolors="black", linewidths=2,
           marker="*", zorder=6, label="финиш (0; 0)")

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Функция Растригина и траектория случайного поиска")
ax.legend(loc="upper left", framealpha=0.92)
fig.tight_layout()
fig.savefig("fig1_contour.png", dpi=150)
print(f"сохранено: fig1_contour.png  (принятых шагов: {len(path) - 1}, финиш: {res['x']})")
