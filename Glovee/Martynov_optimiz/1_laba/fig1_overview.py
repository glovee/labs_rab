"""Рисунок 1: функция Химмельблау (карта значений) и траектория градиентного спуска.
Запуск: python fig1_overview.py   (рядом должен лежать himmelblau_gd.py)
Результат: fig1_overview.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

from main import J, gradient_descent

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (0.0, 0.0)   # начальная точка
H = 0.01          # шаг

# остальные три минимума (минимум (3; 2) - финиш траектории)
OTHER_MINIMA = np.array([
    [-2.805118086953, 3.131312518250],
    [-3.779310253378, -3.283185991286],
    [3.584428340330, -1.848126526964],
])

x_fin, n_it, hist = gradient_descent(X0, h=H)

L = 5.1
g = np.linspace(-L, L, 700)
X1, X2 = np.meshgrid(g, g)
Z = J((X1, X2))

fig, ax = plt.subplots(figsize=(9, 7))
norm = colors.PowerNorm(gamma=0.35, vmin=0, vmax=Z.max())
im = ax.imshow(Z, extent=[-L, L, -L, L], origin="lower", cmap="viridis", norm=norm)
cb = fig.colorbar(im, ax=ax)
cb.set_ticks([0, 5, 20, 50, 100, 200, 400, 800])
cb.set_label(r"$J(x_1, x_2)$")

ax.plot(hist[:, 0], hist[:, 1], "-o", color="red", lw=2, ms=3,
        label="траектория градиентного спуска", zorder=5)
ax.scatter(*X0, s=260, c="white", edgecolors="black", linewidths=2,
           zorder=6, label="старт (0; 0)")
ax.scatter(*x_fin, s=420, c="white", edgecolors="black", linewidths=2,
           marker="*", zorder=6, label="финиш (3; 2)")
ax.scatter(OTHER_MINIMA[:, 0], OTHER_MINIMA[:, 1], s=140, c="white", edgecolors="black",
           marker="X", linewidths=1.5, zorder=6, label="остальные минимумы")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Функция Химмельблау и траектория градиентного спуска")
ax.legend(loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig("grafiki/fig1_overview.png", dpi=150)
print("сохранено: fig1_overview.png")
