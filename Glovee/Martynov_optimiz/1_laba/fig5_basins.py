"""Рисунок 5: области притяжения четырёх минимумов при h = 0,01.
Из каждой точки сетки на квадрате [-5; 5] x [-5; 5] запускается градиентный спуск
(все точки обрабатываются одновременно, векторно); цветом показано, в какой минимум он пришёл.
Запуск: python fig5_basins.py   (рядом должен лежать himmelblau_gd.py)
Результат: fig5_basins.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (0.0, 0.0)   # начальная точка из задачи (только для отметки на рисунке)
H = 0.01          # шаг
N = 360           # сетка N x N
N_ITER = 4000     # число итераций для каждой точки

MINIMA = np.array([
    [3.0, 2.0],
    [-2.805118086953, 3.131312518250],
    [-3.779310253378, -3.283185991286],
    [3.584428340330, -1.848126526964],
])

gb = np.linspace(-5, 5, N)
B1, B2 = np.meshgrid(gb, gb)
P = np.stack([B1.ravel(), B2.ravel()], axis=1)   # все стартовые точки сразу

with np.errstate(all="ignore"):
    for _ in range(N_ITER):
        a = P[:, 0] ** 2 + P[:, 1] - 11
        b = P[:, 0] + P[:, 1] ** 2 - 7
        gx = 4 * P[:, 0] * a + 2 * b
        gy = 2 * a + 4 * P[:, 1] * b
        P = P - H * np.stack([gx, gy], axis=1)
        P = np.where(np.abs(P) > 1e6, np.nan, P)   # отсекаем расходящиеся траектории

lab = np.full(len(P), -1)                          # -1: не сошлось ни к одному минимуму
for i, m in enumerate(MINIMA):
    lab[np.linalg.norm(P - m, axis=1) < 1e-3] = i
lab = lab.reshape(N, N)

print("доли областей притяжения:", [round(float((lab == i).mean()), 3) for i in range(4)],
      "| не сошлись:", round(float((lab == -1).mean()), 3))

cmap = colors.ListedColormap(["#bdbdbd", "#4c78a8", "#f58518", "#54a24b", "#e45756"])
fig, ax = plt.subplots(figsize=(8.5, 7))
ax.imshow(lab, extent=[-5, 5, -5, 5], origin="lower", cmap=cmap, vmin=-1, vmax=3,
          interpolation="nearest")
ax.scatter(MINIMA[:, 0], MINIMA[:, 1], s=170, c="white", edgecolors="black",
           marker="*", linewidths=1.5, zorder=5)
ax.scatter(*X0, s=120, c="white", edgecolors="black", linewidths=2, zorder=6)
ax.annotate("старт (0; 0)", X0, textcoords="offset points", xytext=(8, -16), fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

names = ["(3; 2)", "(−2,81; 3,13)", "(−3,78; −3,28)", "(3,58; −1,85)"]
ax.legend(handles=[Patch(color=cmap(i + 1), label=n) for i, n in enumerate(names)],
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, framealpha=0.9,
          title="минимум, к которому приходит спуск")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title(r"Области притяжения минимумов при $h = 0{,}01$")
fig.tight_layout()
fig.savefig("grafiki/fig5_basins.png", dpi=150)
print("сохранено: fig5_basins.png")
