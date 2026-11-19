# -*- coding: utf-8 -*-
"""Рисунок 3: 3D-поверхность функции Розенброка и траектория метода."""
import numpy as np
from common import plt
from penalty import load_results, f

R = load_results()
path = R["path"]; x = R["x"]
ZMAX = 300   # поверхность обрезается сверху для наглядности

gx = np.linspace(-1.6, 1.6, 200); gy = np.linspace(-0.6, 2.6, 200)
X, Y = np.meshgrid(gx, gy)
Z = np.minimum((1 - X) ** 2 + 100 * (Y - X ** 2) ** 2, ZMAX)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True, rstride=2, cstride=2)
zp = np.array([min(f(p), ZMAX) for p in path]) + 12
ax.plot(path[:, 0], path[:, 1], zp, "r.-", lw=2.2, ms=5, zorder=20, label="траектория метода штрафов")
xx = np.linspace(-1.6, 1.6, 50)
ax.plot(xx, 1 - xx, np.zeros_like(xx), "k--", lw=1, label=r"$x_1+x_2=1$ (проекция)")
ax.scatter(*x, f(x) + 12, marker="*", s=200, c="w", edgecolors="k", zorder=10, label="финиш")
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$"); ax.set_zlabel(r"$f(x_1,x_2)$")
ax.set_zlim(0, ZMAX)
ax.set_title("Функция Розенброка и траектория метода штрафов")
ax.view_init(elev=40, azim=-75)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig("fig3_surface.png", dpi=150)
print("Сохранено: fig3_surface.png")
