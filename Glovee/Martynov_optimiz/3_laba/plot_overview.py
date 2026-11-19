# -*- coding: utf-8 -*-
"""Рисунок 1: линии уровня, допустимая область и траектория метода штрафов."""
import numpy as np
from matplotlib.colors import LogNorm
from common import plt, rosenbrock_grid, draw_region, log_colorbar
from penalty import load_results

R = load_results()
path = R["path"]; x0 = R["x0"]; x = R["x"]
pts = np.array([h["path"][-1] for h in R["hist"]])

X, Y, Z = rosenbrock_grid((-2, 2), (-1, 3))
fig, ax = plt.subplots(figsize=(9, 7))
cs = ax.contourf(X, Y, Z + 1e-6, levels=np.logspace(-3, 3.8, 50), norm=LogNorm(), cmap="viridis", extend="min")
log_colorbar(fig, ax, cs)
draw_region(ax, (-2, 2), (-1, 3))
ax.plot(path[:, 0], path[:, 1], "r.-", lw=1, ms=3, label="траектория метода штрафов")
ax.plot(pts[:, 0], pts[:, 1], "wo", mec="k", ms=5, label=r"решения внешних итераций ($r=1,10,\dots$)")
ax.plot(*x0, "o", mfc="w", mec="k", ms=11, mew=1.8, label="старт")
ax.plot(*x, "*", mfc="w", mec="k", ms=18, mew=1.5, label="финиш")
ax.plot(1, 1, "kx", ms=9, mew=2, label=r"безусловный минимум $(1;1)$")
ax.set_xlim(-2, 2); ax.set_ylim(-1, 3)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
ax.set_title("Функция Розенброка, допустимая область (серым – недопустимая) и траектория")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
fig.tight_layout(); fig.savefig("fig1_overview.png", dpi=150)
print("Сохранено: fig1_overview.png")
