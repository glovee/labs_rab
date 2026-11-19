# -*- coding: utf-8 -*-
"""Рисунок 2: приближенный вид вблизи решения."""
import numpy as np
from matplotlib.colors import LogNorm
from common import plt, rosenbrock_grid, draw_region, log_colorbar
from penalty import load_results

R = load_results()
path = R["path"]; xs = R["xs"]; hist = R["hist"]
pts = np.array([h["path"][-1] for h in hist])

xl, yl = (0.35, 0.95), (0.05, 0.65)
X, Y, Z = rosenbrock_grid(xl, yl)
fig, ax = plt.subplots(figsize=(9, 7))
cs = ax.contourf(X, Y, Z + 1e-6, levels=np.logspace(-3, 1.5, 45), norm=LogNorm(), cmap="plasma", extend="both")
ax.contour(X, Y, Z + 1e-6, levels=np.logspace(-3, 1.5, 19), colors="k", linewidths=0.3)
log_colorbar(fig, ax, cs)
draw_region(ax, xl, yl)
ax.plot(path[:, 0], path[:, 1], "c.-", lw=1, ms=3, label="траектория метода штрафов")
ax.plot(pts[:, 0], pts[:, 1], "o", mfc="w", mec="k", ms=6, label="решения внешних итераций")
for h in hist[:2]:
    ax.annotate(f"r=1e{int(round(np.log10(h['r'])))}", (h["x1"], h["x2"]), fontsize=9,
                xytext=(7, 4), textcoords="offset points")
ax.plot(*xs, "*", mfc="w", mec="k", ms=18, mew=1.5, label="финиш")
ax.plot(1, 1, "kx", ms=9, mew=2, label=r"безусловный минимум $(1;1)$")
ax.set_xlim(*xl); ax.set_ylim(*yl)
ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
ax.set_title("Приближенный вид: подход к границе $x_1+x_2=1$ при росте $r$")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
fig.tight_layout(); fig.savefig("fig2_zoom.png", dpi=150)
print("Сохранено: fig2_zoom.png")
