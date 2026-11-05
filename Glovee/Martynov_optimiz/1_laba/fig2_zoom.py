"""Рисунок 2: приближенный вид - линии уровня и номера итераций.
Запуск: python fig2_zoom.py   (рядом должен лежать himmelblau_gd.py)
Результат: fig2_zoom.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import colors

from main import J, gradient_descent

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (0.0, 0.0)   # начальная точка
H = 0.01          # шаг

x_fin, n_it, hist = gradient_descent(X0, h=H)

# область просмотра
x1l, x1r, x2l, x2r = -0.6, 3.6, -0.6, 2.9
Y1, Y2 = np.meshgrid(np.linspace(x1l, x1r, 700), np.linspace(x2l, x2r, 700))
Zz = J((Y1, Y2))

fig, ax = plt.subplots(figsize=(9, 7))
lvl = np.logspace(-3, np.log10(Zz.max()), 40)
cf = ax.contourf(Y1, Y2, np.maximum(Zz, 1e-3), levels=lvl, cmap="plasma",
                 norm=colors.LogNorm(vmin=1e-3, vmax=Zz.max()))
ax.contour(Y1, Y2, Zz, levels=np.logspace(-1, np.log10(Zz.max()), 18),
           colors="k", linewidths=0.4, alpha=0.6)
cb = fig.colorbar(cf, ax=ax)
cb.set_ticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
cb.set_ticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$", "$10^{2}$"])
cb.set_label(r"$J(x_1, x_2)$ (лог. шкала)")

ax.plot(hist[:, 0], hist[:, 1], "-o", color="red", lw=2, ms=3,
        label="траектория градиентного спуска", zorder=5)
ax.scatter(*X0, s=260, c="white", edgecolors="black", linewidths=2,
           zorder=6, label="старт (0; 0)")
ax.scatter(*x_fin, s=420, c="white", edgecolors="black", linewidths=2,
           marker="*", zorder=6, label="финиш (3; 2)")

# номера итераций
for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]:
    ax.annotate(str(k), hist[k], textcoords="offset points", xytext=(7, 5),
                fontsize=10, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")], zorder=7)

ax.set_xlim(x1l, x1r)
ax.set_ylim(x2l, x2r)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Приближенный вид: линии уровня и номера итераций")
ax.legend(loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig("grafiki/fig2_zoom.png", dpi=150)
print("сохранено: fig2_zoom.png")
