"""Рисунок 3: 3D-поверхность функции Химмельблау и траектория спуска.
Запуск: python fig3_surface.py   (рядом должен лежать himmelblau_gd.py)
Результат: fig3_surface.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import J, gradient_descent

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (0.0, 0.0)   # начальная точка
H = 0.01          # шаг

x_fin, n_it, hist = gradient_descent(X0, h=H)
Jh = np.array([J(p) for p in hist])

gg = np.linspace(-5, 5, 200)
S1, S2 = np.meshgrid(gg, gg)
SZ = J((S1, S2))

fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(S1, S2, SZ, cmap="viridis", alpha=0.75, linewidth=0,
                antialiased=True, rcount=120, ccount=120)

# траекторию слегка приподнимаем над поверхностью (+15), чтобы она не пряталась
ax.plot(hist[:, 0], hist[:, 1], Jh + 15, "-o", color="red", lw=2.5, ms=3,
        label="траектория градиентного спуска", zorder=10)
ax.scatter([0], [0], [J(np.array(X0)) + 15], s=90, c="white", edgecolors="black",
           depthshade=False, label="старт", zorder=11)
ax.scatter([3], [2], [15], s=160, c="white", edgecolors="black", marker="*",
           depthshade=False, label="финиш", zorder=11)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$J(x_1, x_2)$")
ax.set_title("Функция Химмельблау и траектория градиентного спуска")
ax.view_init(elev=32, azim=-62)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("grafiki/fig3_surface.png", dpi=150)
print("сохранено: fig3_surface.png")
