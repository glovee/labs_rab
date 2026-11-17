"""Рисунок 2: 3D-поверхность функции Растригина и траектория случайного поиска.
Запуск: python fig2_surface.py   (рядом должен лежать rastrigin_rs.py)
Результат: fig2_surface.png
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
SEED = 0          # зерно датчика случайных чисел

res = random_search(x0=X0, R0=R0, M=M, q=0.5, golden=True, rng=np.random.default_rng(SEED))
path = res["path"]
Jp = np.array([J(p) for p in path])

L = 3.3
g = np.linspace(-L, L, 260)
S1, S2 = np.meshgrid(g, g)
SZ = J((S1, S2))

fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(S1, S2, SZ, cmap="viridis", alpha=0.75, linewidth=0,
                antialiased=True, rcount=140, ccount=140)

# траекторию слегка приподнимаем над поверхностью (+3), чтобы она не пряталась
ax.plot(path[:, 0], path[:, 1], Jp + 3, "-o", color="red", lw=2, ms=3.5,
        label="траектория случайного поиска", zorder=10)
ax.scatter([X0[0]], [X0[1]], [J(np.array(X0)) + 3], s=90, c="white", edgecolors="black",
           depthshade=False, label="старт", zorder=11)
ax.scatter([0], [0], [3], s=170, c="white", edgecolors="black", marker="*",
           depthshade=False, label="финиш", zorder=11)

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$J(x_1, x_2)$")
ax.set_title("Функция Растригина и траектория случайного поиска")
ax.view_init(elev=34, azim=-58)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("fig2_surface.png", dpi=150)
print("сохранено: fig2_surface.png")
