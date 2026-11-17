"""Рисунок 3: одна итерация - значение J вдоль случайного луча и работа золотого сечения.
Берётся первый принятый шаг траектории из рисунка 1 (то же зерно датчика).
Запуск: python fig3_line_search.py   (рядом должен лежать rastrigin_rs.py)
Результат: fig3_line_search.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rastrigin_rs import J, random_search, golden_section

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (2.5, 2.5)   # начальная точка
R0 = 2.0          # начальный радиус поиска
M = 20            # число неудач подряд до уменьшения радиуса
SEED = 0          # зерно датчика случайных чисел
TOL = 1e-8        # точность золотого сечения

res = random_search(x0=X0, R0=R0, M=M, q=0.5, tol=TOL, golden=True,
                    rng=np.random.default_rng(SEED))
x_old, x_new = res["path"][0], res["path"][1]       # первый принятый шаг
R = res["log"][1][4]                                # радиус поиска на этой итерации
step = x_new - x_old
t_found = np.linalg.norm(step)
d = step / t_found                                  # направление, которое выпало датчику

phi = lambda t: J((x_old[0] + t * d[0], x_old[1] + t * d[1]))   # работает и для числа, и для массива t
f0 = phi(0.0)

# повторяем золотое сечение, запоминая все точки, где считалась функция
probes = []


def phi_rec(t):
    probes.append(t)
    return phi(t)


t_star, n_phi = golden_section(phi_rec, 0.0, R, TOL)
print(f"направление d = ({d[0]:.4f}; {d[1]:.4f}), R = {R}, t* = {t_star:.6f}, "
      f"J(x) = {f0:.4f} -> {phi(t_star):.4f}, вычислений: {n_phi}")

tt = np.linspace(0.0, R, 2000)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (lo, hi), title in zip(
        axs, [(0.0, R), (t_star - 0.08, t_star + 0.08)],
        ["Весь отрезок [0; R]", "Увеличенный фрагмент около найденного шага"]):
    m = (tt >= lo) & (tt <= hi)
    ax.plot(tt[m], phi(tt[m]), color="tab:blue", lw=2, label=r"$\varphi(t) = J(x^k + t\,d)$")
    ax.axhline(f0, color="gray", ls="--", lw=1.2, label=r"$J(x^k)$ (текущее значение)")
    pr = np.array([p for p in probes if lo <= p <= hi])
    ax.scatter(pr, phi(pr), s=28, c="tab:orange", zorder=4, label="точки золотого сечения")
    ax.scatter([t_star], [phi(t_star)], s=260, c="red", marker="*", zorder=5,
               edgecolors="black", label=r"найденный шаг $t^*$")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\varphi(t)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
axs[0].legend(loc="upper right", fontsize=9)

# первые вычисления функции: порядок видно по цифрам
for i, p in enumerate(probes[:6], start=1):
    axs[0].annotate(str(i), (p, phi(p)), textcoords="offset points", xytext=(5, 6), fontsize=9)

fig.tight_layout()
fig.savefig("fig3_line_search.png", dpi=150)
print("сохранено: fig3_line_search.png")
