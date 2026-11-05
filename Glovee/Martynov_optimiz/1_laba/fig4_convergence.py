"""Рисунок 4: значение функции и норма градиента по итерациям (лог. масштаб).
Запуск: python fig4_convergence.py   (рядом должен лежать himmelblau_gd.py)
Результат: fig4_convergence.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import J, grad_J, gradient_descent

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (0.0, 0.0)   # начальная точка
H = 0.01          # шаг
EPS = 1e-8        # порог остановки по норме градиента

x_fin, n_it, hist = gradient_descent(X0, h=H, eps=EPS)

ks = np.arange(len(hist))
# 1e-20: чтобы логарифмическая шкала не ломалась на нулевых значениях
Jv = np.maximum([J(p) for p in hist], 1e-20)
Gv = np.maximum([np.linalg.norm(grad_J(p)) for p in hist], 1e-20)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))

axs[0].semilogy(ks, Jv, "-o", ms=3, color="tab:blue")
axs[0].set_xlabel("номер итерации $k$")
axs[0].set_ylabel(r"$J(x^k)$")
axs[0].set_title("Значение функции")
axs[0].grid(True, which="both", alpha=0.3)

axs[1].semilogy(ks, Gv, "-o", ms=3, color="tab:red")
axs[1].axhline(EPS, color="k", ls="--", lw=1, label=r"порог остановки $\varepsilon = 10^{-8}$")
axs[1].set_xlabel("номер итерации $k$")
axs[1].set_ylabel(r"$\|\nabla J(x^k)\|$")
axs[1].set_title("Норма градиента")
axs[1].legend()
axs[1].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("grafiki/fig4_convergence.png", dpi=150)
print("сохранено: fig4_convergence.png")
