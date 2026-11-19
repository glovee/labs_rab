# -*- coding: utf-8 -*-
"""Рисунок 4: сходимость метода штрафов при росте r."""
import numpy as np
from common import plt
from penalty import load_results

R = load_results()
hist = R["hist"]; xs = np.array(R["xs"]); fs = R["fs"]
rs = [h["r"] for h in hist]

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].loglog(rs, [max(h["viol"], 1e-16) for h in hist], "o-", label=r"$\max(0,g_i)$")
axs[0].loglog(rs, [max(np.linalg.norm([h["x1"] - xs[0], h["x2"] - xs[1]]), 1e-16) for h in hist],
              "s-", label=r"$\|x_k-x^*\|$")
axs[0].set_xlabel("r"); axs[0].set_title("Нарушение ограничения и ошибка")
axs[0].grid(True, which="both", alpha=.3); axs[0].legend()

axs[1].semilogx(rs, [h["f"] for h in hist], "o-", label=r"$f(x_k)$")
axs[1].axhline(fs, color="r", ls="--", label=r"$f(x^*)$")
axs[1].set_xlabel("r"); axs[1].set_title("Значение целевой функции")
axs[1].grid(True, alpha=.3); axs[1].legend()
fig.tight_layout(); fig.savefig("fig4_convergence.png", dpi=150)
print("Сохранено: fig4_convergence.png")
