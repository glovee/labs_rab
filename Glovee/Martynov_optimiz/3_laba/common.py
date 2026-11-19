# -*- coding: utf-8 -*-
"""Общие вспомогательные функции для скриптов построения графиков."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rcParams["font.size"] = 10


def rosenbrock_grid(xlim, ylim, n=600):
    gx = np.linspace(*xlim, n); gy = np.linspace(*ylim, n)
    X, Y = np.meshgrid(gx, gy)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    return X, Y, Z


def draw_region(ax, xlim, ylim):
    """Серым закрашена недопустимая область, штрих - границы x1+x2=1 и x2=0."""
    xx = np.linspace(*xlim, 400)
    XX, YY = np.meshgrid(np.linspace(*xlim, 500), np.linspace(*ylim, 500))
    bad = ((XX + YY > 1) | (YY < 0)).astype(float)
    ax.contourf(XX, YY, bad, levels=[0.5, 1.5], colors=["grey"], alpha=0.35)
    ax.plot(xx, 1 - xx, "k--", lw=1.2, label=r"$x_1+x_2=1$")
    ax.axhline(0, color="k", ls=":", lw=1.2, label=r"$x_2=0$")


def log_colorbar(fig, ax, cs):
    cb = fig.colorbar(cs, ax=ax, label=r"$f(x_1,x_2)$ (лог. шкала)")
    lo, hi = np.log10(cs.levels[0]), np.log10(cs.levels[-1])
    cb.set_ticks(10.0 ** np.arange(np.ceil(lo), np.floor(hi) + 1))
    cb.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    cb.ax.minorticks_off()
    return cb
