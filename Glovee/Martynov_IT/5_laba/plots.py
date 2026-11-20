"""Виджеты matplotlib для PyQt: переходный процесс и годограф."""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class _BaseCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)


class StepCanvas(_BaseCanvas):
    def update_plot(self, t, y, info, band):
        ax = self.ax
        ax.clear()
        ax.plot(t, y, label="y(t)", color="tab:blue")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="g(t) = 1")

        if info["stable"]:
            yf = info["y_final"]
            ax.axhline(yf, color="green", ls=":", lw=0.8, label="y уст.")
            ax.fill_between(t, yf * (1 - band), yf * (1 + band),
                            color="orange", alpha=0.2, label=f"трубка ±{band * 100:.0f}%")
            if info["t_settle"] is not None:
                ax.axvline(info["t_settle"], color="red", ls="--", lw=0.8,
                           label=f"t п = {info['t_settle']:.1f} c")
        ax.set_title("Переходный процесс (реакция на единичный скачок)")
        ax.set_xlabel("t, c")
        ax.set_ylabel("y(t)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        if not info["stable"]:
            ymax = max(2.0, float(np.percentile(np.abs(y), 90)) * 1.5)
            ax.set_ylim(-ymax, ymax)
        self.draw()


class NyquistCanvas(_BaseCanvas):
    def update_plot(self, W, margins):
        ax = self.ax
        ax.clear()
        ax.plot(W.real, W.imag, color="tab:blue", label="W(jω), ω ≥ 0")
        ax.plot(W.real, -W.imag, color="tab:blue", ls="--", lw=0.8, label="ω < 0")
        ax.plot(-1, 0, "r+", ms=12, mew=2, label="(−1, j0)")

        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.6, ls=":")

        # Видимая область: ограничиваем, чтобы огромные значения на ω→0 не «съедали» график
        lim = 3.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("Годограф Найквиста разомкнутой системы")
        ax.set_xlabel("Re W")
        ax.set_ylabel("Im W")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        self.draw()
