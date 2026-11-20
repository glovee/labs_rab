import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ALPHA = 0.05
PAIRS = [("УАБ1", "УАБ2"), ("УАБ6", "УАБ8")]  # вариант 1

raw = pd.read_csv(HERE / "Data.csv", sep=";", decimal=",", encoding="utf-8-sig")
out = []

for k, (a, b) in enumerate(PAIRS, 1):
    x, y = raw[a].to_numpy(), raw[b].to_numpy()
    n, m = len(x), len(y)
    r = {"names": [a, b], "n": n, "m": m, "x": x.tolist(), "y": y.tolist(),
         "median": [float(np.median(x)), float(np.median(y))]}

    # Ранги
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled)
    Rx, Ry = ranks[:n].sum(), ranks[n:].sum()
    U_rank = Ry - m * (m + 1) / 2
    table = []
    for v in np.unique(pooled):
        first, last = int(np.sum(pooled < v) + 1), int(np.sum(pooled <= v))
        table.append(dict(v=float(v), cx=int(np.sum(x == v)), cy=int(np.sum(y == v)),
                          first=first, last=last, rank=float(ranks[pooled == v][0])))
    r["ranks"] = dict(table=table, Rx=Rx, Ry=Ry, total=Rx + Ry, check=(n + m) * (n + m + 1) / 2, U=U_rank)

    # Попарное сравнение
    wins = int(sum(1 for xi in x for yj in y if xi < yj))
    ties = int(sum(1 for xi in x for yj in y if xi == yj))
    r["pairs"] = dict(total=n * m, wins=wins, ties=ties, losses=n * m - wins - ties, U=wins + 0.5 * ties)

    # Манн–Уитни
    mw = stats.mannwhitneyu(y, x, alternative="two-sided", method="asymptotic")
    mu_U = n * m / 2
    sig_U = np.sqrt(n * m * (n + m + 1) / 12)
    Z = (U_rank - mu_U) / sig_U
    zc = stats.norm.isf(ALPHA / 2)
    # поправка на совпадения для sigma
    _, cnt = np.unique(pooled, return_counts=True)
    N = n + m
    sig_tie = np.sqrt(n * m / 12 * ((N + 1) - (cnt ** 3 - cnt).sum() / (N * (N - 1))))
    r["mw"] = dict(U=mw.statistic, p=mw.pvalue, mu=mu_U, sigma=sig_U, sigma_tie=sig_tie, Z=Z, zc=zc,
                   p_z=2 * stats.norm.sf(abs(Z)), U_lo=mu_U - zc * sig_U, U_hi=mu_U + zc * sig_U,
                   Z_tie=(U_rank - mu_U) / sig_tie, reject=bool(mw.pvalue < ALPHA))

    # Колмогоров–Смирнов
    ks = stats.ks_2samp(x, y, alternative="two-sided", method="auto")
    loc = ks.statistic_location
    lam = np.sqrt(n * m / (n + m))
    d_crit = stats.kstwobign.isf(ALPHA) / lam
    r["ks"] = dict(D=ks.statistic, loc=float(loc), Fx=float(np.mean(x <= loc)), Fy=float(np.mean(y <= loc)),
                   d_crit=d_crit, p=ks.pvalue, reject=bool(ks.pvalue < ALPHA))

    # Графики
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = np.arange(1, 11)
    w = 0.38
    ax.bar(vals - w / 2, [np.sum(x == v) for v in vals], w, edgecolor="black", label=a)
    ax.bar(vals + w / 2, [np.sum(y == v) for v in vals], w, edgecolor="black", label=b)
    ax.set(xticks=vals, title=f"Гистограммы: {a} и {b}", xlabel="Точность наведения, баллы", ylabel="Частота")
    ax.legend(); fig.tight_layout(); fig.savefig(HERE / f"fig{k}_hist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.ecdf(x, label=a); ax.ecdf(y, label=b)
    ax.vlines(loc + 0.5, r["ks"]["Fx"], r["ks"]["Fy"], color="red", lw=2, label=f"D = {ks.statistic:.2f}")
    ax.set(title=f"Выборочные функции распределения: {a} и {b}", xlabel="Точность наведения, баллы", ylabel="F(x)")
    ax.legend(); fig.tight_layout(); fig.savefig(HERE / f"fig{k}_ecdf.png", dpi=150); plt.close(fig)

    g = np.linspace(0, 0.8, 500)
    pdf = stats.kstwobign.pdf(g * lam) * lam
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(g, pdf, "k")
    ax.fill_between(g, pdf, where=g >= d_crit, alpha=0.3, label=f"Критическая область ({ALPHA})")
    ax.axvline(d_crit, color="blue", ls="--", label=f"D крит = {d_crit:.4f}")
    ax.axvline(min(ks.statistic, 0.8), color="red", label=f"D = {ks.statistic:.2f}")
    ax.set(title=f"Распределение статистики D, n = {n}, m = {m}", xlabel="D", ylabel="Плотность")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(HERE / f"fig{k}_kolm.png", dpi=150); plt.close(fig)
    out.append(r)

    print(f"\n=== {a} vs {b}; медианы {r['median']}")
    print("ranks", {kk: v for kk, v in r["ranks"].items() if kk != "table"})
    print("pairs", r["pairs"])
    print("mw", {kk: round(float(v), 4) for kk, v in r["mw"].items()})
    print("ks", {kk: round(float(v), 4) for kk, v in r["ks"].items()})
    print("совпадение U:", U_rank == r["pairs"]["U"] == mw.statistic)

(HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
