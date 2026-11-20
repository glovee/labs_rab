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


def normality(s):
    mu, std = stats.norm.fit(s)
    edges = np.linspace(s.min(), s.max(), 9)
    obs, _ = np.histogram(s, bins=edges)
    exp = len(s) * np.diff(stats.norm.cdf(edges, mu, std))
    exp = exp * obs.sum() / exp.sum()
    chi2, p = stats.chisquare(obs, exp, ddof=2)
    ks = stats.kstest(s, stats.norm(s.mean(), s.std(ddof=1)).cdf)
    sw = stats.shapiro(s)
    return dict(chi2=chi2, df=len(obs) - 3, chi2_p=p, ks_d=ks.statistic, ks_p=ks.pvalue,
                sw_w=sw.statistic, sw_p=sw.pvalue, edges=edges.tolist(), observed=obs.tolist())


out = []
for k, (a, b) in enumerate(PAIRS, 1):
    x, y = raw[a].to_numpy(), raw[b].to_numpy()
    n, m = len(x), len(y)
    r = {"names": [a, b], "n": n, "m": m, "x": x.tolist(), "y": y.tolist()}
    r["desc"] = {nm: dict(mean=s.mean(), sd=s.std(ddof=1), var=s.var(ddof=1), median=float(np.median(s)),
                          min=s.min(), max=s.max()) for nm, s in ((a, x), (b, y))}
    r["norm"] = {a: normality(x), b: normality(y)}

    # Фишер
    F = x.var(ddof=1) / y.var(ddof=1)
    d1, d2 = n - 1, m - 1
    p_f = 2 * min(stats.f.cdf(F, d1, d2), stats.f.sf(F, d1, d2))
    f_lo, f_hi = stats.f.ppf(ALPHA / 2, d1, d2), stats.f.isf(ALPHA / 2, d1, d2)
    equal = p_f >= ALPHA
    r["fisher"] = dict(F=F, df1=d1, df2=d2, p=p_f, lo=f_lo, hi=f_hi, equal=bool(equal))

    # Стьюдент / Уэлч
    t = stats.ttest_ind(x, y, equal_var=equal)
    dft = t.df
    t_crit = stats.t.isf(ALPHA / 2, dft)
    diff = x.mean() - y.mean()
    se_diff = np.sqrt(x.var(ddof=1) / n + y.var(ddof=1) / m) if not equal else \
        np.sqrt(((n - 1) * x.var(ddof=1) + (m - 1) * y.var(ddof=1)) / (n + m - 2) * (1 / n + 1 / m))
    r["t"] = dict(t=t.statistic, df=dft, p=t.pvalue, crit=t_crit, diff=diff, se=se_diff,
                  ci=[diff - t_crit * se_diff, diff + t_crit * se_diff], welch=not equal,
                  reject=bool(t.pvalue < ALPHA))

    # Графики
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 13)
    ax.hist(x, bins=bins, alpha=0.6, edgecolor="black", label=a)
    ax.hist(y, bins=bins, alpha=0.6, edgecolor="black", label=b)
    ax.set(title=f"Гистограммы: {a} и {b}", xlabel="Степень автономности полёта", ylabel="Частота")
    ax.legend(); fig.tight_layout(); fig.savefig(HERE / f"fig{k}_hist.png", dpi=150); plt.close(fig)

    lim = max(5, abs(t.statistic) + 1)
    g = np.linspace(-lim, lim, 600)
    pdf = stats.t.pdf(g, dft)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(g, pdf, "k")
    ax.fill_between(g, pdf, where=np.abs(g) <= t_crit, alpha=0.3, label=f"Область принятия H0 ({1 - ALPHA:.2f})")
    ax.axvline(-t_crit, color="blue", ls="--", label=f"±t крит = ±{t_crit:.2f}")
    ax.axvline(t_crit, color="blue", ls="--")
    ax.axvline(t.statistic, color="red", label=f"t = {t.statistic:.2f}")
    ax.set(title=f"Распределение Стьюдента, df = {dft:.2f}", xlabel="t", ylabel="Плотность")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(HERE / f"fig{k}_t.png", dpi=150); plt.close(fig)
    out.append(r)

    print(f"\n=== {a} vs {b}")
    print({kk: (round(float(v), 4) if not isinstance(v, (list, bool)) else v) for kk, v in r["fisher"].items()})
    print({kk: (round(float(v), 5) if not isinstance(v, (list, bool)) else v) for kk, v in r["t"].items()})
    for nm in (a, b):
        z = r["norm"][nm]
        print(nm, {kk: round(float(v), 4) for kk, v in z.items() if kk not in ("edges", "observed")}, z["observed"])
        print("  desc", {kk: round(float(v), 4) for kk, v in r["desc"][nm].items()})

(HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
