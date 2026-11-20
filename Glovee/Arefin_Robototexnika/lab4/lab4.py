import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, f as fdist

HERE = Path(__file__).resolve().parent
ALPHA = 0.05
# Вариант 1: два набора типов ГСН
SETS = [["Л", "Т", "Т_КР"], ["Л", "Т", "ИНС_СНС"]]

raw = pd.read_csv(HERE / "Data.csv", sep=";", decimal=",", encoding="utf-8-sig", index_col=0)
raw.columns = [c.strip() for c in raw.columns]


def normality(s, name, fname):
    mu, sd = s.mean(), s.std(ddof=1)
    ks = stats.kstest(s, stats.norm(mu, sd).cdf)
    sw = stats.shapiro(s)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    nb = int(round(1 + 3.322 * np.log10(len(s))))
    ax.hist(s, bins=nb, density=True, edgecolor="black", color="#8fb4d9", label="Эмпирическая")
    g = np.linspace(s.min() - sd, s.max() + sd, 300)
    ax.plot(g, stats.norm.pdf(g, mu, sd), "r-", label="Нормальная плотность")
    ax.set(title=f"Промах, ГСН {name}", xlabel="Промах", ylabel="Плотность")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(HERE / fname, dpi=150); plt.close(fig)
    return dict(mean=mu, sd=sd, ks_d=ks.statistic, ks_p=ks.pvalue, sw_w=sw.statistic, sw_p=sw.pvalue,
                normal=bool(ks.pvalue >= ALPHA and sw.pvalue >= ALPHA))


out = []
for si, cols in enumerate(SETS, 1):
    data = [raw[c].to_numpy() for c in cols]
    k, N = len(cols), sum(len(d) for d in data)
    r = {"cols": cols, "k": k, "N": N, "n": [len(d) for d in data]}
    r["norm"] = {c: normality(raw[c].to_numpy(), c, f"fig{si}_norm_{i}.png") for i, c in enumerate(cols)}
    r["desc"] = {c: dict(mean=d.mean(), sd=d.std(ddof=1), median=float(np.median(d)), min=d.min(), max=d.max(),
                         se=d.std(ddof=1) / np.sqrt(len(d))) for c, d in zip(cols, data)}
    r["all_normal"] = all(v["normal"] for v in r["norm"].values())

    lev = stats.levene(*data, center="median")
    r["levene"] = dict(W=lev.statistic, p=lev.pvalue)

    if r["all_normal"]:
        r["method"] = "anova"
        grand = np.concatenate(data).mean()
        ss_eff = sum(len(d) * (d.mean() - grand) ** 2 for d in data)
        ss_err = sum(((d - d.mean()) ** 2).sum() for d in data)
        df_e, df_r = k - 1, N - k
        ms_e, ms_r = ss_eff / df_e, ss_err / df_r
        F = ms_e / ms_r
        p = fdist.sf(F, df_e, df_r)
        r["anova"] = dict(ss_eff=ss_eff, df_eff=df_e, ms_eff=ms_e, ss_err=ss_err, df_err=df_r, ms_err=ms_r, F=F, p=p,
                          crit=fdist.isf(ALPHA, df_e, df_r), p_scipy=stats.f_oneway(*data).pvalue,
                          eta2=ss_eff / (ss_eff + ss_err))
        # Поправка Уэлча (на случай неравных дисперсий)
        w_ = np.array([len(d) / d.var(ddof=1) for d in data]); m_ = np.array([d.mean() for d in data])
        mw_ = (w_ * m_).sum() / w_.sum()
        lam_ = ((1 - w_ / w_.sum()) ** 2 / np.array([len(d) - 1 for d in data])).sum()
        F_w = ((w_ * (m_ - mw_) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam_)
        df_w = (k ** 2 - 1) / (3 * lam_)
        r["welch"] = dict(F=F_w, df1=k - 1, df2=df_w, p=fdist.sf(F_w, k - 1, df_w))
        # Тьюки (апостериорное сравнение)
        tk = stats.tukey_hsd(*data)
        r["tukey"] = [dict(a=cols[i], b=cols[j], diff=data[i].mean() - data[j].mean(), p=tk.pvalue[i, j])
                      for i in range(k) for j in range(i + 1, k)]
        # График средних с 95 % ДИ
        fig, ax = plt.subplots(figsize=(6, 4))
        means = [d.mean() for d in data]
        ci = [stats.t.isf(ALPHA / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) for d in data]
        ax.errorbar(cols, means, yerr=ci, fmt="o-", capsize=6, color="black")
        ax.set(title="Средние значения промаха и 95 % доверительные интервалы", xlabel="Тип ГСН", ylabel="Промах")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(HERE / f"fig{si}_means.png", dpi=150); plt.close(fig)
    else:
        r["method"] = "kw"
        pooled = np.concatenate(data)
        ranks = stats.rankdata(pooled)
        pos = np.cumsum([0] + [len(d) for d in data])
        sums = [ranks[pos[i]:pos[i + 1]].sum() for i in range(k)]
        H, p = stats.kruskal(*data)
        r["kw"] = dict(rank_sums=sums, mean_ranks=[s / len(d) for s, d in zip(sums, data)], H=H, df=k - 1, p=p,
                       crit=chi2.isf(ALPHA, k - 1))
        # Критерий медиан
        st, pm, med, tbl = stats.median_test(*data, ties="below", correction=False)
        n_arr = np.array([len(d) for d in data])
        below_o, above_o = tbl[1], tbl[0]  # tbl[0] — число значений > медианы, tbl[1] — <=
        tot_b, tot_a = below_o.sum(), above_o.sum()
        below_e, above_e = n_arr * tot_b / N, n_arr * tot_a / N
        r["median"] = dict(median=float(med), chi2=st, df=k - 1, p=pm, below_o=below_o.tolist(),
                           above_o=above_o.tolist(), below_e=below_e.tolist(), above_e=above_e.tolist())
        # Попарные сравнения (Манн–Уитни, поправка Бонферрони)
        pairs = []
        m_ = k * (k - 1) // 2
        for i in range(k):
            for j in range(i + 1, k):
                mw = stats.mannwhitneyu(data[i], data[j], alternative="two-sided")
                pairs.append(dict(a=cols[i], b=cols[j], U=mw.statistic, p=mw.pvalue, p_bonf=min(1.0, mw.pvalue * m_)))
        r["pairs"] = pairs
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, tick_labels=cols, showmeans=True)
        ax.set(title="Распределение промаха по типам ГСН", xlabel="Тип ГСН", ylabel="Промах")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(HERE / f"fig{si}_box.png", dpi=150); plt.close(fig)
    out.append(r)

    print(f"\n=== Набор {si}: {cols}; метод: {r['method']}")
    for c in cols: print(c, {kk: round(float(v), 4) for kk, v in r["norm"][c].items()})
    print("Levene", r["levene"])
    for key in ("anova", "welch", "tukey", "kw", "median", "pairs"):
        if key in r: print(key, r[key])

(HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
