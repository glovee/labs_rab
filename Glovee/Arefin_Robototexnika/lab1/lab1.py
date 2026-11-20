import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = Path(__file__).resolve().parent
ALPHA = 0.05
COLUMN = "УАБ1"  # вариант 1

raw = pd.read_csv(HERE / "Data.csv", sep=";", decimal=",", encoding="utf-8-sig")
x = raw[COLUMN].to_numpy()
n = len(x)
res = {"column": COLUMN, "n": n, "data": x.tolist()}

# Параметры выборки
mean, sd, med = x.mean(), x.std(ddof=1), float(np.median(x))
se = sd / np.sqrt(n)
t_c = stats.t.isf(ALPHA / 2, n - 1)
q1, q3 = np.percentile(x, [25, 75])
res["params"] = dict(mean=mean, sd=sd, median=med, min=x.min(), max=x.max(), se=se,
                     ci=[mean - t_c * se, mean + t_c * se], q1=q1, q3=q3,
                     skew=stats.skew(x, bias=False), kurt=stats.kurtosis(x, bias=False))

# Гистограмма с нормальной плотностью
num_bins = 8
edges = np.linspace(x.min(), x.max(), num_bins + 1)
observed, _ = np.histogram(x, bins=edges)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(x, bins=edges, edgecolor="black", color="#8fb4d9", label="Эмпирическая")
g = np.linspace(x.min() - 0.05, x.max() + 0.05, 300)
ax.plot(g, stats.norm.pdf(g, mean, sd) * n * (edges[1] - edges[0]), "r-", label="Нормальная (ожидаемая)")
ax.set(title=f"Гистограмма: {COLUMN}", xlabel="Промах, у.е.", ylabel="Частота")
ax.legend(); fig.tight_layout(); fig.savefig(HERE / "fig1_hist.png", dpi=150); plt.close(fig)

# Выборочная функция распределения
fig, ax = plt.subplots(figsize=(6.5, 4))
xs = np.sort(x)
ax.step(xs, np.arange(1, n + 1) / n, where="post", label="F*(x)")
ax.plot(g, stats.norm.cdf(g, mean, sd), "r-", label="Нормальная F(x)")
ax.set(title=f"Выборочная функция распределения: {COLUMN}", xlabel="Промах, у.е.", ylabel="F*(x)")
ax.legend(); fig.tight_layout(); fig.savefig(HERE / "fig2_ecdf.png", dpi=150); plt.close(fig)

# Ящик с усами и нормальная вероятностная бумага
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
axs[0].boxplot(x, showmeans=True); axs[0].set(title="Ящик с усами", xticks=[1], xticklabels=[COLUMN], ylabel="Промах, у.е.")
stats.probplot(x, dist="norm", plot=axs[1]); axs[1].set(title="Нормальная вероятностная бумага", xlabel="Квантили N(0,1)", ylabel="Значения")
fig.tight_layout(); fig.savefig(HERE / "fig3_box_qq.png", dpi=150); plt.close(fig)

# Хи-квадрат (нормальный закон)
mu, std = stats.norm.fit(x)
def chi_test(dist, params, d):
    exp = n * np.diff(dist.cdf(edges, *params))
    exp = exp * observed.sum() / exp.sum()
    c, p = stats.chisquare(observed, exp, ddof=d)
    return dict(expected=exp.tolist(), chi2=c, df=num_bins - 1 - d, p=p, crit=stats.chi2.isf(ALPHA, num_bins - 1 - d))
res["edges"] = edges.tolist(); res["observed"] = observed.tolist()
res["norm"] = dict(mu=mu, std=std, **chi_test(stats.norm, (mu, std), 2))
ks = stats.kstest(x, stats.norm(mean, sd).cdf)
sw = stats.shapiro(x)
res["norm"]["ks_d"] = ks.statistic; res["norm"]["ks_p"] = ks.pvalue
res["norm"]["sw_w"] = sw.statistic; res["norm"]["sw_p"] = sw.pvalue

# Критерий Лиллиефорса (поправка на оценку параметров) через Монте-Карло
rng = np.random.default_rng(1)
def ks_stat(s):
    return stats.kstest(s, stats.norm(s.mean(), s.std(ddof=1)).cdf).statistic
sim = np.array([ks_stat(rng.normal(size=n)) for _ in range(20000)])
res["norm"]["lillie_p"] = float((sim >= ks.statistic).mean())

# Другие законы
laws = [("Равномерный", stats.uniform, stats.uniform.fit(x), 2),
        ("Экспоненциальный", stats.expon, stats.expon.fit(x, floc=0), 1),
        ("Логнормальный", stats.lognorm, stats.lognorm.fit(x, floc=0), 2)]
res["laws"] = []
for name, dist, params, d in laws:
    r = chi_test(dist, params, d)
    k_ = stats.kstest(x, lambda v: dist.cdf(v, *params))
    res["laws"].append(dict(name=name, params=list(map(float, params)), ks_d=k_.statistic, ks_p=k_.pvalue, **r))
sw_ln = stats.shapiro(np.log(x))
res["lognorm_sw"] = dict(w=sw_ln.statistic, p=sw_ln.pvalue)

# Сравнение законов на одном графике
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(x, bins=edges, density=True, edgecolor="black", color="#cfe0f1", label="Эмпирическая")
gg = np.linspace(max(x.min() - 0.1, 1e-3), x.max() + 0.1, 400)
ax.plot(gg, stats.norm.pdf(gg, mu, std), label="Нормальный")
for name, dist, params, d in laws:
    ax.plot(gg, dist.pdf(gg, *params), "--", label=name)
ax.set(title=f"Сравнение законов распределения: {COLUMN}", xlabel="Промах, у.е.", ylabel="Плотность")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(HERE / "fig4_laws.png", dpi=150); plt.close(fig)

(HERE / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

print(f"Столбец: {COLUMN}, n = {n}")
print(f"mean={mean:.4f} sd={sd:.4f} median={med:.4f}")
print("Нормальный:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["norm"].items() if k != "expected"})
for l in res["laws"]:
    print(l["name"], f"chi2={l['chi2']:.3f} df={l['df']} p={l['p']:.4f} KS D={l['ks_d']:.4f} p={l['ks_p']:.4f}")
print("Шапиро ln(x):", res["lognorm_sw"])
