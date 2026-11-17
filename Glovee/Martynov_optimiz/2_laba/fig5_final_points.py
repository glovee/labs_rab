"""Рисунок 5: в какие минимумы приходит поиск (200 запусков, зерна 0...199).
Сравниваются три варианта: золотое сечение (M = 20), обычный поиск с фиксированным шагом
(M = 20) и обычный поиск (M = 300). Во всех случаях R0 = 2, старт (2,5; 2,5).
Запуск: python fig5_final_points.py   (рядом должен лежать rastrigin_rs.py)
Результат: fig5_final_points.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rastrigin_rs import J, random_search

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.default"] = "regular"

X0 = (2.5, 2.5)
R0 = 2.0
N_RUNS = 200

METHODS = [
    ("золотое сечение, M = 20", dict(golden=True, M=20), "#4c78a8"),
    ("обычный поиск, M = 20", dict(golden=False, M=20), "#f58518"),
    ("обычный поиск, M = 300", dict(golden=False, M=300), "#54a24b"),
]


def category(x):
    """Ближайший узел решётки (|m|; |n|), m >= n: локальные минимумы лежат рядом с целыми точками."""
    a, b = sorted(np.abs(np.round(x)).astype(int), reverse=True)
    return (int(a), int(b))


counts, f_sum = [], {}
for name, kw, _ in METHODS:
    c = {}
    for seed in range(N_RUNS):
        r = random_search(x0=X0, R0=R0, q=0.5, rng=np.random.default_rng(seed), **kw)
        cat = category(r["x"])
        c[cat] = c.get(cat, 0) + 1
        f_sum.setdefault(cat, []).append(r["f"])
    counts.append(c)

cats = sorted({cat for c in counts for cat in c}, key=lambda t: (t[0] ** 2 + t[1] ** 2, t))
labels = []
for cat in cats:
    jv = float(np.mean(f_sum[cat]))
    labels.append(f"({cat[0]}; {cat[1]})\nJ ≈ {jv:.2f}".replace(".", ","))

fig, ax = plt.subplots(figsize=(11, 5))
w = 0.8 / len(METHODS)
xs = np.arange(len(cats))
for i, ((name, _, col), c) in enumerate(zip(METHODS, counts)):
    vals = [100.0 * c.get(cat, 0) / N_RUNS for cat in cats]
    bars = ax.bar(xs + (i - 1) * w, vals, w, color=col, label=name)
    for b, v in zip(bars, vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:g}".replace(".", ","),
                    ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("доля запусков, %")
ax.set_xlabel("минимум около узла решётки (m; n), в котором закончился поиск "
              "(знаки и порядок координат не учитываются)")
ax.set_title(f"Куда приходит случайный поиск ({N_RUNS} запусков, R0 = 2)")
ax.set_ylim(0, 112)
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("fig5_final_points.png", dpi=150)
for (name, _, _), c in zip(METHODS, counts):
    print(name, {k: v for k, v in sorted(c.items(), key=lambda kv: kv[0][0] ** 2 + kv[0][1] ** 2)})
print("сохранено: fig5_final_points.png")
