# -*- coding: utf-8 -*-
"""Лабораторная работа №3. Вариант 1. Метод штрафных функций.

Минимизировать f = (1-x1)^2 + 100*(x2-x1^2)^2 при x1 + x2 <= 1, x2 >= 0.

Основной скрипт расчёта. Запуск:  python penalty.py
Результат сохраняется в results.json, его используют скрипты построения
графиков (plot_*.py) и таблицы (make_table.py).
"""
import json
import numpy as np
from scipy.optimize import minimize, brentq

X0 = (-1.0, 1.0)   # начальная точка


def f(x):
    """Целевая функция (Розенброк)."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_f(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                     200 * (x[1] - x[0] ** 2)])


def g(x):
    """Ограничения в виде g_i <= 0."""
    return np.array([x[0] + x[1] - 1.0,   # x1 + x2 <= 1
                     -x[1]])              # x2 >= 0


def penalty(x):
    """Сумма квадратов нарушений ограничений: sum max(0, g_i)^2."""
    return np.sum(np.maximum(0.0, g(x)) ** 2)


def F(x, r):
    """Штрафная функция F = f + r * sum max(0, g_i)^2."""
    return f(x) + r * penalty(x)


def grad_F(x, r):
    gv = np.maximum(0.0, g(x))
    dg = np.array([[1.0, 1.0], [0.0, -1.0]])      # градиенты g_1, g_2
    return grad_f(x) + 2 * r * dg.T @ gv


def penalty_method(x0, r0=1.0, c=10.0, eps=1e-8, kmax=15):
    """Метод штрафных функций: r_k = r0 * c^k, внутренняя задача - BFGS."""
    x = np.array(x0, float)
    r = r0
    hist = []            # итоги внешних итераций
    path = [x.copy()]    # вся траектория внутренней минимизации
    for k in range(kmax):
        cur = [x.copy()]
        res = minimize(lambda z: F(z, r), x, jac=lambda z: grad_F(z, r),
                       method="BFGS", options={"gtol": 1e-12, "maxiter": 10000},
                       callback=lambda z: cur.append(z.copy()))
        x_new = res.x
        path += cur[1:]
        viol = np.max(np.maximum(0.0, g(x_new)))
        hist.append(dict(k=k, r=r, x1=x_new[0], x2=x_new[1], f=f(x_new),
                         viol=viol, P=penalty(x_new), F=F(x_new, r),
                         nit=res.nit, path=np.array(cur)))
        done = (k > 0 and np.linalg.norm(x_new - x) < eps and viol < 1e-5)
        x = x_new
        if done or (r * penalty(x) < eps and viol < 1e-6 and k > 0):
            break
        r *= c
    return x, hist, np.array(path)


def exact_solution():
    """Точное решение для проверки: на границе x1 + x2 = 1 (x1 = t, x2 = 1 - t)."""
    dh = lambda t: -2 * (1 - t) + 200 * (1 - t - t * t) * (-1 - 2 * t)
    t = brentq(dh, 0.0, 1.0)
    return np.array([t, 1 - t])


def multistart(n=40, seed=0):
    """Проверка: запуск метода из n случайных точек области [-2,2]x[-1,3]."""
    rng = np.random.default_rng(seed)
    found = set()
    for _ in range(n):
        x = rng.uniform([-2, -1], [2, 3])
        r = 1.0
        for _k in range(8):
            x = minimize(lambda z: F(z, r), x, jac=lambda z: grad_F(z, r),
                         method="BFGS", options={"gtol": 1e-12}).x
            r *= 10
        found.add((round(float(x[0]), 4), round(float(x[1]), 4)))
    return found


def load_results(path="results.json"):
    """Чтение results.json (используется скриптами графиков и таблиц)."""
    R = json.load(open(path, encoding="utf-8"))
    for h in R["hist"]:
        h["path"] = np.array(h["path"])
    R["path"] = np.array(R["path"])
    return R


if __name__ == "__main__":
    x, hist, path = penalty_method(X0)
    xs = exact_solution()

    print(f"{'k':>2} {'r':>9} {'x1':>10} {'x2':>10} {'f':>10} {'max g+':>10} {'P':>10} {'F':>10} {'iter':>5}")
    for h in hist:
        print(f"{h['k']:>2} {h['r']:>9.0e} {h['x1']:>10.6f} {h['x2']:>10.6f} {h['f']:>10.6f} "
              f"{h['viol']:>10.2e} {h['P']:>10.2e} {h['F']:>10.6f} {h['nit']:>5}")
    print("Результат :", x, f(x), g(x))
    print("Точное    :", xs, f(xs), g(xs))
    print("Ошибка    :", np.linalg.norm(x - xs))
    ms = multistart()
    print("Мультистарт (40 точек):", ms)

    json.dump(dict(x0=list(X0), x=x.tolist(), fx=float(f(x)), xs=xs.tolist(), fs=float(f(xs)),
                   multistart=sorted(ms), path=path.tolist(),
                   hist=[{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in h.items()} for h in hist]),
              open("results.json", "w", encoding="utf-8"))
    print("Сохранено: results.json")
