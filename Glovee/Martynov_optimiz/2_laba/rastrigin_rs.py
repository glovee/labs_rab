"""Случайный поиск с золотым сечением для функции Растригина (вариант 4).

J(x) = 20 + x1^2 + x2^2 - 10 (cos(2 pi x1) + cos(2 pi x2)),  x0 = (2,5; 2,5).
Глобальный минимум: x* = (0; 0), J(x*) = 0. Локальных минимумов много
(около каждой точки целочисленной решётки).

Алгоритм (одна итерация):
  1. Выбрать случайное направление d (равномерно на единичной окружности).
  2. Вдоль луча x + t d, 0 <= t <= R, найти шаг t* методом золотого сечения.
  3. Если J(x + t* d) < J(x), перейти в новую точку (успех), иначе остаться (неудача).
  4. После M неудач подряд радиус поиска R уменьшается: R := q R.
  5. Остановка: R < R_min или исчерпан лимит итераций.
"""
import numpy as np

PHI = (np.sqrt(5.0) - 1.0) / 2.0        # 0,618... - коэффициент золотого сечения


def J(x):
    """Функция Растригина (n = 2)."""
    x1, x2 = x
    return 20.0 + x1 ** 2 + x2 ** 2 - 10.0 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))


def golden_section(phi, a, b, tol=1e-8):
    """Минимум phi на [a, b] методом золотого сечения.
    Возвращает (t*, число вычислений phi)."""
    c = b - PHI * (b - a)
    d = a + PHI * (b - a)
    fc, fd = phi(c), phi(d)
    n = 2
    while b - a > tol:
        if fc < fd:                       # минимум в [a, d]
            b, d, fd = d, c, fc
            c = b - PHI * (b - a)
            fc = phi(c)
        else:                             # минимум в [c, b]
            a, c, fc = c, d, fd
            d = a + PHI * (b - a)
            fd = phi(d)
        n += 1
    return 0.5 * (a + b), n


def random_search(x0=(2.5, 2.5), R0=1.0, M=10, q=0.5, R_min=1e-6,
                  golden=True, rng=None, max_iter=100000, tol=1e-8):
    """Случайный поиск.

    golden=True  - шаг вдоль случайного направления подбирается золотым сечением на [0, R];
    golden=False - шаг фиксирован и равен R (обычный случайный поиск, для сравнения).

    Возвращает словарь:
      x, f        - найденная точка и значение функции,
      path        - принятые точки (массив, первая строка - x0),
      log         - список (итерация, x1, x2, J, R, число вычислений J) по каждому успеху,
      n_iter      - число выполненных итераций (пробных направлений),
      n_eval      - общее число вычислений J,
      f_hist      - лучшее значение J после каждой итерации,
      R_hist      - радиус поиска на каждой итерации.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.array(x0, dtype=float)
    f = J(x)
    n_eval = 1
    R = R0
    fails = 0
    path = [x.copy()]
    log = [(0, x[0], x[1], f, R, n_eval)]
    f_hist, R_hist = [f], [R]
    k = 0
    while R >= R_min and k < max_iter:
        k += 1
        phi_angle = rng.uniform(0.0, 2.0 * np.pi)
        d = np.array([np.cos(phi_angle), np.sin(phi_angle)])
        if golden:
            t, n = golden_section(lambda s: J(x + s * d), 0.0, R, tol)
            n_eval += n
        else:
            t = R
        x_new = x + t * d
        f_new = J(x_new)
        n_eval += 1
        if f_new < f:
            x, f = x_new, f_new
            fails = 0
            path.append(x.copy())
            log.append((k, x[0], x[1], f, R, n_eval))
        else:
            fails += 1
            if fails >= M:
                R *= q
                fails = 0
        f_hist.append(f)
        R_hist.append(R)
    return {"x": x, "f": f, "path": np.array(path), "log": log, "n_iter": k,
            "n_eval": n_eval, "f_hist": np.array(f_hist), "R_hist": np.array(R_hist)}


def run_statistics(n_runs=200, x0=(2.5, 2.5), success_tol=1e-3, **kw):
    """Серия запусков с зёрнами 0...n_runs-1.
    Запуск считается успешным, если найденная точка ближе success_tol к глобальному минимуму (0; 0).
    Возвращает (доля успешных запусков, медиана числа вычислений J, среднее найденное J).
    Остальные параметры (R0, M, q, golden, ...) передаются в random_search."""
    ok, evals, fs = 0, [], []
    for seed in range(n_runs):
        r = random_search(x0=x0, rng=np.random.default_rng(seed), **kw)
        ok += bool(np.linalg.norm(r["x"]) < success_tol)
        evals.append(r["n_eval"])
        fs.append(r["f"])
    return ok / n_runs, float(np.median(evals)), float(np.mean(fs))


if __name__ == "__main__":
    res = random_search(rng=np.random.default_rng(1))
    print("x* =", res["x"], " J =", res["f"], " итераций:", res["n_iter"], " вычислений J:", res["n_eval"])
