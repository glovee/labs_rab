"""Показатели переходного процесса и запасы устойчивости по Найквисту."""
import numpy as np

from model import ControlSystem


def suggest_time_horizon(system: ControlSystem, default=200.0):
    """Подбирает время моделирования по медленнейшему полюсу."""
    poles = system.poles()
    stable_poles = poles[np.real(poles) < -1e-9]
    if stable_poles.size == 0:
        return default
    slowest = np.min(np.abs(np.real(stable_poles)))
    return float(np.clip(6.0 / slowest, 20.0, 5000.0))


def transient_metrics(t, y, y_final, band=0.05):
    """Перерегулирование, установившееся значение, время переходного процесса.

    Время переходного процесса — последний момент, когда |y - y_уст| > band*|y_уст|.
    """
    y_peak = float(np.max(y))
    if abs(y_final) > 1e-9:
        overshoot = max(0.0, (y_peak - y_final) / abs(y_final) * 100.0)
    else:
        overshoot = 0.0

    tol = band * abs(y_final)
    outside = np.where(np.abs(y - y_final) > tol)[0]
    if outside.size == 0:
        t_settle = 0.0
    elif outside[-1] >= len(t) - 1:
        t_settle = None  # не уложился в окно моделирования
    else:
        t_settle = float(t[outside[-1] + 1])
    return overshoot, t_settle


def compute_step_metrics(system: ControlSystem, band=0.05, n_points=4000):
    """Возвращает (t, y, словарь показателей)."""
    horizon = suggest_time_horizon(system)
    t = np.linspace(0, horizon, n_points)
    y = system.step_response(t)

    stable = system.is_stable()
    if stable:
        # Теорема о конечном значении: y_уст = T(0)
        y_final = float(system.closed_num[-1] / system.closed_den[-1])
    else:
        y_final = float(y[-1])

    overshoot, t_settle = (None, None)
    if stable:
        overshoot, t_settle = transient_metrics(t, y, y_final, band)

    return t, y, {
        "stable": stable,
        "y_final": y_final,
        "overshoot": overshoot,
        "t_settle": t_settle,
    }


def nyquist_data(system: ControlSystem, w_min=1e-4, w_max=1e3, n=20000):
    """Годограф W(jw) для w >= 0 и зеркальная часть (w < 0)."""
    w = np.logspace(np.log10(w_min), np.log10(w_max), n)
    return w, system.freq_response(w)


def stability_margins(system: ControlSystem):
    """Запас по амплитуде (GM) и по фазе (PM) по разомкнутой частотной характеристике.

    Возвращает словарь; значения None, если соответствующей частоты нет.
    """
    w = np.logspace(-4, 3, 200000)
    W = system.freq_response(w)
    mag = np.abs(W)
    phase = np.unwrap(np.angle(W))  # рад

    result = {"pm_deg": None, "w_gc": None, "gm": None, "gm_db": None, "w_pc": None}

    # Частота среза: |W| = 1 (последнее пересечение сверху вниз)
    idx = np.where((mag[:-1] - 1) * (mag[1:] - 1) < 0)[0]
    if idx.size:
        i = idx[-1]
        w_gc = _interp(w[i], w[i + 1], mag[i] - 1, mag[i + 1] - 1)
        ph = np.interp(w_gc, w, phase)
        result["w_gc"] = float(w_gc)
        result["pm_deg"] = float(np.degrees(ph) + 180.0)

    # Частота -180°: пересечение фазой уровня -pi (по модулю 2pi)
    target = -np.pi
    # смещаем фазу так, чтобы искать пересечение -pi
    d = phase - target
    idx = np.where(d[:-1] * d[1:] < 0)[0]
    if idx.size:
        i = idx[0]
        w_pc = _interp(w[i], w[i + 1], d[i], d[i + 1])
        m = np.interp(w_pc, w, mag)
        result["w_pc"] = float(w_pc)
        result["gm"] = float(1.0 / m)
        result["gm_db"] = float(-20 * np.log10(m))
    return result


def _interp(x0, x1, f0, f1):
    """Линейная интерполяция корня между двумя точками."""
    return x0 - f0 * (x1 - x0) / (f1 - f0)


def nyquist_verdict(system: ControlSystem, margins: dict):
    """Текстовая оценка устойчивости.

    Устойчивость определяется критерием Найквиста: замкнутая система устойчива,
    если годограф W(jw) не охватывает точку (-1, j0) (разомкнутая система
    здесь устойчива/нейтральна, т.е. P = 0). Эквивалентная проверка —
    корни характеристического полинома — используется для строгого вывода.
    """
    stable = system.is_stable()
    lines = [("Устойчива" if stable else "НЕУСТОЙЧИВА") + " (критерий Найквиста)"]
    if margins["pm_deg"] is not None:
        lines.append(f"Запас по фазе: {margins['pm_deg']:.1f}° (ω = {margins['w_gc']:.3f} рад/с)")
    else:
        lines.append("Запас по фазе: |W| < 1 на всех частотах")
    if margins["gm"] is not None:
        lines.append(f"Запас по амплитуде: {margins['gm']:.2f} ({margins['gm_db']:.1f} дБ), "
                     f"ω = {margins['w_pc']:.3f} рад/с")
    else:
        lines.append("Запас по амплитуде: фаза не достигает −180° (∞)")
    return "\n".join(lines)
