"""Построение динамической системы (рис. 1) на signal.TransferFunction."""
import numpy as np
from scipy import signal


def series(*tfs):
    """Последовательное соединение звеньев: перемножение числителей и знаменателей."""
    num, den = np.array([1.0]), np.array([1.0])
    for n, d in tfs:
        num = np.polymul(num, n)
        den = np.polymul(den, d)
    return num, den


def pid_parts(kp, ki, kd):
    """PID = Kp + Ki/p + Kd*p = (Kd p^2 + Kp p + Ki) / p  (несобственное звено)."""
    if ki == 0:  # без интегратора сокращаем общий множитель p
        return np.array([kd, kp], dtype=float), np.array([1.0])
    return np.array([kd, kp, ki], dtype=float), np.array([1.0, 0.0])


def trim(num):
    """Убирает ведущие нули (иначе polyval/roots работают некорректно)."""
    num = np.atleast_1d(np.array(num, dtype=float))
    nz = np.nonzero(num)[0]
    return num[nz[0]:] if nz.size else np.array([0.0])


class ControlSystem:
    """Замкнутый контур: PID -> привод -> объект, единичная отрицательная ОС."""

    def __init__(self, plant, actuator, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.plant = plant
        self.actuator = actuator

        pid_n, pid_d = pid_parts(kp, ki, kd)
        # Разомкнутая система W(p) = W_pid * W_п * W_об.
        # Объединяем PID с остальными звеньями, поэтому дифференцирующая
        # часть не создаёт проблем с несобственностью.
        self.open_num, self.open_den = series((pid_n, pid_d), actuator, plant)
        self.open_num = trim(self.open_num)
        self.open_den = trim(self.open_den)

        # Замкнутая: T = W / (1 + W) = N / (D + N)
        n = self.open_num
        d = self.open_den
        char = np.polyadd(d, n)  # характеристический полином D + N
        self.closed_num = n
        self.closed_den = trim(char)

    # --- объекты SciPy ---
    @property
    def open_tf(self):
        return signal.TransferFunction(self.open_num, self.open_den)

    @property
    def closed_tf(self):
        return signal.TransferFunction(self.closed_num, self.closed_den)

    # --- свойства ---
    def poles(self):
        """Полюсы замкнутой системы (корни характеристического уравнения)."""
        return np.roots(self.closed_den)

    def is_stable(self):
        return bool(np.all(np.real(self.poles()) < 0))

    def freq_response(self, w):
        """W(jw) разомкнутой системы."""
        s = 1j * w
        return np.polyval(self.open_num, s) / np.polyval(self.open_den, s)

    def step_response(self, t):
        _, y = signal.step(self.closed_tf, T=t)
        return y
