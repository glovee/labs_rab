"""
Вариант 1.
Движение тела в центральном (гравитационном) поле, круговая орбита (e = 0).

1) Численное интегрирование уравнений движения через scipy (solve_ivp):
       x'' = -GM * x / r^3,   y'' = -GM * y / r^3
2) Аналитическое решение через уравнение Кеплера:
       M = n * t,  M = E - e * sin(E)   (решается численно, при e = 0 E = M)
       tg(nu/2) = sqrt((1+e)/(1-e)) * tg(E/2),  r = a * (1 - e * cos(E))
3) Сравнение двух решений.

Используются безразмерные единицы: GM = 1, a = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


GM = 1.0          # гравитационный параметр
a = 1.0           # большая полуось (для круга - радиус орбиты)
e = 0.0           # эксцентриситет (круговая орбита)
n_orbits = 3      # сколько оборотов считаем
n_points = 100   # число точек по времени

n = np.sqrt(GM / a**3)       # среднее движение
T = 2 * np.pi / n            # период обращения
t_eval = np.linspace(0, n_orbits * T, n_points)


#scipy
def central_field(t, s):
    """s = [x, y, vx, vy]"""
    x, y, vx, vy = s
    r3 = (x * x + y * y) ** 1.5
    return [vx, vy, -GM * x / r3, -GM * y / r3]


# Начальные условия в перицентре. Для e = 0 это круговая скорость sqrt(GM/a)
r0 = a * (1 - e)
v0 = np.sqrt(GM * (1 + e) / (a * (1 - e)))
s0 = [r0, 0.0, 0.0, v0]

sol = solve_ivp(
    central_field, (0, t_eval[-1]), s0,
    t_eval=t_eval, method="DOP853", rtol=1e-12, atol=1e-12,
)
x_num, y_num = sol.y[0], sol.y[1]
r_num = np.hypot(x_num, y_num)


#Кеплер
def solve_kepler(M, ecc):
    """Решает E - ecc*sin(E) = M для каждого элемента массива M."""
    E = np.empty_like(M)
    for i, m in enumerate(M):
        m_red = np.mod(m, 2 * np.pi)          # приводим к [0, 2*pi)
        k = m - m_red
        E[i] = brentq(lambda En: En - ecc * np.sin(En) - m_red,
                      0.0, 2 * np.pi) + k
    return E


M = n * t_eval                                  # средняя аномалия
E = solve_kepler(M, e)                          # эксцентрическая аномалия
nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                    np.sqrt(1 - e) * np.cos(E / 2))   # истинная аномалия
r_kep = a * (1 - e * np.cos(E))
x_kep, y_kep = r_kep * np.cos(nu), r_kep * np.sin(nu)


# ---------------- 3. Сравнение ----------------
dist = np.hypot(x_num - x_kep, y_num - y_kep)   # расстояние между решениями

# энергия численного решения (должна сохраняться)
v2 = sol.y[2] ** 2 + sol.y[3] ** 2
energy = 0.5 * v2 - GM / r_num

print(f"Эксцентриситет e = {e},  a = {a},  период T = {T:.6f}")
print(f"Макс. отличие решений |Δr|:           {dist.max():.3e}")
print(f"Макс. отклонение радиуса от a (scipy): {np.abs(r_num - a).max():.3e}")
print(f"Дрейф энергии (scipy):                 {np.abs(energy - energy[0]).max():.3e}")


# ---------------- Графики ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].plot(x_num, y_num, "b-", lw=3, alpha=0.5, label="scipy (solve_ivp)")
ax[0].plot(x_kep, y_kep, "r--", lw=1.2, label="уравнение Кеплера")
ax[0].plot(0, 0, "yo", ms=12, label="центр притяжения")
ax[0].plot(x_num[0], y_num[0], "ks", label="старт")
ax[0].set_aspect("equal")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Орбита")
ax[0].grid(alpha=0.3)
ax[0].legend(loc="upper right", fontsize=8)

ax[1].semilogy(t_eval / T, dist + 1e-18, label="|r_scipy - r_Кеплер|")
ax[1].semilogy(t_eval / T, np.abs(r_num - a) + 1e-18, label="|r_scipy - a|")
ax[1].set_xlabel("время, периоды T")
ax[1].set_ylabel("ошибка")
ax[1].set_title("Расхождение решений")
ax[1].grid(alpha=0.3, which="both")
ax[1].legend()

ax[2].plot(t_eval / T, energy, label="энергия E (scipy)")
ax[2].axhline(-GM / (2 * a), color="r", ls="--", label="-GM/(2a)")
ax[2].set_xlabel("время, периоды T")
ax[2].set_ylabel("удельная энергия")
ax[2].set_title("Сохранение энергии")
ax[2].grid(alpha=0.3)
ax[2].legend()

fig.suptitle(f"Круговая орбита (e = {e}): scipy vs уравнение Кеплера", fontsize=14)
plt.tight_layout()
plt.savefig("kepler_orbit.png", dpi=150)
plt.show()