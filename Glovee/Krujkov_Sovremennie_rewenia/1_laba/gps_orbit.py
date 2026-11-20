#!/usr/bin/env python3
"""
Чтение навигационного кадра RINEX 3.02 (GPS) и расчёт траектории спутника
по broadcast-эфемеридам (алгоритм: ICD-GPS-200 / Seeber, Applied GPS App.3).

Использование:
    python3 gps_orbit.py Brdc0020.25n 5 "2025-01-02 00:00:00"
"""
import sys
import math
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GM = 3.986005e14            # м^3/с^2, WGS-84
OMEGA_E = 7.292115e-5       # рад/с, WGS-84
PI = 3.1415926535898
GPS_EPOCH = datetime(1980, 1, 6)
WEEK_SEC = 604800.0

# Имена полей в порядке следования в записи (после 3 коэффициентов часов).
_ORBIT_FIELDS = [
    "IODE", "Crs", "dn", "M0",
    "Cuc", "e", "Cus", "sqrtA",
    "Toe", "Cic", "Omega0", "Cis",
    "i0", "Crc", "omega", "OmegaDot",
    "IDOT", "L2codes", "week", "L2Pflag",
    "SVacc", "health", "TGD", "IODC",
    "ttx", "fit",
]


def _f(s):
    """Число из RINEX-поля (экспонента через D); пустое поле -> 0.0."""
    s = s.strip()
    return float(s.replace("D", "E").replace("d", "e")) if s else 0.0


# ---------------------------------------------------------------------------
# 1. Чтение навигационного файла
# ---------------------------------------------------------------------------
def read_nav_file(path, sat, t_eph):
    """
    Считывает RINEX-3 навигационный файл и возвращает эфемериды спутника.

    path  : путь к файлу (RINEX 3.xx, GPS)
    sat   : номер спутника GPS (int, 1..32) или строка вида 'G05'
    t_eph : время эфемерид (datetime, шкала GPS) — выбирается запись,
            у которой эпоха Toc ближе всего к t_eph

    Возвращает (eph, records):
        eph     - словарь параметров выбранной записи
        records - все записи этого спутника из файла (для расчёта на сутки)
    """
    prn = f"G{sat:02d}" if isinstance(sat, int) else sat.upper()

    with open(path, "r", encoding="ascii", errors="replace") as fh:
        lines = fh.read().splitlines()

    # пропуск заголовка
    i = 0
    while i < len(lines) and "END OF HEADER" not in lines[i]:
        i += 1
    i += 1

    records = []
    while i < len(lines):
        line = lines[i]
        if len(line) >= 3 and line[0] in "GRECJSI" and line[1:3].isdigit():
            # GPS-запись: строка эпохи + 7 строк орбиты = 8 строк
            n_lines = 8 if line[0] in "GEC" or line[0] == "J" else 4
            if line[:3] == prn:
                block = lines[i:i + n_lines]
                records.append(_parse_gps_record(block))
            i += n_lines
        else:
            i += 1

    if not records:
        raise ValueError(f"Спутник {prn} не найден в {path}")

    eph = select_ephemeris(records, t_eph)
    return eph, records


def _parse_gps_record(block):
    l0 = block[0]
    y, mo, d, h, mi, s = (int(x) for x in l0[4:23].split())
    rec = {
        "prn": l0[:3],
        "toc": datetime(y, mo, d, h, mi, s),
        "a0": _f(l0[23:42]),
        "a1": _f(l0[42:61]),
        "a2": _f(l0[61:80]),
    }
    vals = []
    for ln in block[1:]:
        ln = ln.ljust(79)
        for k in range(4):                 # поля по 19 символов, отступ 4
            vals.append(_f(ln[4 + 19 * k: 4 + 19 * (k + 1)]))
    for name, v in zip(_ORBIT_FIELDS, vals):
        rec[name] = v
    rec["week"] = int(rec["week"])
    # момент Toe в календарном времени (GPS)
    rec["toe_dt"] = GPS_EPOCH + timedelta(weeks=rec["week"], seconds=rec["Toe"])
    return rec


def select_ephemeris(records, t):
    """Запись с минимальным |t - Toe|."""
    return min(records, key=lambda r: abs((t - r["toe_dt"]).total_seconds()))


# ---------------------------------------------------------------------------
# 2. Положение спутника в ECEF (WGS-84)
# ---------------------------------------------------------------------------
def sat_position(eph, t):
    """ECEF-координаты (м) спутника на момент t (datetime, шкала GPS)."""
    tk = (t - eph["toe_dt"]).total_seconds()        # время от Toe, c

    A = eph["sqrtA"] ** 2
    e = eph["e"]
    n = math.sqrt(GM / A ** 3) + eph["dn"]          # скорректированное ср. движение
    M = eph["M0"] + n * tk

    E = M                                           # уравнение Кеплера
    for _ in range(30):
        E_new = M + e * math.sin(E)
        if abs(E_new - E) < 1e-13:
            E = E_new
            break
        E = E_new

    sinE, cosE = math.sin(E), math.cos(E)
    nu = math.atan2(math.sqrt(1 - e * e) * sinE, cosE - e)   # истинная аномалия
    phi = nu + eph["omega"]                                   # аргумент широты

    s2, c2 = math.sin(2 * phi), math.cos(2 * phi)
    du = eph["Cuc"] * c2 + eph["Cus"] * s2
    dr = eph["Crc"] * c2 + eph["Crs"] * s2
    di = eph["Cic"] * c2 + eph["Cis"] * s2

    u = phi + du
    r = A * (1 - e * cosE) + dr
    i = eph["i0"] + eph["IDOT"] * tk + di

    xp, yp = r * math.cos(u), r * math.sin(u)
    Om = eph["Omega0"] + (eph["OmegaDot"] - OMEGA_E) * tk - OMEGA_E * eph["Toe"]

    x = xp * math.cos(Om) - yp * math.cos(i) * math.sin(Om)
    y = xp * math.sin(Om) + yp * math.cos(i) * math.cos(Om)
    z = yp * math.sin(i)
    return np.array([x, y, z])


def ecef_to_geodetic(x, y, z):
    """ECEF -> (широта°, долгота°, высота м), WGS-84."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(6):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return np.degrees(lat), np.degrees(lon), h


# ---------------------------------------------------------------------------
# 3. Траектория за сутки с шагом 10 минут
# ---------------------------------------------------------------------------
def trajectory(records, t_start, hours=24, step_min=10):
    """Для каждой эпохи берётся ближайшая по Toe эфемерида (они действуют ~2 ч)."""
    n = int(hours * 60 / step_min) + 1
    times = [t_start + timedelta(minutes=step_min * k) for k in range(n)]
    xyz = np.array([sat_position(select_ephemeris(records, t), t) for t in times])
    return times, xyz


def plot_trajectory(times, xyz, prn, path):
    lat, lon, h = ecef_to_geodetic(*xyz.T)
    hrs = [(t - times[0]).total_seconds() / 3600 for t in times]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Спутник {prn}: траектория {times[0]:%Y-%m-%d %H:%M} – "
                 f"{times[-1]:%Y-%m-%d %H:%M} (GPST), шаг 10 мин", fontsize=13)

    # трасса на поверхности (разрыв линии при переходе через ±180°)
    ax1 = fig.add_subplot(2, 2, 1)
    lon_p = lon.copy().astype(float)
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0]
    lon_p, lat_p = list(lon), list(lat)
    for j in reversed(jumps):
        lon_p.insert(j + 1, np.nan)
        lat_p.insert(j + 1, np.nan)
    ax1.plot(lon_p, lat_p, "-", color="0.6", lw=1)
    sc = ax1.scatter(lon, lat, c=hrs, cmap="viridis", s=18, zorder=3)
    ax1.plot(lon[0], lat[0], "r^", ms=10, label="начало")
    ax1.plot(lon[-1], lat[-1], "ks", ms=7, label="конец")
    ax1.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="долгота, °",
            ylabel="широта, °", title="Трасса (подспутниковая точка)")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower left")
    fig.colorbar(sc, ax=ax1, label="часы от начала")

    # 3D в ECEF
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    R = 6378.137
    ax2.plot_surface(R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v),
                     R * np.cos(v), color="lightblue", alpha=0.35, lw=0)
    k = xyz / 1e3
    ax2.plot(k[:, 0], k[:, 1], k[:, 2], "-", color="0.5", lw=0.8)
    ax2.scatter(k[:, 0], k[:, 1], k[:, 2], c=hrs, cmap="viridis", s=14)
    ax2.plot(*k[0], "r^", ms=9)
    ax2.set(xlabel="X, км", ylabel="Y, км", zlabel="Z, км",
            title="Орбита в ECEF (WGS-84)")
    ax2.set_box_aspect((1, 1, 1))
    lim = 27000
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); ax2.set_zlim(-lim, lim)

    # координаты от времени
    ax3 = fig.add_subplot(2, 2, 3)
    for j, name in enumerate("XYZ"):
        ax3.plot(hrs, xyz[:, j] / 1e3, ".-", ms=4, label=name)
    ax3.set(xlabel="часы от начала", ylabel="км", title="Координаты ECEF")
    ax3.grid(alpha=0.3); ax3.legend(ncol=3)

    # радиус и высота
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(hrs, np.linalg.norm(xyz, axis=1) / 1e3, ".-", ms=4, color="C3")
    ax4.set(xlabel="часы от начала", ylabel="геоцентрический радиус, км",
            title="Расстояние до центра Земли")
    ax4.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv):
    nav = argv[1] if len(argv) > 1 else "Brdc0020.25n"
    sat = int(argv[2]) if len(argv) > 2 else 5
    t_eph = (datetime.fromisoformat(argv[3]) if len(argv) > 3
             else datetime(2025, 1, 2, 0, 0, 0))

    eph, records = read_nav_file(nav, sat, t_eph)
    print(f"{eph['prn']}: найдено записей {len(records)}; "
          f"выбрана Toc={eph['toc']}, Toe={eph['toe_dt']} (неделя {eph['week']}, "
          f"Toe={eph['Toe']:.0f} c), IODE={eph['IODE']:.0f}, health={eph['health']:.0f}")

    # сутки от начала суток времени эфемерид
    t0 = t_eph.replace(hour=0, minute=0, second=0, microsecond=0)
    times, xyz = trajectory(records, t0)

    out_csv = f"{eph['prn']}_{t0:%Y%m%d}.csv"
    lat, lon, h = ecef_to_geodetic(*xyz.T)
    with open(out_csv, "w") as fh:
        fh.write("time_gpst,X_m,Y_m,Z_m,lat_deg,lon_deg,h_m\n")
        for t, p, la, lo, hh in zip(times, xyz, lat, lon, h):
            fh.write(f"{t:%Y-%m-%d %H:%M:%S},{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},"
                     f"{la:.6f},{lo:.6f},{hh:.3f}\n")

    out_png = f"{eph['prn']}_{t0:%Y%m%d}.png"
    plot_trajectory(times, xyz, eph["prn"], out_png)
    print("точек:", len(times), "->", out_csv, out_png)
    return eph, records, times, xyz


if __name__ == "__main__":
    main(sys.argv)
