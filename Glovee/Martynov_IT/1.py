import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_sphere(radius: float, n_points: int, seed: int | None = None):
    """Возвращает (оценка объёма, точки, маска попадания в шар)."""
    rng = np.random.default_rng(seed)

    # N равномерных точек в кубе [-R, R]^3
    points = rng.uniform(-radius, radius, size=(n_points, 3))

    # Попадание внутрь шара
    inside = np.sum(points ** 2, axis=1) <= radius ** 2

    cube_volume = (2 * radius) ** 3
    volume = cube_volume * np.count_nonzero(inside) / n_points
    return volume, points, inside


def main():
    radius = 1.0
    n_points = 100_000
    seed = 42

    volume, points, inside = monte_carlo_sphere(radius, n_points, seed)
    exact = 4 / 3 * np.pi * radius ** 3
    error = abs(volume - exact)

    print(f"Радиус сферы:           R = {radius}")
    print(f"Число точек:            N = {n_points}")
    print(f"Объём (Монте-Карло):    V = {volume:.6f}")
    print(f"Объём (точный):         V = {exact:.6f}")
    print(f"Абсолютная погрешность: {error:.6f}")
    print(f"Относительная погрешн.: {error / exact * 100:.3f} %")

    # ---------- Визуализация ----------
    fig = plt.figure(figsize=(14, 6))

    # 1) 3D-график: часть точек (чтобы не тормозило)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    n_show = 3000
    p, ins = points[:n_show], inside[:n_show]
    ax1.scatter(*p[ins].T, s=3, c="tab:red", alpha=0.7, label="внутри шара")
    ax1.scatter(*p[~ins].T, s=3, c="tab:blue", alpha=0.25, label="снаружи")
    ax1.set_title(f"Точки Монте-Карло (показано {n_show} из {n_points})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_box_aspect((1, 1, 1))
    ax1.legend(loc="upper left")

    # 2) Сходимость оценки объёма при росте N
    ax2 = fig.add_subplot(1, 2, 2)
    counts = np.cumsum(inside)
    n_range = np.arange(1, n_points + 1)
    estimates = (2 * radius) ** 3 * counts / n_range
    ax2.plot(n_range, estimates, lw=1, label="оценка Монте-Карло")
    ax2.axhline(exact, color="red", ls="--", label=f"точное: {exact:.4f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("число точек N")
    ax2.set_ylabel("объём V")
    ax2.set_title(f"Сходимость: V = {volume:.4f}, ошибка = {error:.4f}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Объём сферы методом Монте-Карло", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()