import numpy as np


def J(x):
    """Функция Химмельблау."""
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def grad_J(x):
    """Градиент функции Химмельблау."""
    x1, x2 = x
    a = x1**2 + x2 - 11
    b = x1 + x2**2 - 7
    dx1 = 4 * x1 * a + 2 * b
    dx2 = 2 * a + 4 * x2 * b
    return np.array([dx1, dx2])


def gradient_descent(x0, h=0.01, eps=1e-8, max_iter=100000):
    """Градиентный спуск с постоянным шагом h:
    x^{k+1} = x^k - h * grad J(x^k).
    Остановка: ||grad J(x^k)|| < eps или k == max_iter.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for k in range(max_iter):
        g = grad_J(x)
        if np.linalg.norm(g) < eps:
            break
        x = x - h * g
        history.append(x.copy())
    return x, k, np.array(history)


if __name__ == "__main__":
    x0 = (0.0, 0.0)
    h = 0.01

    x_min, iters, hist = gradient_descent(x0, h=h)

    print(f"Начальная точка: x0 = {x0}")
    print(f"Шаг h = {h}")
    print(f"Число итераций: {iters}")
    print(f"Найденная точка минимума: x* = ({x_min[0]:.8f}, {x_min[1]:.8f})")
    print(f"J(x*) = {J(x_min):.3e}")
    print(f"||grad J(x*)|| = {np.linalg.norm(grad_J(x_min)):.3e}")

    # Необязательный график траектории
    try:
        import matplotlib.pyplot as plt

        xs = np.linspace(-5, 5, 400)
        X1, X2 = np.meshgrid(xs, xs)
        Z = J((X1, X2))
        plt.contour(X1, X2, Z, levels=np.logspace(-1, 3.5, 25), cmap="viridis")
        plt.plot(hist[:, 0], hist[:, 1], "r.-", markersize=3, label="траектория")
        plt.plot(*x0, "go", label="x0")
        plt.plot(*x_min, "r*", markersize=12, label="x*")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.title("Градиентный спуск, функция Химмельблау")
        plt.savefig("grafiki/himmelblau_path.png", dpi=120)
    except ImportError:
        pass