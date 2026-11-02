import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Пути к данным
BASE_DIR = Path(__file__).parent
BULLETIN_DIR = BASE_DIR / "iers_bulletin_a"
C04_PATH = BASE_DIR / "eopc04_20.1962-now.csv"

# Параметры, для которых считаем ошибки
PARAMS = {
    "x_pole": "x_pole",
    "y_pole": "y_pole",
    "UT1-UTC": "UT1-UTC",
}




def fix_leap_second(series):
    result = series.copy()
    result[result > 0.8] -= 1
    result[result < -0.8] += 1
    return result





def load_c04(path: Path) -> pd.DataFrame:
    """Читает каталог C04 и возвращает DataFrame с MJD, x_pole, y_pole, UT1-UTC."""
    df = pd.read_csv(path, sep=";", decimal=".")
    df = df[["MJD", "x_pole", "y_pole", "UT1-UTC"]].copy()

    df["MJD"] = pd.to_numeric(df["MJD"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["MJD"])

    for col in ["x_pole", "y_pole", "UT1-UTC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_bulletins(folder: Path) -> pd.DataFrame:
    """Читает все бюллетени Bulletin A и собирает прогнозные строки в один DataFrame."""
    files = sorted(folder.glob("bulletina-xxvii-*.csv"))
    if not files:
        raise FileNotFoundError(f"Не найдены файлы в {folder}")

    forecasts = []

    for file in files:
        df = pd.read_csv(file, sep=";", decimal=".")

        # Берём только нужные колонки. Первый Type — это статус rapid/prediction.
        df = df[["MJD", "Year", "Month", "Day", "Type", "x_pole", "y_pole", "UT1-UTC"]].copy()

        # Оставляем только прогнозные строки
        df = df[df["Type"] == "prediction"].copy()
        if df.empty:
            continue

        # T — дата начала прогноза (первая прогнозная строка в файле)
        first = df.iloc[0]
        T_mjd = int(first["MJD"])
        T_date = pd.Timestamp(
            year=int(first["Year"]),
            month=int(first["Month"]),
            day=int(first["Day"])
        )

        df["T_mjd"] = T_mjd
        df["T_date"] = T_date.strftime("%Y-%m-%d")
        df["lead_time"] = df["MJD"] - T_mjd

        forecasts.append(df)

    if not forecasts:
        raise ValueError("Не найдено ни одной строки с Type == 'prediction'")

    df_forecast = pd.concat(forecasts, ignore_index=True)

    for col in ["MJD", "lead_time", "T_mjd"]:
        df_forecast[col] = pd.to_numeric(df_forecast[col], errors="coerce").astype("Int64")

    for col in ["x_pole", "y_pole", "UT1-UTC"]:
        df_forecast[col] = pd.to_numeric(df_forecast[col], errors="coerce")

    return df_forecast


def compute_errors(df_forecast: pd.DataFrame, df_c04: pd.DataFrame) -> pd.DataFrame:
    """Объединяет прогнозы с C04 по MJD и вычисляет ошибки forecast - c04."""
    df_c04 = df_c04.drop_duplicates(subset=["MJD"])

    df = pd.merge(
        df_forecast,
        df_c04,
        on="MJD",
        how="inner",
        suffixes=("_forecast", "_c04")
    )

    for param in PARAMS.values():
        df[f"error_{param}"] = df[f"{param}_forecast"] - df[f"{param}_c04"]

    # Работаем с leap second
    df["error_UT1-UTC"] = fix_leap_second(df["error_UT1-UTC"])

    return df


def make_matrix(df_errors: pd.DataFrame, param: str) -> pd.DataFrame:
    """Строит матрицу ошибок: index = lead_time, columns = T_date."""
    matrix = df_errors.pivot_table(
        index="lead_time",
        columns="T_date",
        values=f"error_{param}",
        aggfunc="mean"
    )

    matrix = matrix.sort_index(axis=0)   # сортировка по lead_time
    matrix = matrix.sort_index(axis=1)   # сортировка по дате T

    return matrix


def plot_heatmap(matrix: pd.DataFrame, title: str, param: str) -> None:
    """Рисует тепловую карту ошибок."""
    plt.figure(figsize=(16, 8))

    step_y = max(1, len(matrix.index) // 20)

    sns.heatmap(
        matrix,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": f"Ошибка {param}"},
        xticklabels=1,
        yticklabels=step_y,
    )

    plt.title(title)
    plt.xlabel("Дата начала прогноза T")
    plt.ylabel("Длительность прогноза, дни")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


print("Загрузка C04...")
df_c04 = load_c04(C04_PATH)
print(f"C04: {len(df_c04)} строк, MJD от {df_c04['MJD'].min()} до {df_c04['MJD'].max()}")

print("Загрузка бюллетеней...")
df_forecast = load_bulletins(BULLETIN_DIR)
print(
    f"Прогнозы: {len(df_forecast)} строк, "
    f"T от {df_forecast['T_date'].min()} до {df_forecast['T_date'].max()}"
)

print("Вычисление ошибок...")
df_errors = compute_errors(df_forecast, df_c04)
print(f"Объединённые данные: {len(df_errors)} строк")


for param, col in PARAMS.items():
    print(f"Построение heatmap для {param}...")
    matrix = make_matrix(df_errors, col)
    plot_heatmap(
        matrix,
        f"Ошибка прогноза {param} (прогноз - C04)",
        param
    )