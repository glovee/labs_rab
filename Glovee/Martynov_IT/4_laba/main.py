import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QComboBox, QGroupBox, QFormLayout, QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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


# ---------------------------------------------------------------- данные

def fix_leap_second(series, threshold=0.8):
    result = series.copy()
    result[result > threshold] -= 1
    result[result < -threshold] += 1
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
        df = df[["MJD", "Year", "Month", "Day", "Type", "x_pole", "y_pole", "UT1-UTC"]].copy()

        df = df[df["Type"] == "prediction"].copy()
        if df.empty:
            continue

        first = df.iloc[0]
        T_mjd = int(first["MJD"])
        T_date = pd.Timestamp(
            year=int(first["Year"]),
            month=int(first["Month"]),
            day=int(first["Day"]),
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
    """Объединяет прогнозы с C04 по MJD и вычисляет «сырые» ошибки forecast - c04.
    Поправка на leap second применяется позже, в GUI (порог настраивается ползунком)."""
    df_c04 = df_c04.drop_duplicates(subset=["MJD"])

    df = pd.merge(
        df_forecast,
        df_c04,
        on="MJD",
        how="inner",
        suffixes=("_forecast", "_c04"),
    )

    for param in PARAMS.values():
        df[f"error_{param}"] = df[f"{param}_forecast"] - df[f"{param}_c04"]

    df = df.dropna(subset=["lead_time"])
    df["lead_time"] = df["lead_time"].astype(int)
    return df


# ---------------------------------------------------------------- GUI

def make_slider(minimum, maximum, value):
    s = QSlider(Qt.Horizontal)
    s.setRange(minimum, maximum)
    s.setValue(value)
    return s


def row(slider, label):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(slider, stretch=1)
    label.setMinimumWidth(90)
    lay.addWidget(label)
    return w


class MainWindow(QWidget):
    def __init__(self, df_errors: pd.DataFrame):
        super().__init__()
        self.setWindowTitle("Ошибки прогноза Bulletin A относительно C04")
        self.resize(1300, 800)

        self.df_errors = df_errors
        self.t_dates = sorted(df_errors["T_date"].unique())
        max_lead = int(df_errors["lead_time"].max())

        # --- панель управления
        box = QGroupBox("Параметры")
        form = QFormLayout(box)

        self.combo = QComboBox()
        self.combo.addItems(list(PARAMS.keys()))
        form.addRow("Параметр:", self.combo)

        self.sl_lead = make_slider(1, max_lead, max_lead)
        self.lbl_lead = QLabel()
        form.addRow("Макс. длительность прогноза:", row(self.sl_lead, self.lbl_lead))

        n = len(self.t_dates)
        self.sl_t0 = make_slider(0, n - 1, 0)
        self.lbl_t0 = QLabel()
        form.addRow("Начало T:", row(self.sl_t0, self.lbl_t0))

        self.sl_t1 = make_slider(0, n - 1, n - 1)
        self.lbl_t1 = QLabel()
        form.addRow("Конец T:", row(self.sl_t1, self.lbl_t1))

        # --- график
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(self.canvas, stretch=1)

        # Отложенная перерисовка, чтобы не рисовать на каждый тик ползунка
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_plot)

        self.combo.currentIndexChanged.connect(self.on_change)
        for s in (self.sl_lead, self.sl_t0, self.sl_t1):
            s.valueChanged.connect(self.on_change)

        self.update_labels()
        self.update_plot()

    def update_labels(self):
        self.lbl_lead.setText(f"{self.sl_lead.value()} дн.")
        self.lbl_t0.setText(self.t_dates[self.sl_t0.value()])
        self.lbl_t1.setText(self.t_dates[self.sl_t1.value()])

    def on_change(self):
        # начало T не должно быть позже конца T
        if self.sl_t0.value() > self.sl_t1.value():
            sender = self.sender()
            if sender is self.sl_t0:
                self.sl_t1.setValue(self.sl_t0.value())
            else:
                self.sl_t0.setValue(self.sl_t1.value())
        self.update_labels()
        self.timer.start()

    def update_plot(self):
        param = self.combo.currentText()
        col = PARAMS[param]

        t0 = self.t_dates[self.sl_t0.value()]
        t1 = self.t_dates[self.sl_t1.value()]
        max_lead = self.sl_lead.value()

        df = self.df_errors
        df = df[
            (df["lead_time"] <= max_lead)
            & (df["T_date"] >= t0)
            & (df["T_date"] <= t1)
        ].copy()

        err = df[f"error_{col}"]
        if param == "UT1-UTC":
            err = fix_leap_second(err)
        df["err"] = err

        self.fig.clear()

        if df.empty:
            self.canvas.draw_idle()
            return

        matrix = df.pivot_table(
            index="lead_time", columns="T_date", values="err", aggfunc="mean"
        ).sort_index(axis=0).sort_index(axis=1)

        ax = self.fig.add_subplot(111)
        step_y = max(1, len(matrix.index) // 20)
        step_x = max(1, len(matrix.columns) // 60)

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": f"Ошибка {param}"},
            xticklabels=step_x,
            yticklabels=step_y,
        )

        ax.set_title(f"Ошибка прогноза {param} (прогноз - C04)")
        ax.set_xlabel("Дата начала прогноза T")
        ax.set_ylabel("Длительность прогноза, дни")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)

    try:
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
    except Exception as e:
        QMessageBox.critical(None, "Ошибка загрузки данных", str(e))
        sys.exit(1)

    win = MainWindow(df_errors)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()