"""Главное окно приложения (PyQt5)."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
                             QMainWindow, QSlider, QTabWidget, QVBoxLayout, QWidget,
                             QGridLayout)

import analysis
import config
from model import ControlSystem
from plots import NyquistCanvas, StepCanvas

SLIDER_STEPS = 1000


class PidControl(QWidget):
    """Ползунок + числовое поле для одного коэффициента."""

    def __init__(self, name, lo, hi, value, on_change):
        super().__init__()
        self.lo, self.hi = lo, hi
        self.on_change = on_change

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(0.0, 1e6)  # в поле можно ввести значение и за пределами ползунка
        self.spin.setSingleStep((hi - lo) / 100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(name))
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.spin.setValue(value)
        self._sync_slider_from_spin()
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)

    def value(self):
        return self.spin.value()

    def _slider_moved(self, pos):
        v = self.lo + (self.hi - self.lo) * pos / SLIDER_STEPS
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.on_change()

    def _spin_changed(self, _):
        self._sync_slider_from_spin()
        self.on_change()

    def _sync_slider_from_spin(self):
        frac = (self.spin.value() - self.lo) / (self.hi - self.lo)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(min(max(frac, 0.0), 1.0) * SLIDER_STEPS)))
        self.slider.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self, variant=config.DEFAULT_VARIANT):
        super().__init__()
        self.setWindowTitle("PID-регулятор: переходный процесс и устойчивость")
        self.resize(1100, 700)
        self.variant = config.VARIANTS[variant]

        # --- панель управления ---
        self.controls = {}
        box = QGroupBox("Коэффициенты PID-регулятора")
        v = QVBoxLayout(box)
        for key, label in (("kp", "Kp"), ("ki", "Ки"), ("kd", "Кд")):
            lo, hi = config.SLIDER_RANGES[key]
            ctrl = PidControl(label, lo, hi, config.DEFAULT_PID[key], self.recalculate)
            self.controls[key] = ctrl
            v.addWidget(ctrl)

        # --- показатели ---
        res_box = QGroupBox("Показатели переходного процесса")
        g = QGridLayout(res_box)
        self.lbl_stable = QLabel("—")
        self.lbl_over = QLabel("—")
        self.lbl_final = QLabel("—")
        self.lbl_ts = QLabel("—")
        for row, (txt, lbl) in enumerate((
                ("Устойчивость:", self.lbl_stable),
                ("Перерегулирование:", self.lbl_over),
                ("Установившееся значение:", self.lbl_final),
                (f"Время переходного процесса ({config.SETTLING_BAND * 100:.0f}%):", self.lbl_ts))):
            g.addWidget(QLabel(txt), row, 0)
            g.addWidget(lbl, row, 1)

        nyq_box = QGroupBox("Критерий Найквиста")
        nv = QVBoxLayout(nyq_box)
        self.lbl_nyquist = QLabel("—")
        self.lbl_nyquist.setWordWrap(True)
        nv.addWidget(self.lbl_nyquist)

        left = QVBoxLayout()
        left.addWidget(box)
        left.addWidget(res_box)
        left.addWidget(nyq_box)
        left.addStretch(1)
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(420)

        # --- графики ---
        self.step_canvas = StepCanvas()
        self.nyq_canvas = NyquistCanvas()
        tabs = QTabWidget()
        tabs.addTab(self.step_canvas, "Переходный процесс")
        tabs.addTab(self.nyq_canvas, "Годограф (Найквист)")

        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(left_w)
        root.addWidget(tabs, 1)
        self.setCentralWidget(central)

        self.recalculate()

    def build_system(self):
        c = self.controls
        return ControlSystem(self.variant["plant"], self.variant["actuator"],
                             c["kp"].value(), c["ki"].value(), c["kd"].value())

    def recalculate(self):
        system = self.build_system()
        band = config.SETTLING_BAND

        t, y, info = analysis.compute_step_metrics(system, band)
        self.step_canvas.update_plot(t, y, info, band)

        w, W = analysis.nyquist_data(system)
        margins = analysis.stability_margins(system)
        self.nyq_canvas.update_plot(W, margins)

        # Подписи
        self.lbl_stable.setText("устойчива" if info["stable"] else "неустойчива")
        if info["stable"]:
            self.lbl_over.setText(f"{info['overshoot']:.2f} %")
            self.lbl_final.setText(f"{info['y_final']:.4f}")
            ts = info["t_settle"]
            self.lbl_ts.setText(f"{ts:.2f} c" if ts is not None else "не установился")
        else:
            self.lbl_over.setText("—")
            self.lbl_final.setText("—")
            self.lbl_ts.setText("—")
        self.lbl_nyquist.setText(analysis.nyquist_verdict(system, margins))
