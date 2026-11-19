# -*- coding: utf-8 -*-
"""Запуск всего: расчёт -> графики -> таблица."""
import subprocess, sys
for s in ["penalty.py", "plot_overview.py", "plot_zoom.py", "plot_surface.py",
          "plot_convergence.py", "make_table.py"]:
    print(f"=== {s} ===", flush=True)
    subprocess.run([sys.executable, s], check=True)
