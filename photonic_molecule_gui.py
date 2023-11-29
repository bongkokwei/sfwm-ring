import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QSlider,
    QLabel,
    QHBoxLayout,
    QPushButton,
)
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from photonic_molecule import PhotonicMolecule
import util
import numpy as np


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.coupling = [0.23, 0.4, 0.4]
        self.fsr = [7e-9, 10e-9]
        self.loss = [3]  # dB/cm
        self.central_wavelength = [1550e-9]
        self.sliders = {}  # Dictionary to store sliders

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.add_slider(
            "kappa_1",
            attribute="coupling",
            variable=self.coupling,
            index=0,
            prefactor=1e4,
            default_value=self.coupling[0],
            max=10000,
            step=10000,
        )
        self.add_slider(
            "kappa_2",
            attribute="coupling",
            variable=self.coupling,
            index=1,
            prefactor=1e4,
            default_value=self.coupling[1],
            max=10000,
            step=10000,
        )
        self.add_slider(
            "kappa_3",
            attribute="coupling",
            variable=self.coupling,
            index=2,
            prefactor=1e4,
            default_value=self.coupling[2],
            max=10000,
            step=10000,
        )
        self.add_slider(
            "primary fsr (nm)",
            attribute="fsr",
            variable=self.fsr,
            index=0,
            prefactor=1e9,
            default_value=self.fsr[0],
            min=1,
            max=50,
        )
        self.add_slider(
            "auxiliary fsr (nm)",
            variable=self.fsr,
            attribute="fsr",
            index=1,
            prefactor=1e9,
            default_value=self.fsr[1],
            min=1,
            max=50,
        )
        self.add_slider(
            "loss (dB/cm)",
            variable=self.loss,
            attribute="loss",
            index=0,
            prefactor=1e2,
            default_value=self.loss[0],
            min=0,
            max=600,
        )
        self.add_slider(
            "wavelength (nm)",
            variable=self.central_wavelength,
            attribute="central_wavelength",
            index=0,
            prefactor=1e9,
            default_value=self.central_wavelength[0],
            min=1500,
            max=1600,
        )

        # Create Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.update_plot()
        self.layout.addWidget(self.canvas)

        self.setGeometry(0, 0, 800, 1000)
        self.setWindowTitle("Photonic Molecule MRR")
        self.show()

    def add_slider(
        self,
        label_str,
        min=0,
        max=100,
        step=1,
        prefactor=1,  # prefactor for default value
        default_value=50,
        variable=None,
        index=None,
        attribute=None,
    ):
        default_int = int(default_value * prefactor)
        slider_layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setSingleStep(step)
        slider.setValue(default_int)

        label = QLabel(label_str + f": {slider.value()}")
        slider_layout.addWidget(label)
        slider_layout.addWidget(slider)

        # Connect the valueChanged signal to the update_value method
        slider.valueChanged.connect(
            lambda value, label=label, variable=variable, index=index, attribute=attribute, prefactor=prefactor: self.update_value(
                value,
                label,
                variable,
                index,
                attribute,
                prefactor,
            )
        )

        # Store the slider in the dictionary
        key = label_str.replace(" ", "_").lower()
        self.sliders[key] = {
            "slider": slider,
            "label": label,
            "variable": variable,
            "index": index,
            "attribute": attribute,
        }

        self.layout.addLayout(slider_layout)
        slider.valueChanged.connect(self.update_plot)

    def update_value(self, value, label, variable, index, attribute, prefactor):
        if variable is not None:
            variable[index] = value / prefactor  # Assuming the variable is a list
        elif attribute is not None:
            setattr(self, attribute, value)  # Set attribute value

        label.setText(f"{label.text().split(':')[0]}: {value}")

    def update_plot(self):
        self.layout.addWidget(self.canvas)
        self.ax.clear()

        wavelength = np.linspace(1550e-9 - 15e-9, 1550e-9 + 15e-9, num=10000)
        wavevector = util.wavevector(wavelength)

        pm = PhotonicMolecule(
            self.coupling,
            self.fsr,
            self.loss[0],
            self.central_wavelength[0],
        )
        E_out = pm.E_out(wavevector)

        self.ax.plot(wavelength * 1e9, np.abs(E_out) ** 2)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Transmission (arb unit)")
        self.ax.grid(True)

        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main_window = MainWindow()
    sys.exit(app.exec_())
