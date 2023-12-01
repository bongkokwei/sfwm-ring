"""
Making it more general so I don't have to rewrite GUI with each new ring resonators
TODO: A more elegant way to import new mrr into the GUI
"""


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
from scipy.signal import find_peaks, peak_widths


class MRR_GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.default_param = {
            "coupling_coeff": {
                "prefactor": 1e4,
                "slider_properties": (0, 1e4, 1),  # (min, max, step)
                "values": [(0.23, "kappa_1"), (0.4, "kappa_2"), (0.4, "kappa_mzi")],
            },
            "fsr": {
                "prefactor": 1e9,
                "slider_properties": (1, 50, 1),  # (min, max, step)
                "values": [(5e-9, "primary ring (nm)"), (10e-9, "auxiliary ring (nm)")],
            },
            "loss": {
                "prefactor": 1e2,
                "slider_properties": (0, 600, 1),  # (min, max, step)
                "values": [(3, "loss (dB/cm)")],
            },
            "central_wavelength": {
                "prefactor": 1e9,
                "slider_properties": (1300, 1700, 1),
                "values": [(1550e-9, "wavelength (nm)")],
            },
        }

        self.sliders = {}  # Dictionary to store sliders
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        for key, data in self.default_param.items():
            prefactor = data["prefactor"]
            value_list = data["values"]
            slider_properties = data["slider_properties"]
            for value in value_list:
                value, description = value
                # Add slider for all the parameters
                self.add_slider(
                    label_str=description,
                    attribute=key,
                    prefactor=prefactor,
                    default_value=value,
                    slider_properties=slider_properties,
                )

        # Create Matplotlib figure and canvas
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.canvas = FigureCanvas(self.fig)
        self.update_plot()
        self.layout.addWidget(self.canvas)

        self.setGeometry(0, 0, 1280, 720)
        self.setWindowTitle("Micro-ring resonator GUI")
        self.show()

    def add_slider(
        self,
        label_str,
        slider_properties=(0, 100, 1),
        prefactor=1,  # prefactor for default value
        default_value=50,
        variable=None,
        index=None,
        attribute=None,
    ):
        min, max, step = slider_properties
        slider_layout = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min))
        slider.setMaximum(int(max))
        slider.setSingleStep(int(step))
        slider.setValue(int(default_value * prefactor))

        qt_label = QLabel(label_str + f": {slider.value()}")  # to differentiate
        qt_label.setTextFormat(Qt.RichText)
        slider_layout.addWidget(qt_label)
        slider_layout.addWidget(slider)

        # Connect the valueChanged signal to the update_value method
        slider.valueChanged.connect(
            lambda value, label=label_str, attribute=attribute, prefactor=prefactor: self.change_param_values(
                value,
                attribute,  # main key of param dictionary
                label,  # description
                prefactor,
            )
        )
        slider.valueChanged.connect(  # To update label in GUI
            lambda value: qt_label.setText(f"{qt_label.text().split(':')[0]}: {value}")
        )
        # slider.valueChanged.connect(self.update_plot)
        slider.sliderReleased.connect(self.update_plot)
        # Store the slider in the dictionary
        key = label_str.replace(" ", "_").lower()
        self.sliders[key] = {
            "slider": slider,
            "label": qt_label,
            "variable": variable,
            "index": index,
            "attribute": attribute,
        }

        self.layout.addLayout(slider_layout)

    def update_plot(self):
        self.layout.addWidget(self.canvas)
        self.ax1.clear()
        self.ax2.clear()

        ################################################################################
        # Self defined plotting function
        ################################################################################
        wavelength, step = np.linspace(
            1550e-9 - 15e-9, 1550e-9 + 15e-9, num=5000, retstep=True
        )
        wavevector = util.wavevector(wavelength)

        pm = PhotonicMolecule(
            self.extract_attributes_values("coupling_coeff"),
            self.extract_attributes_values("fsr"),
            self.extract_attributes_values("loss")[0],
            self.extract_attributes_values("central_wavelength")[0],
        )
        transmission_spectrum = np.abs(pm.E_out(wavevector)) ** 2
        # Find minima positions
        minima_positions, _ = find_peaks(1 - transmission_spectrum)
        # Calculate peak widths
        widths, width_heights, left_ips, right_ips = peak_widths(
            1 - transmission_spectrum,
            minima_positions,
            rel_height=0.5,
        )

        # Annotate peak widths
        for i, position in enumerate(minima_positions):
            self.ax1.annotate(
                f"Width: {widths[i]*step*1e9:.2f}nm",
                xy=(wavelength[position] * 1e9, transmission_spectrum[position]),
                xytext=(
                    wavelength[position] * 1e9,
                    transmission_spectrum[position] + 0.2,
                ),
                # arrowprops=dict(facecolor="black", arrowstyle="simple"),
            )

        # Add horizontal lines to indicate
        self.ax1.hlines(
            1 - width_heights,
            wavelength[left_ips.astype(int)] * 1e9,  # position of left point
            wavelength[right_ips.astype(int)] * 1e9,  # position of right point
            color="C2",
        )
        self.ax1.scatter(
            wavelength[minima_positions] * 1e9,
            transmission_spectrum[minima_positions],
            color="C3",
            marker="o",
            label="Minima",
        )
        self.ax1.plot(wavelength * 1e9, transmission_spectrum)
        self.ax1.set_xlabel("Wavelength (nm)")
        self.ax1.set_ylabel("Transmission (arb unit)")
        self.ax1.grid(True)
        ################################################################################
        # Joint spectral intensity of the MRR
        ################################################################################

        fsr = 1e-9
        sig_wave = np.linspace(
            wavelength[minima_positions[2]] - fsr / 2,
            wavelength[minima_positions[2]] + fsr / 2,
            100,
        )
        idl_wave = np.linspace(
            wavelength[minima_positions[4]] - fsr / 2,
            wavelength[minima_positions[4]] + fsr / 2,
            100,
        )
        k_sig = util.wavevector(sig_wave)
        k_idl = util.wavevector(idl_wave)

        pump_func = lambda ks, ki: (1 - np.abs(pm.E_out(ks + ki)))
        joint_spec_func = lambda ks, ki: (1 - np.abs(pm.E_out(ks))) * (
            1 - np.abs(pm.E_out(ki))
        )
        # Turn it into a 2D matrix
        pef = np.abs(util.func_to_matrix(pump_func, k_sig, k_idl))
        pmf = util.func_to_matrix(joint_spec_func, k_sig, k_idl)
        jsi = np.abs(pef * pmf) ** 2

        # Calculate purity
        purity, entropy = util.get_purity(jsi)

        self.ax2.contourf(sig_wave * 1e9, idl_wave * 1e9, np.abs(jsi), levels=200)
        self.ax2.set_xlabel("Wavelength (nm)")
        self.ax2.set_aspect("equal")
        self.ax2.text(
            0.95,
            0.05,
            f"Purity: {purity * 1e2:0.2f}%" + f"\n Entropy: {entropy:0.4f}",
            fontsize=12,
            ha="right",
            va="bottom",
            color="white",
            transform=self.ax2.transAxes,
        )
        ################################################################################

        self.canvas.draw_idle()

    def change_param_values(self, new_value, attribute, target_description, prefactor):
        for index, (value, description) in enumerate(
            self.default_param[attribute]["values"]
        ):
            if description == target_description:
                self.default_param[attribute]["values"][index] = (
                    new_value / prefactor,
                    target_description,
                )
                break

    def extract_attributes_values(self, attributes):
        # Extract values from the parameter dictionary and
        # append them to an array

        # Initialize an empty array to store attributes values
        values_array = []

        # Loop through the values associated with "attributes"
        for value, description in self.default_param[attributes]["values"]:
            values_array.append(value)

        # Return the array of values
        return values_array


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main_window = MRR_GUI()
    sys.exit(app.exec_())
