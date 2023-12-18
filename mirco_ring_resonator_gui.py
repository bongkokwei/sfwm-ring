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
from plot_settings import PlotSettings
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
                "values": {"kappa_1": 0.23, "kappa_2": 0.1, "kappa_mzi": 0.14},
            },
            "fsr": {
                "prefactor": 1e9,
                "slider_properties": (1, 50, 1),  # (min, max, step)
                "values": {"primary ring (nm)": 5e-9, "auxilary ring (nm)": 10e-9},
            },
            "loss": {
                "prefactor": 1e2,
                "slider_properties": (0, 600, 1),  # (min, max, step)
                "values": {"loss (dB/cm)": 3},
            },
            "central_wavelength": {
                "prefactor": 1e9,
                "slider_properties": (1300, 1700, 1),
                "values": {"amzi central wavelength (nm)": 1550e-9},
            },
            "pump_wavelength": {
                "prefactor": 1e9,
                "slider_properties": (1300, 1700, 1),
                "values": {"pump wavelength (nm)": 1550e-9},
            },
        }

        self.settings_dict = {
            "min_wavelength": {"label": "Min Wavelength:", "value": 1535},
            "max_wavelength": {"label": "Max Wavelength:", "value": 1565},
            "num_samp": {"label": "Number of wavelength steps", "value": 5000},
            "num_grids": {"label": "Number of grids", "value": 100},
            "num_fsr": {"label": "Number of FSR from pump wavelength", "value": 1},
            "jsi_range": {"label": "Contour range (1-10)", "value": 5},
            # Add more settings as needed
        }

        # Plot settings
        self.update_wavelength_range()
        self.update_ring_param()  # initialise ring resonator
        self.sliders = {}  # Dictionary to store sliders
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        for key, data in self.default_param.items():
            prefactor = data["prefactor"]
            value_dict = data["values"]
            slider_properties = data["slider_properties"]
            for description, value in value_dict.items():
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
        self.update_contour()
        self.layout.addWidget(self.canvas)

        self.setGeometry(0, 0, 1280, 720)
        self.setWindowTitle("Micro-ring resonator GUI")
        self.show()

        # Create a button for opening the plot settings popup
        plot_settings_button = QPushButton("Plot Settings", self)
        # Set a fixed width for the button (adjust the value as needed)
        plot_settings_button.setFixedWidth(100)
        plot_settings_button.clicked.connect(self.show_plot_settings_dialog)

        self.layout.addWidget(plot_settings_button)

    def show_plot_settings_dialog(self):
        # Create and show the plot settings popup
        settings_dialog = PlotSettings(self, self.settings_dict)
        settings_dialog.exec_()

    def update_wavelength_range(self):
        self.wavelength, self.step = np.linspace(
            self.settings_dict["min_wavelength"]["value"] * 1e-9,
            self.settings_dict["max_wavelength"]["value"] * 1e-9,
            num=int(self.settings_dict["num_samp"]["value"]),
            retstep=True,
        )

    def update_settings(self, new_settings):
        # Update the main window settings based on the values in the dictionary
        for setting_name, setting_info in new_settings.items():
            if setting_name in self.settings_dict:
                self.settings_dict[setting_name]["value"] = setting_info["value"]

        # Perform specific actions based on the updated settings, if needed
        self.update_wavelength_range()
        # Update plot and force a re-layout
        self.update_plot()
        self.update_contour()
        self.layout.update()

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
        slider.valueChanged.connect(self.update_ring_param)
        slider.valueChanged.connect(self.update_plot)
        slider.sliderReleased.connect(self.update_contour)

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

    def update_ring_param(self):
        self.ring = PhotonicMolecule(
            self.extract_attributes_values("coupling_coeff"),
            self.extract_attributes_values("fsr"),
            self.extract_attributes_values("loss")[0],
            self.extract_attributes_values("central_wavelength")[0],
        )

    def update_plot(self):
        self.layout.addWidget(self.canvas)
        self.ax1.clear()

        ################################################################################
        # Self defined plotting function
        ################################################################################

        wavevector = util.wavevector(self.wavelength)
        transmission_spectrum = np.abs(self.ring.E_out(wavevector)) ** 2

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
                f"{widths[i]*self.step*1e9:.2f}nm",
                xy=(self.wavelength[position] * 1e9, transmission_spectrum[position]),
                xytext=(
                    self.wavelength[position] * 1e9,
                    transmission_spectrum[position] + 0.2,
                ),
            )

        # Add horizontal lines to indicate peak widths
        self.ax1.hlines(
            1 - width_heights,
            self.wavelength[left_ips.astype(int)] * 1e9,  # position of left point
            self.wavelength[right_ips.astype(int)] * 1e9,  # position of right point
            color="C2",
        )
        self.ax1.scatter(
            self.wavelength[minima_positions] * 1e9,
            transmission_spectrum[minima_positions],
            color="C3",
            marker="o",
            label="Minima",
        )
        self.ax1.plot(self.wavelength * 1e9, transmission_spectrum)
        self.ax1.set_xlabel("Wavelength (nm)")
        self.ax1.set_ylabel("Transmission (arb unit)")
        self.ax1.grid(True)
        self.canvas.draw_idle()

    def update_contour(self):
        ################################################################################
        # Joint spectral intensity of the MRR
        ################################################################################
        self.ax2.clear()

        # Centering the jsi plot
        pump_wave = self.default_param["pump_wavelength"]["values"][
            "pump wavelength (nm)"
        ]
        fsr = self.default_param["fsr"]["values"]["primary ring (nm)"]
        m = self.settings_dict["num_fsr"]["value"]
        sig_wave_est = pump_wave - m * fsr
        idl_wave_est = pump_wave + m * fsr
        sig_wave_central, sig_index = util.find_closest_value(
            self.wavelength,
            sig_wave_est,
        )
        idl_wave_central, idl_index = util.find_closest_value(
            self.wavelength,
            idl_wave_est,
        )

        jsi_range = self.settings_dict["jsi_range"]["value"]
        sig_wave = np.linspace(
            sig_wave_central - fsr / jsi_range,
            sig_wave_central + fsr / jsi_range,
            int(self.settings_dict["num_grids"]["value"]),
        )
        idl_wave = np.linspace(
            idl_wave_central - fsr / jsi_range,
            idl_wave_central + fsr / jsi_range,
            int(self.settings_dict["num_grids"]["value"]),
        )
        k_sig = util.wavevector(sig_wave)
        k_idl = util.wavevector(idl_wave)

        pump_func = lambda ks, ki: (np.abs(self.ring.E_out((ks + ki) / 0.5) - 1))
        joint_spec_func = lambda ks, ki: (1 - np.abs(self.ring.E_out(ks))) * (
            1 - np.abs(self.ring.E_out(ki))
        )
        # Turn it into a 2D matrix
        pef = np.abs(util.func_to_matrix(pump_func, k_sig, k_idl))
        pmf = util.func_to_matrix(joint_spec_func, k_sig, k_idl)
        jsi = np.abs(pef * pmf) ** 2

        # Calculate purity
        purity, entropy = util.get_purity(jsi)

        self.ax2.contourf(sig_wave * 1e9, idl_wave * 1e9, np.abs(jsi), levels=200)

        # Set x-axis ticks
        xticks = np.linspace(
            min(sig_wave) * 1e9, max(sig_wave) * 1e9, num=4
        )  # Adjust num as needed
        self.ax2.set_xticks(xticks)
        self.ax2.set_xlabel("Wavelength (nm)")

        # Set y-axis ticks
        yticks = np.linspace(
            min(idl_wave) * 1e9, max(idl_wave) * 1e9, num=4
        )  # Adjust num as needed
        self.ax2.set_yticks(yticks)

        # Set aspect ratio
        self.ax2.set_aspect("equal")

        # Add other labels and text as needed
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

        self.canvas.draw_idle()
        ################################################################################

    def change_param_values(self, new_value, attribute, target_description, prefactor):
        self.default_param[attribute]["values"][target_description] = (
            new_value / prefactor
        )

    def extract_attributes_values(self, attribute):
        # Extract values from the parameter dictionary and
        # append them to an array
        values_array = list(self.default_param[attribute]["values"].values())
        return values_array


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main_window = MRR_GUI()
    sys.exit(app.exec_())
