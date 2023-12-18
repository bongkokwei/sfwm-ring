import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c, pi

import util


def E_in(freq0, epsilon=1e-5):
    return lambda freq: 1.0 / (pi * epsilon * (1 + ((freq - freq0) / epsilon) ** 2))


class PhotonicMolecule:
    def __init__(self, coupling, fsr, loss, central_wavelength):
        self.E_in = lambda freq: 1
        self.kappa = coupling
        self.fsr = fsr
        self.loss_dB_cm = loss
        self.central_wavelength = central_wavelength
        self.param = self.create_param_dict()

    def T_ring2(self, k):
        # complex valued tranmission in the aux ring
        # From equation 4
        T = (
            (
                self.param["r_mzi"] ** 2
                - self.param["k_mzi"] ** 2
                * self.param["r_mziL"]
                * np.exp(1j * k * self.param["delta_L_amzi"])
            )
            * self.param["r_2L"]
            * np.exp(1j * k * self.param["L2"])
        )

        return T

    def E_ins2(self, k):
        # From equation 10
        numerator = (
            -self.param["k_1"]
            * self.param["k_2"]
            * self.param["r_1L"]
            * np.exp(1j * k * self.param["L1"] / 2)
            * self.E_in(1)
        )
        denom = (
            1
            - self.param["r_1"]
            * self.param["r_2"]
            * self.param["r_1L"]
            * np.exp(1j * k * self.param["L1"])
        ) * (1 - self.param["r_2"] * self.T_ring2(k)) + (
            self.param["k_2"] ** 2
            * self.param["r_1L"]
            * self.param["r_1"]
            * np.exp(1j * k * self.param["L1"])
            * self.T_ring2(k)
        )
        return numerator / denom

    def E_ins1(self, k):
        # from equation 8
        numerator = 1j * self.param["k_2"] * self.param["r_1"] * np.exp(
            1j * k * self.param["L1"] / 2
        ) * self.T_ring2(k) * self.E_ins2(k) + 1j * self.param["k_1"] * self.E_in(1)
        denom = 1 - self.param["r_1"] * self.param["r_2"] * self.param["r_1L"] * np.exp(
            1j * k * self.param["L1"]
        )
        return numerator / denom

    def E_out(self, k):
        term_1 = (
            1j
            * self.param["k_1"]
            * self.param["r_2"]
            * self.param["r_1L"]
            * np.exp(1j * k * self.param["L1"])
            * self.E_ins1(k)
        )
        term_2 = (
            self.param["k_1"]
            * self.param["k_2"]
            * np.exp(1j * k * self.param["L1"] / 2)
            * self.T_ring2(k)
            * self.E_ins2(k)
        )
        term_3 = self.param["r_1"] * self.E_in(1)

        return term_1 - term_2 + term_3

    def create_param_dict(self):
        k_dict = {
            "k_1": np.sqrt(self.kappa[0]),
            "k_2": np.sqrt(self.kappa[1]),
            "k_mzi": np.sqrt(self.kappa[2]),
        }

        length_dict = {
            "L1": util.fsr2length(
                self.fsr[0],
                self.central_wavelength,
                util.refractive_index(self.central_wavelength * 1e6),
            ),
            "L2": util.fsr2length(
                self.fsr[1],
                self.central_wavelength,
                util.refractive_index(self.central_wavelength * 1e6),
            ),
            "delta_L_amzi": self.central_wavelength  # phase_shift = (2*pi*n_eff*delta_L)/wavelength
            / util.refractive_index(self.central_wavelength * 1e6),
        }

        loss_dict = {
            "k_mziL": np.sqrt(
                1 - util.loss(self.loss_dB_cm, length_dict["delta_L_amzi"])
            ),
            "k_1L": np.sqrt(1 - util.loss(self.loss_dB_cm, length_dict["L1"])),
            "k_2L": np.sqrt(1 - util.loss(self.loss_dB_cm, length_dict["L2"])),
        }

        k_dict.update(loss_dict)
        r_dict = {
            key.replace("k", "r"): np.sqrt(1 - k**2) for key, k in k_dict.items()
        }

        param = {}
        param.update(k_dict)
        param.update(r_dict)
        param.update(length_dict)
        param.update(loss_dict)
        return param


if __name__ == "__main__":
    """
    Various parameters for the photonic molecule
    """
    wavelength = np.linspace(1550e-9 - 15e-9, 1550e-9 + 15e-9, num=40000)
    wavevector = util.wavevector(wavelength)

    coupling = [0.23, 0.1, 0.14]
    fsr = [5e-9, 10e-9]
    loss = 3  # dB/cm
    central_wavelength = 1550e-9

    pm = PhotonicMolecule(coupling, fsr, loss, central_wavelength)
    E_out = pm.E_out(wavevector)

    fig, ax = plt.subplots()
    ax.plot(wavelength * 1e9, np.abs(E_out) ** 2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (arb unit)")

    plt.show()
