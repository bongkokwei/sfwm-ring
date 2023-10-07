import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c, pi

import util


def param_dict(k_dict):
    param_dict = k_dict.copy()
    r_dict = {key.replace("k", "r"): np.sqrt(1 - k**2) for key, k in k_dict.items()}
    param_dict.update(r_dict)

    return param_dict


def E_in(freq0, epsilon=1e-5):
    return lambda freq: 1.0 / (pi * epsilon * (1 + ((freq - freq0) / epsilon) ** 2))


class PhotonicMolecule:
    def __init__(self, param_dict, E_in):
        self.param = param_dict
        self.E_in = E_in

    def T_ring2(self, phase):
        # complex valued tranmission in the aux ring
        # From equation 4
        T = (
            (
                self.param["r_mzi"] ** 2
                - self.param["k_mzi"] ** 2 * self.param["r_mziL"] * np.exp(1j * phase)
            )
            * self.param["r_2L"]
            * np.exp(1j * phase)
        )

        return T

    def E_ins2(self, phase):
        # From equation 10

        numerator = (
            -self.param["k_1"]
            * self.param["k_2"]
            * self.param["r_1L"]
            * np.exp(1j * phase / 2)
            * self.E_in(phase)
        )
        denom = (
            1
            - self.param["r_1"]
            * self.param["r_2"]
            * self.param["r_1L"]
            * np.exp(1j * phase)
        ) * (1 - self.param["r_2"] * self.T_ring2(phase)) + (
            self.param["k_2"] ** 2
            * self.param["r_1L"]
            * self.param["r_1"]
            * np.exp(1j * phase)
            * self.T_ring2(phase)
        )
        return numerator / denom

    def E_ins1(self, phase):
        # from equation 8
        numerator = 1j * self.param["k_2"] * self.param["r_1"] * np.exp(
            1j * phase / 2
        ) * self.T_ring2(phase) * self.E_ins2(phase) + 1j * self.param[
            "k_1"
        ] * self.E_in(
            phase
        )
        denom = 1 - self.param["r_1"] * self.param["r_2"] * self.param["r_1L"] * np.exp(
            1j * phase
        )
        return numerator / denom

    def E_out(self, phase):
        term_1 = (
            1j
            * self.param["k_1"]
            * self.param["r_2"]
            * self.param["r_1L"]
            * np.exp(1j * phase)
            * self.E_ins1(phase)
        )
        term_2 = (
            self.param["k_1"]
            * self.param["k_2"]
            * np.exp(1j * phase / 2)
            * self.T_ring2(phase)
            * self.E_ins2(phase)
        )
        term_3 = self.param["r_1"] * self.E_in(phase)

        return term_1 - term_2 + term_3


if __name__ == "__main__":
    """
    Various parameters for the photonic molecule
    """
    Gamma = np.sqrt(0.01)

    k_dict = {
        "k_1": 1 - Gamma,
        "k_2": np.sqrt(0.4),
        "k_mzi": np.sqrt(0.2),
        "k_mziL": np.sqrt(0.01),
        "k_1L": Gamma,
        "k_2L": np.sqrt(0.01),
    }

    phase = np.linspace(-7 * pi, 7 * pi, 2000)

    param = param_dict(k_dict)
    E_in = lambda freq: 1
    pm = PhotonicMolecule(param, E_in)
    E_out = pm.E_out(phase)

    fig, ax = plt.subplots()
    ax.plot(phase / pi, np.abs(E_out) ** 2)
    ax.set_xlabel("Freq (THz)")
    ax.set_ylabel("Transmission (arb unit)")

    plt.show()
