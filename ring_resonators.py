"""
Simulation for single ring resonator coupled to a straight waveguide
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c, pi

import util


# ----------------------------------------------------------------------
# Ring parameters
# ----------------------------------------------------------------------
def peak_number(m):
    return int(m)


# ----------------------------------------------------------------------
# Ring models
# ----------------------------------------------------------------------


def simple_ring(detuning, alpha, t):
    numer = alpha**2 + np.abs(t) ** 2 - 2 * alpha * np.abs(t) * np.cos(detuning)
    denom = 1 + np.abs(t) ** 2 * alpha**2 - 2 * alpha * np.abs(t) * np.cos(detuning)
    return numer / denom


def add_drop_ring(detuning, t1, t2, alpha):
    # Transmission spectrum of a ring coupled to 2 waveguides
    denom = 1 - 2 * t1 * t2 * alpha * np.cos(detuning) + (t1 * t2 * alpha) ** 2
    T_pass = (
        t2**2 * alpha**2 - 2 * t1 * t2 * alpha * np.cos(detuning) + t1**2
    ) / denom
    # T_drop = ((1 - t1**2) * (1 - t2**2) * alpha) / denom

    return T_pass





# ----------------------------------------------------------------------
# Joint spectrum
# ----------------------------------------------------------------------


def func_to_matrix(lambda_func, list1, list2):
    A = []
    for idx1 in list1:
        A_row = []
        for idx2 in list2:
            A_row += [lambda_func(idx1, idx2)]
        A += [A_row]
    return A


def sech_squared(x):
    sech_x = 1 / np.cosh(x)
    return sech_x**2


if __name__ == "__main__":
    # Ring parameter
    alpha = 0.85
    t1 = alpha  # coupling coeff
    t2 = np.abs(t1) / alpha

    # Wavelength range
    m = 1
    fsr = 10e-9
    c_wavelength = 1550e-9
    s_wavelength = c_wavelength + m * fsr
    i_wavelength = c_wavelength - m * fsr

    sig_wave = np.linspace(s_wavelength + fsr / 2, s_wavelength - fsr / 2, 500)
    idl_wave = np.linspace(i_wavelength + fsr / 2, i_wavelength - fsr / 2, 500)
    sig_group_idx = util.group_index(sig_wave * 1e6)
    idl_group_idx = util.group_index(idl_wave * 1e6)
    k_sig = util.wavevector(sig_wave, util.group_index)
    k_idl = util.wavevector(idl_wave, util.group_index)

    length = util.fsr2length(
        fsr,
        c_wavelength,
        util.group_index(c_wavelength * 1e6),
    )

    prop_const_sig = k_sig * length
    prop_const_idl = k_idl * length
    sig, idl = np.meshgrid(sig_wave, idl_wave)

    # Setting up the spectrum of the pump and signal/idler with the transmission
    # spectrum of the ring
    pump_sech = lambda sig, idl: sech_squared((sig + idl - 2 * c_wavelength) / 2e-9)
    pump_lorentzian = lambda sig, idl: (1 - add_drop_ring(sig + idl, alpha, t1, t2))
    sig_idl_lorentzian = lambda sig, idl: (1 - add_drop_ring(sig, alpha, t1, t2)) * (
        1 - add_drop_ring(idl, alpha, t1, t2)
    )

    # Turn it into a 2D matrix
    lorentz_pump = func_to_matrix(pump_lorentzian, prop_const_sig, prop_const_idl)
    sech_pump = func_to_matrix(pump_sech, sig_wave, idl_wave)
    pef = np.abs(lorentz_pump) * np.abs(sech_pump)
    pmf = func_to_matrix(sig_idl_lorentzian, prop_const_sig, prop_const_idl)
    jsi = np.abs(pef) * np.abs(pmf)

    # Plot the spectrum
    util.plot_jsi(sig * 1e9, idl * 1e9, pef, pmf, jsi)

    # fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    # ax1.plot(sig_wave * 1e9, add_drop_ring(prop_const_sig, alpha, t1, t2))
    # ax2.plot(idl_wave * 1e9, add_drop_ring(prop_const_idl, alpha, t1, t2))

    # Show the figure
    plt.tight_layout()
    plt.show()
