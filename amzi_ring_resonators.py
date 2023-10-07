import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c

import util


# Define the transfer matrix for a waveguide section
def amzi_transfer_matrix(prop_constant, length, theta=pi):
    propagation_phase = np.exp(1j * prop_constant * length + theta)
    return np.array([[propagation_phase, 0], [0, 1]])


# Define the coupling coefficient matrix
def coupler_transfer_matrix(coupling_coefficient):
    return np.array(
        [
            [np.sqrt(1 - coupling_coefficient), 1j * np.sqrt(coupling_coefficient)],
            [1j * np.sqrt(coupling_coefficient), np.sqrt(1 - coupling_coefficient)],
        ]
    )


# Calculate the overall transfer matrix for the entire system
def calculate_transfer_matrix(
    wavelength,
    waveguide_length,
    coupling_coefficient,
):
    prop_constant = util.wavevector(wavelength, util.group_index)
    M_amzi = amzi_transfer_matrix(prop_constant, waveguide_length, theta=pi)
    M_coupler = coupler_transfer_matrix(coupling_coefficient)
    M_total = np.einsum("ij, jk, kl -> il", M_coupler, M_amzi, M_coupler)
    return M_total


# Calculate the transmission coefficient
def calculate_transmission_coefficient(transfer_matrix):
    return np.abs(transfer_matrix[1, 0] / transfer_matrix[0, 0]) ** 2


if __name__ == "__main__":
    # Define parameters
    centre_wavelength = 1550e-9
    wavelength_range = np.linspace(1500e-9, 1600e-9, 1000)  # Adjust the range as needed
    waveguide_length = pi / util.wavevector(
        centre_wavelength,
        util.group_index,
    )  # Waveguide length in meters
    coupling_coefficient = 0.5  # Coupling coefficient (0 <= kappa <= 1)

    # Calculate the transmission spectrum for a range of wavelengths
    transmission_spectrum = []

    for wl in wavelength_range:
        M = calculate_transfer_matrix(
            wl,
            waveguide_length,
            coupling_coefficient,
        )
        transmission = calculate_transmission_coefficient(M)
        transmission_spectrum.append(transmission)

    # Plot the transmission spectrum
    plt.figure()
    plt.plot(
        wavelength_range * 1e9, transmission_spectrum
    )  # Convert to nm for plotting
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("Transmission Spectrum")
    plt.grid(True)
    plt.show()
