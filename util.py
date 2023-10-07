import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.constants import c, pi


def freq2wave(freq):
    """
    Convert frequency to wavelength.

    Parameters:
        freq (float): Frequency in Hz.

    Returns:
        float: Wavelength in meters.
    """
    return c / freq


def wave2freq(wave):
    """
    Convert wavelength to frequency.

    Parameters:
        wave (float): Wavelength in meters.

    Returns:
        float: Frequency in Hz.
    """
    return c / wave


def ang_freq(freq):
    """
    Calculate angular frequency.

    Parameters:
        freq (float): Frequency in Hz.

    Returns:
        float: Angular frequency in radians per second.
    """
    return 2 * pi * freq


def prop_const(wavelength, n_eff):
    """
    Calculate propagation constant.

    Parameters:
        wavelength (float): Wavelength in meters.
        n_eff (float): Effective refractive index.

    Returns:
        float: Propagation constant.
    """
    return (2 * pi * n_eff) / wavelength


def theta(wavelength, n_eff, L):
    """
    Calculate phase shift.

    Parameters:
        wavelength (float): Wavelength in meters.
        n_eff (float): Effective refractive index.
        L (float): Length in meters.

    Returns:
        float: Phase shift in radians.
    """
    return (2 * pi * n_eff) * (L / wavelength)


def free_spectral_range(wavelength, n_eff, L):
    """
    Calculate free spectral range.

    Parameters:
        wavelength (float): Wavelength in meters.
        n_eff (float): Effective refractive index.
        L (float): Length in meters.

    Returns:
        float: Free spectral range in Hz.
    """
    return wavelength**2 / (n_eff * L)


def fsr2length(fsr, wavelength, n_eff):
    """
    Calculate length from free spectral range.

    Parameters:
        fsr (float): Free spectral range in meters.
        wavelength (float): Wavelength in meters.
        n_eff (float): Effective refractive index.

    Returns:
        float: Length in meters.
    """
    return wavelength**2 / (n_eff * fsr)


def fwhm(kappa, wavelength, L, n_eff):
    """
    Calculate full width at half maximum (FWHM).

    Parameters:
        kappa (complex): Coupling coefficient.
        wavelength (float): Wavelength in meters.
        L (float): Length in meters.
        n_eff (float): Effective refractive index.

    Returns:
        float: FWHM in meters.
    """
    return (np.abs(kappa) ** 2 * wavelength**2) / (pi * L * n_eff)


def fwhm2length(fwhm, kappa, wavelength, n_eff):
    """
    Calculate length from FWHM.

    Parameters:
        fwhm (float): FWHM in meters.
        kappa (complex): Coupling coefficient.
        wavelength (float): Wavelength in meters.
        n_eff (float): Effective refractive index.

    Returns:
        float: Length in meters.
    """
    return (np.abs(kappa) ** 2 * wavelength**2) / (pi * fwhm * n_eff)


def theta(n_eff, r, wavelength):
    """
    Calculate phase shift.

    Parameters:
        n_eff (float): Effective refractive index.
        r (float): Radius.
        wavelength (float): Wavelength in meters.

    Returns:
        float: Phase shift in radians.
    """
    return 4 * pi**2 * n_eff * (r / wavelength)


def refractive_index(wave, coeff=(3.0249, 0.1353406, 40314, 1239.842)):
    """
    Calculate refractive index for silicon nitride using the Sellmeier equation.

    Parameters:
        wave (float): Wavelength in microns.
        coeff (tuple): Sellmeier coefficients (a, b, c, d).

    Returns:
        float: Refractive index.
    """
    a, b, c, d = coeff
    n = np.sqrt(
        ((a * wave**2) / (wave**2 - b**2))
        + ((c * wave**2) / (wave**2 - d**2))
        + 1
    )

    return n


def numerical_derivative(func, x, h=1e-5):
    """
    Numerically approximate the derivative of a function at a point x.

    Parameters:
        func (function): The function to differentiate.
        x (float): The point at which to calculate the derivative.
        h (float): Step size for finite difference method.

    Returns:
        float: Approximated derivative.
    """
    return (func(x + h) - func(x - h)) / (2 * h)


def derivative_of_refractive_index(
    wave, coeff=(3.0249, 0.1353406, 40314, 1239.842), h=1e-5
):
    """
    Calculate the derivative of refractive index with respect to wavelength using the Sellmeier equation.

    Parameters:
        wave (float): Wavelength in microns.
        coeff (tuple): Sellmeier coefficients (a, b, c, d).
        h (float): Step size for numerical differentiation.

    Returns:
        float: Derivative of refractive index with respect to wavelength.
    """
    return numerical_derivative(lambda w: refractive_index(w, coeff), wave, h)


def group_index(wave, coeff=(3.0249, 0.1353406, 40314, 1239.842), h=1e-5):
    """
    Calculate group index for silicon nitride using the Sellmeier equation.

    Parameters:
        wave (float): Wavelength in microns.
        coeff (tuple): Sellmeier coefficients (a, b, c, d).
        h (float): Step size for numerical differentiation.

    Returns:
        float: Group index.
    """
    n = refractive_index(wave, coeff)
    dn_dlambda = derivative_of_refractive_index(wave, coeff, h)
    ng = n + wave * dn_dlambda
    return ng


def wavevector(wavelength, n=refractive_index):
    """
    Calculate wave vector.

    Parameters:
        wavelength (float): Wavelength in meter.
        n (function): Function to calculate refractive index.

    Returns:
        float: Wave vector.
    """
    return ((2 * pi) / wavelength) * n(wavelength * 1e6)


def coupler_transfer_matrix(coupling_coefficient):
    """
    Calculate the transfer matrix for a coupler.

    Parameters:
        coupling_coefficient (float): Coupling coefficient.

    Returns:
        numpy.ndarray: Transfer matrix.
    """
    return np.array(
        [
            [np.sqrt(1 - coupling_coefficient), 1j * np.sqrt(coupling_coefficient)],
            [1j * np.sqrt(coupling_coefficient), np.sqrt(1 - coupling_coefficient)],
        ]
    )


def phase_shift(phase_difference):
    """
    Calculate a phase shift matrix.

    Parameters:
        phase_difference (float): Phase difference in radians.

    Returns:
        numpy.ndarray: Phase shift matrix.
    """
    return np.array([[np.exp(1j * phase_difference), 0], [0, 1]])


def lorentzian(x, x0, gamma):
    """
    Compute the Lorentzian function L(x; x0, gamma).

    Parameters:
        x (float or array-like): The input value(s) at which to evaluate the function.
        x0 (float): The center of the Lorentzian distribution.
        gamma (float): The half-width at half-maximum (HWHM), which controls the width of the distribution.

    Returns:
        float or array-like: The value(s) of the Lorentzian function at the given input(s).
    """
    return 1.0 / (np.pi * gamma * (1 + ((x - x0) / gamma) ** 2))


def get_purity(jsa):
    """
    Calculate purity and entropy of a joint spectral amplitude.

    Parameters:
        jsa (numpy.ndarray): Joint spectral amplitude.

    Returns:
        float: Purity of the joint spectral amplitude.
        float: Entropy of the joint spectral amplitude.
    """
    u, s, vh = np.linalg.svd(jsa, full_matrices=True)
    s /= np.sqrt(np.sum(s**2))  # Normalize Schmidt coefficients

    # From Dosseva et al. 2016, pg 3
    entropy = -np.sum(np.abs(s) ** 2 * np.log(np.abs(s) ** 2))
    purity = np.sum(s**4)
    return purity, entropy


def func_to_matrix(lambda_func, list1, list2):
    A = []
    for idx1 in list1:
        A_row = []
        for idx2 in list2:
            A_row += [lambda_func(idx1, idx2)]
        A += [A_row]
    return A


def plot_jsi(
    sig,
    idl,
    pef,
    pmf,
    jsi,
    y_label_str="Idler frequecy (nm)",
    x_label_str="Signal frequecy (nm)",
):
    """
    Plot the joint spectral intensity and related functions.

    Parameters:
        sig (numpy.ndarray): Signal frequencies.
        idl (numpy.ndarray): Idler frequencies.
        pef (numpy.ndarray): Pump envelope function.
        pmf (numpy.ndarray): Phase-matched function.
        jsi (numpy.ndarray): Joint spectral intensity.
        y_label_str (str): Label for the y-axis.
        x_label_str (str): Label for the x-axis.

    Returns:
        None
    """
    # Create a figure with three subplots
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Calculate purity
    purity, entropy = get_purity(jsi)

    # Plot the filled contour plots in each subplot
    ax1.contourf(sig, idl, np.abs(pmf), levels=200)
    ax1.set_title("Phase-matched function")
    ax1.set_ylabel(y_label_str)
    ax1.set_xlabel(x_label_str)
    ax1.set_aspect("equal")

    ax2.contourf(sig, idl, np.abs(pef), levels=200)
    ax2.set_title("Pump envelope function")
    ax2.set_xlabel(x_label_str)
    ax2.set_aspect("equal")

    ax3.contourf(sig, idl, jsi, levels=200)
    ax3.set_title("Joint Spectrum Intensity")
    ax3.set_xlabel(x_label_str)
    ax3.set_aspect("equal")
    ax3.text(
        0.95,
        0.05,
        f"Purity: {purity * 1e2:0.2f}%",
        fontsize=12,
        ha="right",
        va="bottom",
        color="white",
        transform=ax3.transAxes,
    )


if __name__ == "__main__":
    wavelength = 1.55e-6  # Wavelength in meter
    group_idx = group_index(wavelength * 1e6)
    round_trip_length = fsr2length(2e-9, wavelength, group_idx)
    print(f"Radius of ring resonator: {round_trip_length*1e6/(2*pi):.2f} micron")
    print(wavevector(wavelength, group_index))
