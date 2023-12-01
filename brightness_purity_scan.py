import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths

from photonic_molecule import PhotonicMolecule
import util

kappa_2_squared = np.linspace(0.05, 0.5, 100)
kappa_amzi_squared = np.linspace(0.05, 0.5, 100)
