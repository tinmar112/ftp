import numpy as np


def height(delta_phi: np.ndarray, p: float, L: float, D: float) -> np.ndarray:
    """Reconstructs the height based on phase readings."""
    num = L * delta_phi
    denom = delta_phi - (2 * np.pi * D)/p
    return np.divide(num, denom)
