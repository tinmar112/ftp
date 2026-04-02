import numpy as np

from skimage.restoration import unwrap_phase


def phase_diff(other_image: np.ndarray, ref_image: np.ndarray, ) -> np.ndarray:
    """Computes the phase difference between two (complex valued) images."""

    prod = other_image * np.conjugate(ref_image)
    delta_phi = np.imag(np.log(prod))

    delta_phi = unwrap_phase(delta_phi)

    return delta_phi
