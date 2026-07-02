import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def interpolate(im: np.ndarray, im_ref, r: float = 0.23) -> np.ndarray:
    """Interpolates shadows with replacement pixels from a reference image."""
    
    q1, q3 = np.percentile(im, [25, 75])
    mask = (im <= q1 - r*(q3-q1))
    
    im_interpolated = np.where(mask, im_ref, im)
    return im_interpolated
