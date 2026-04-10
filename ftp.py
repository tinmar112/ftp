import numpy as np
from skimage.restoration import unwrap_phase

from fourier import Fourier

class FTP:

    def __init__(self, image: np.ndarray, image_ref: np.ndarray, step_x: float, step_y: float,
                 window: str, padding: float, filter_width: float) -> None:
        
        self._fourier0 = Fourier(image_ref, step_x=step_x, step_y=step_y,
                                 window=window, padding=padding, filter_width=filter_width)

        self._fourier = Fourier(image, step_x=step_x, step_y=step_y,
                                window=window, padding=padding, filter_width=filter_width)

    def compute(self, verbose: bool = True) -> None:
        """Executes the FTP algorithm on both signals."""
        
        self._fourier0.fft()
        self._fourier0._find_fundamental()
        self._fourier0._filter()
        self._fourier0._inverse_fft()

        self._fourier.fft()
        self._fourier._find_fundamental()
        self._fourier._filter()
        self._fourier._inverse_fft()

        if verbose:
            print(f"---------- Signal: {'Reference Image'} ----------")
            print(f"Fundamental wavelength for x: {1/self._fourier0.fund_x} m")
            print(f"Fundamental wavelength for y: {1/self._fourier0.fund_y} m" + "\n")
            print(f"---------- Signal: {'Other Image'} ----------")
            print(f"Fundamental wavelength for x: {1/self._fourier.fund_x} m")
            print(f"Fundamental wavelength for y: {1/self._fourier.fund_y} m" + "\n")
    
    def phase_diff(self) -> np.ndarray:
        """Computes the phase difference between the two images."""

        other_image, ref_image = self._fourier.inv, self._fourier0.inv

        prod = other_image * np.conjugate(ref_image)
        delta_phi = np.imag(np.log(prod))

        delta_phi = unwrap_phase(delta_phi)

        return delta_phi
    