import numpy as np
from skimage.restoration import unwrap_phase

from fourier import Fourier

class FTP:

    def __init__(self, image: np.ndarray, image_ref: np.ndarray, step_x: float, step_y: float,
                 window: str, padding: float | None, filter_width: float) -> None:
        
        self._fourier0 = Fourier(image_ref, step_x=step_x, step_y=step_y,
                                 window=window, padding=padding, filter_width=filter_width)

        self._fourier = Fourier(image, step_x=step_x, step_y=step_y,
                                window=window, padding=padding, filter_width=filter_width)

    def compute(self, verbose: bool = True) -> None:
        """Executes the FTP algorithm on both signals."""
        
        self._fourier0.window()
        self._fourier0.pad() if self._fourier0._padding else None
        self._fourier0.fft()
        self._fourier0.plot()
        self._fourier0.find_fundamental()
        self._fourier0.filter()
        self._fourier0.inverse_fft()
        self._fourier0.unpad() if self._fourier0._padding else None

        self._fourier.window()
        self._fourier.pad() if self._fourier._padding else None
        self._fourier.fft()
        self._fourier.plot()
        self._fourier.find_fundamental()
        self._fourier.filter()
        self._fourier.inverse_fft()
        self._fourier.unpad() if self._fourier._padding else None

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
    