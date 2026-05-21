import numpy as np

from skimage.restoration import unwrap_phase

from ftp import FTP
from fourier2d import Fourier2D


class FTP2D(FTP):

    def __init__(self, image: np.ndarray, image_ref: np.ndarray, step_x: float, step_y: float,
                 window_beta: float, padding: float | None, filter_width: float) -> None:
        
        FTP.__init__(self, image, image_ref, step_x, step_y, window_beta, padding, filter_width)
    
    def _phase_diff(self) -> np.ndarray:
        """Computes the phase difference between the two images."""

        other_image, ref_image = self._fourier.inv, self._fourier0.inv

        prod = other_image * np.conjugate(ref_image)
        delta_phi = np.angle(prod)

        delta_phi = unwrap_phase(delta_phi)

        return delta_phi

    def compute(self, plot_spectrum: bool = False, verbose: bool = True) -> np.ndarray:
        """Executes the FTP algorithm on both signals."""
        
        self._fourier0 = Fourier2D(self._image_ref, step_x=self._step_x, step_y=self._step_y,
                                   window_beta=self._window_beta, padding=self._padding,
                                   filter_width=self._filter_width)

        self._fourier = Fourier2D(self._image, step_x=self._step_x, step_y=self._step_y,
                                  window_beta=self._window_beta, padding=self._padding,
                                  filter_width=self._filter_width)

        self._fourier0.window()
        self._fourier.window()

        if self._fourier0._padding is not None:
            self._fourier0.pad() 
            self._fourier.pad()
        
        self._fourier0.fft()
        self._fourier.fft()

        if plot_spectrum:
            self._fourier0.plot()
            self._fourier.plot()
        
        self._fourier0.find_fundamental()
        #self._fourier.find_fundamental()
        self._fourier.fund_x = self._fourier0.fund_x
        self._fourier.fund_y = self._fourier0.fund_y

        self._fourier0.filter()
        self._fourier.filter()

        self._fourier0.inverse_fft()
        self._fourier.inverse_fft()
        
        if self._fourier0._padding is not None:
            self._fourier0.unpad() 
            self._fourier.unpad()

        if verbose:
            print(f"---------- Signal: {'Reference Image'} ----------")
            print(f"Fundamental wavelength for x: {1/self._fourier0.fund_x} m")
            print(f"Fundamental wavelength for y: {1/self._fourier0.fund_y} m" + "\n")
            print(f"---------- Signal: {'Other Image'} ----------")
            print(f"Fundamental wavelength for x: {1/self._fourier.fund_x} m")
            print(f"Fundamental wavelength for y: {1/self._fourier.fund_y} m" + "\n")

        # phase difference
        return self._phase_diff()

    @property
    def p(self) -> float:
        """Returns the wavelength of the fringe pattern."""
        return 1/self._fourier0.fund_y
