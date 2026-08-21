import numpy as np
from skimage.restoration import unwrap_phase

from fourier1d import Fourier1D
from ftp import FTP


class FTP1D(FTP):
    """ONLY IMPLEMENTED ALONG ONE OF THE AXES! - y."""

    def __init__(self, image: np.ndarray, image_ref: np.ndarray, step_x: float, step_y: float,
                 window_beta: float, padding: float | None, filter_width: float) -> None:
        
        FTP.__init__(self, image, image_ref, step_x, step_y, window_beta, padding, filter_width)

        self._p = np.empty(shape=image_ref.shape[1], dtype=np.float64)

    def compute(self, plot_spectrum: bool = False, verbose: bool = True) -> np.ndarray:
        """Executes the FTP algorithm on both signals."""

        to_concat = []

        for i in range(self._image_ref.shape[1]): 
        # slicing up along x-axis for y-axis analysis
            
            # reference image
            signal0 = self._image_ref[:, i]
            fourier0 = Fourier1D(signal=signal0, step=self._step_y,
                                 window_beta=self._window_beta,
                                 padding=self._padding,
                                 filter_width=self._filter_width)
            # deformed image 
            signal = self._image[:, i]
            fourier = Fourier1D(signal=signal, step=self._step_y,
                                window_beta=self._window_beta,
                                padding=self._padding,
                                filter_width=self._filter_width)

            fourier0.window()
            fourier.window()
            
            if self._padding is not None:
                fourier0.pad()
                fourier.pad()
            
            fourier0.fft()
            fourier.fft()
            
            if plot_spectrum:
                fourier0.plot()
                fourier.plot()
            
            fourier0.find_fundamental()
            #fourier.find_fundamental()
            fourier._fund = fourier0._fund
            
            fourier0.filter()
            fourier.filter()
            
            fourier0.inverse_fft()
            self._p[i] = fourier0._fund
            fourier.inverse_fft()
            
            if self._padding is not None:
                fourier0.unpad()
                fourier.unpad()

            # Phase difference computation
            prod = fourier.inv * np.conjugate(fourier0.inv)
            line_phi = np.angle(prod)

            # add phase_diff slice
            to_concat.append(line_phi.reshape((-1, 1)))
        
        delta_phi = np.concatenate(to_concat, axis=1)
        return unwrap_phase(delta_phi)

    @property
    def p(self) -> float:
        """Returns the wavelength of the fringe pattern."""
        return 1./np.mean(self._p) # type: ignore
