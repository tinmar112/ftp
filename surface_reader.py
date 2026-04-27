import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from scipy.ndimage import uniform_filter # type: ignore

from ftp import FTP
from height import height

class SurfaceReader:

    def __init__(self, step: float, path: str, path_ref: str) -> None:
        
        self._step = step

        jpg0 = Image.open(path_ref).convert('L')  # 'L' mode converts to grayscale
        jpg = Image.open(path).convert('L')

        self._image_ref: np.ndarray = np.array(jpg0)
        self._image: np.ndarray = np.array(jpg)

    def read(self, L: float, D: float, average: bool) -> None:
        """Reads the surface profile by FTP and displays it."""

        ftp = FTP(image=self._image, image_ref=self._image_ref,
                  step_x=self._step, step_y=self._step,
                  window_beta=3, padding=0.1, filter_width=0.634)
        
        ftp.compute(plot_spectrum=False)
        delta_phi = ftp.phase_diff()

        profile = height(delta_phi=delta_phi,
                         p=ftp.p, L=L, D=D)
        
        self._profile = profile

        if average:
            self._profile = uniform_filter(self._profile, size=5, mode='constant')
        
        # displaying
        extent = (0, profile.shape[1] * self._step, 0, profile.shape[0] * self._step)
        plt.imshow(profile, cmap='jet', extent=extent)
        plt.title('Height profile')
        plt.xlabel(r'$x \: (m)$')
        plt.ylabel(r'$y \: (m)$')
        plt.colorbar(label='Height (m)')
        plt.show()
    
    def compare_with(self, expected_profile: np.ndarray) -> None:

        error = self._profile - expected_profile
        
        print(error.sum().sum())

        # displaying
        extent = (0, error.shape[1] * self._step, 0, error.shape[0] * self._step)
        plt.imshow(error/1e-3, cmap='gray', extent=extent)
        plt.title('Height error = Reconstructed - Expected')
        plt.xlabel(r'$x \: (m)$')
        plt.ylabel(r'$y \: (m)$')
        plt.colorbar(label='Error (mm)')
        plt.show()
