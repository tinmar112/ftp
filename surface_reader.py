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

    def read(self, filter_width: float,
             L: float, D: float, average: bool,
             show: bool = True) -> None:
        """Reads the surface profile by FTP and displays it."""

        ftp = FTP(image=self._image, image_ref=self._image_ref,
                  step_x=self._step, step_y=self._step,
                  window_beta=3, padding=0.1,
                  filter_width=filter_width)
        
        ftp.compute(plot_spectrum=False)
        delta_phi = ftp.phase_diff()

        self._profile = height(delta_phi=delta_phi, p=ftp.p, L=L, D=D)

        if average:
            self._profile = uniform_filter(self._profile, size=10, mode='constant', axes=0) # axes 0 is for y!
            self._profile = uniform_filter(self._profile, size=5, mode='constant', axes=1) # axes 1 is for x!
        
        if show:
            # displaying
            profile = self._profile
            extent = (0, profile.shape[1] * self._step / 1e-3, 0, profile.shape[0] * self._step / 1e-3) # type: ignore
            plt.imshow(profile/1e-3, cmap='jet', extent=extent)
            plt.title('Height profile')
            plt.xlabel(r'$x \: (mm)$')
            plt.ylabel(r'$y \: (mm)$')
            plt.colorbar(label='Height (mm)')
            plt.show()
    
    def error(self, expected_profile: np.ndarray, show: str ) -> float:

        error = self._profile - expected_profile

        if show == '2D':
            # Cross-section
            y = np.linspace(0, error.shape[0] * self._step, num=len(error))/1e-3
            plt.plot(y, self._profile[:,1312]/1e-3, label='Reconstruction')
            plt.plot(y, expected_profile[:,1312]/1e-3, label='Expected')
            plt.xlabel('y (mm)')
            plt.ylabel('Height (mm)')
            plt.title('Cross-section')
            plt.legend()
            plt.show()

        elif show == '3D':
            # Heat-map
            extent = (0, error.shape[1] * self._step, 0, error.shape[0] * self._step)
            plt.imshow(error/1e-3, cmap='gray', extent=extent)
            plt.title('Height error = Reconstructed - Expected')
            plt.xlabel(r'$x \: (m)$')
            plt.ylabel(r'$y \: (m)$')
            plt.colorbar(label='Error (mm)')
            plt.show()

        return (error ** 2).sum()
