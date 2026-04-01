import matplotlib.pyplot as plt
import numpy as np

class FTP:

    def __init__(self, signal: np.ndarray, step_x: float, step_y: float) -> None:
        self._signal = signal
        self._step_x = step_x
        self._step_y = step_y

        self.FX: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.FY: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.transform: np.ndarray = np.empty(shape=signal.shape, dtype=np.complex128)
        
        self.FX_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.FY_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.transform_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.complex128)
        
        self.fund_x: float = 0.
        self.fund_y: float = 0.
        
        self.filter_coeffs: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)

        self.inv: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        

    def fft(self) -> None:
        """Applies the FTP algorithm to an image (2D Numpy ndarray format)."""

        # Remove background
        signal = self._signal - self._signal.mean() # Replace with B in real algorithm!
        
        ft = np.fft.fft2(signal)
        (M, N) = signal.shape
        fx = np.fft.fftfreq(N, d=self._step_x) # reverse order for M,N bc x,y is the reverse
        fy = np.fft.fftfreq(M, d=self._step_y)
        
        FX, FY = np.meshgrid(fx, fy)
        self.FX, self.FY, self.transform = FX, FY, ft

        # shift
        fx_shifted = np.fft.fftshift(fx)
        fy_shifted = np.fft.fftshift(fy)
        ft_shifted = np.fft.fftshift(ft)

        FX_shifted, FY_shifted = np.meshgrid(fx_shifted, fy_shifted)
        self.FX_shifted, self.FY_shifted, self.transform_shifted = FX_shifted, FY_shifted, ft_shifted

    def plot(self) -> None:
        """Plots frequency modules along each axis -- x and y."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # along x-axis (sum over y)
        module_x = np.abs(self.transform_shifted.sum(axis=0))
        ax1.stem(self.FX_shifted[0, :], module_x)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('x-axis')
        ax1.grid(True)

        # along y-axis (sum over x)
        module_y = np.abs(self.transform_shifted.sum(axis=1))
        ax2.stem(self.FY_shifted[:, 0], module_y)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('y-axis')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def _find_fundamental(self):
        """Find the fundamental frequencies for both axes"""

        module = np.abs(self.transform)
        (i,j) = np.unravel_index(module.argmax(), module.shape)
        self.fund_x, self.fund_y = self.FX[0,j], self.FY[i,0]
        self.fund_x, self.fund_y = np.abs(self.fund_x), np.abs(self.fund_y) # positive freq

        print(f"Fundamental wavelength for x: {1/self.fund_x} m")
        print(f"Fundamental wavelength for y: {1/self.fund_y} m")

    def _filter(self, sigma: float, fc_x: float, fc_y: float) -> None:

        def filter(x: float, y: float) -> float:

            g_x = np.exp(-(x-fc_x)**2/(2*sigma**2))
            g_y = np.exp(-(y-fc_y)**2/(2*sigma**2))

            return g_x * g_y # should be 1/(2*np.pi*sigma**2) - I want a gain of 1 at fc_x/y
        
        filter = np.vectorize(filter)

        coefficients = filter(self.FX, self.FY)
        self.transform = coefficients * self.transform # element-wise multiplication
        #self.transform = self.transform / np.sum(coefficients**2) # normalise to preserve energy!

    def _inverse_fft(self) -> None:
        
        ifft = np.fft.ifft2(self.transform) # use the unshifted transform
        self.inv = ifft
    
    def compute(self):

        self.fft()

        self._find_fundamental()

        # adjust sigma? - # narrow filtering: sigma >> fc
        self._filter(sigma=10*self.fund_y,fc_x=self.fund_x, fc_y=self.fund_y)
        
        self._inverse_fft()
    