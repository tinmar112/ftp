import matplotlib.pyplot as plt
import numpy as np

class FTP:

    def __init__(self, signal: np.ndarray, name: str, step_x: float, step_y: float,
                 window: str, padding: float = 0.1, filter_width: float=4) -> None:
        self._signal = signal
        self._signal_name = name
        self._step_x = step_x
        self._step_y = step_y
        
        self._Window = window
        self._pad_ratio = padding
        self._padding = (0, 0)

        self.FX: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.FY: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.transform: np.ndarray = np.empty(shape=signal.shape, dtype=np.complex128)
        
        self.FX_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.FY_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self.transform_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.complex128)
        
        self.fund_x: float = 0.
        self.fund_y: float = 0.
        
        self._filter_width = filter_width
        self.filter_coeffs: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)

        self.inv: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
    
    def _pad(self) -> None:
        """Adds 0-padding to the signal."""
        p1, p2 = self._signal.shape
        p1, p2 = int(self._pad_ratio * p1), int(self._pad_ratio * p2)
        self._signal = np.pad(self._signal, pad_width=((p1,p1),(p2,p2)), 
                              mode='constant', constant_values=0)
        self._padding = (p1, p2)

    def _window(self) -> None:
        dic = {'blackman': np.blackman, 'hamming': np.hamming,'hann': np.hanning}
        window = dic[self._Window]

        (n,m) = self._signal.shape
        window_x = window(m)
        window_y = window(n)
        window_2D = np.outer(window_y, window_x)

        self._signal = self._signal * window_2D

    def fft(self) -> None:
        """Applies the FTP algorithm to an image (2D Numpy ndarray format)."""

        self._window()
        self._pad()
        
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

    def _filter(self, sigma: float, fc_x: float, fc_y: float) -> None:

        def filter(x: float, y: float) -> float:

            g_x = np.exp(-(x-fc_x)**2/(2*sigma**2))
            g_y = np.exp(-(y-fc_y)**2/(2*sigma**2))

            return g_x * g_y # should be 1/(2*np.pi*sigma**2) - I want a gain of 1 at fc_x
        
        filter = np.vectorize(filter)

        coefficients = filter(self.FX, self.FY)
        self.transform = coefficients * self.transform # element-wise multiplication
        #self.transform = self.transform / np.sum(coefficients**2) # normalise to preserve energy!

    def _inverse_fft(self) -> None:
        
        self.inv = np.fft.ifft2(self.transform) # use the unshifted transform
    
    def compute(self):

        self.fft()

        self._find_fundamental()

        self._filter(sigma=self.fund_y/self._filter_width, fc_x=self.fund_x, fc_y=self.fund_y) # narrow filtering
        
        self._inverse_fft()
    