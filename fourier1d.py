import matplotlib.pyplot as plt
import numpy as np


class Fourier1D:

    def __init__(self, signal: np.ndarray, step: float,
                 window_beta: float, padding: float | None,
                 filter_width: float=4) -> None:
        self._signal = signal - np.mean(signal) # centred right away
        self._step = step
        
        self._window_beta = window_beta
        self._padding = padding
        self._pad = 0

        self._f: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self._transform: np.ndarray = np.empty(shape=signal.shape, dtype=np.complex128)
        
        self._f_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.float64)
        self._transform_shifted: np.ndarray = np.empty(shape=signal.shape, dtype=np.complex128)
        
        self._fund: float = 0.
        
        self._filter_width = filter_width

        self.inv: np.ndarray = np.empty(shape=signal.shape, dtype=np.complex128)
    
    def pad(self) -> None:
        """Adds 0-padding to the signal."""
        assert self._padding is not None
        p = len(self._signal)
        p = int(self._padding * p)
        self._signal = np.pad(self._signal, pad_width=((p, p)), 
                              mode='constant', constant_values=0)
        self._pad = p

    def window(self) -> None:
        """Applies a Kaiser window to the image."""

        window = np.kaiser(len(self._signal), beta=self._window_beta)
        self._signal = self._signal * window

    def fft(self) -> None:
        """Applies the FTP algorithm to an image (2D Numpy ndarray format)."""

        # Remove background
        signal = self._signal - np.mean(self._signal) # Replace with B in real algorithm!

        n = len(self._signal)
        self._f, self._transform = np.fft.fftfreq(n, d=self._step), np.fft.fft(signal)
        self._f_shifted, self._transform_shifted = np.fft.fftshift(self._f),  np.fft.fftshift(self._transform)

    def plot(self) -> None:
        """Plots frequency modules."""

        spectrum = 20 * np.log10(np.abs(self._transform_shifted)) # dB scale

        plt.plot(self._f_shifted, spectrum)
        plt.title('Frequency Spectrum')
        plt.xlabel(r'$f \: (m^{—1})$')
        plt.ylabel(r'$Amplitude \: (dB)$')
        plt.show()

    def find_fundamental(self):
        """Finds the fundamental frequency."""

        i = np.abs(self._transform).argmax()
        self._fund = np.abs(self._f[i]) # positive frequency

    def filter(self) -> None:
        """Filters out the fundamental frequency."""

        sigma = self._filter_width * self._fund

        gaussian = np.exp(-(self._f - self._fund)**2/(2 * sigma ** 2))
        
        self._transform = gaussian * self._transform # element-wise multiplication
        
    def inverse_fft(self) -> None:
        """Computes the inverse FFT of the processed image."""
        
        self.inv = np.fft.ifft(self._transform) # use the unshifted transform

    def unpad(self) -> None:
        """Removes padding."""
        assert self._padding is not None
        p = self._pad
        self.inv = self.inv[p: -p]
