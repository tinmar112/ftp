import matplotlib.pyplot as plt
import numpy as np

from ftp import FTP
from sinusoidal import SinusoidalImage

# parameters
lambda_x = 3e-2
p = 5e-3

def A(x: float, y: float) -> float:
	"""Amplitude of the sine wave"""
	return np.cos((2*np.pi/lambda_x)*x + y)

# translation invariance along x: phase independent of x
def phase0(y: float) -> float:
	"""Reference phase"""
	return 0

def phase1(y: float) -> float:
    """Phase of deformed surface"""
    return (10*y)

# image bounds
x_min, x_max = 0., 2 * lambda_x
y_min, y_max = 0., 10 * p
Nx, Ny = 1000, 1000

# image generation

im = SinusoidalImage(wawelength=p,
					 amplitude=A,
					 phase=phase0).generate(x_min=x_min, x_max=x_max,
							 y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

fourier = FTP(im, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny)
fourier.fft()

fourier._find_fundamental()

fourier._filter(sigma=10/p,fc_x=1/fourier.fund_x, fc_y=1/fourier.fund_y) # narrow filtering: sigma >> fc
fourier._inverse_fft()

plt.imshow(fourier.inv, cmap='gray')
plt.show()