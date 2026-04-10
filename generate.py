import matplotlib.pyplot as plt
import numpy as np

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
Nx, Ny = 500, 500

# image generation

X, Y, image = SinusoidalImage(wavelength=p,
						amplitude=A,
						phase=phase0).generate(x_min=x_min, x_max=x_max,
							 y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
