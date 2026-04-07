import matplotlib.pyplot as plt
import numpy as np

from ftp import FTP
from height import height
from sinusoidal import SinusoidalImage
from phase_diff import phase_diff

# parameters
lambda_x = 3e-3
p = 5e-3
L = 0.5
D = 0.25

def A(x: float, y: float) -> float:
	"""Amplitude of the sine wave"""
	return np.cos((2*np.pi/lambda_x)*x)

# translation invariance along x: phase independent of x
def phase0(y: float) -> float:
	"""Reference phase"""
	return 0

def phase1(y: float) -> float:
	"""Phase of deformed surface"""
	#h = 0.01 * np.sin((2*np.pi/p) * y) # surface profile
	#return (((2*np.pi*D)/p) * h)/(h - L)
	if -p/2 <= y <= p/2:
		return -0.01 * 0.5 * (1 + np.cos((2*np.pi/p) * y))
	else:
		return 0.

# image bounds
x_min, x_max = -3 * lambda_x, 3 * lambda_x
y_min, y_max = -5 * p, 5 * p
Nx, Ny = 1080, 1920

# image generation

X1, Y1, im1 = SinusoidalImage(wavelength=p,
						 amplitude=A,
						 phase=phase0).generate(x_min=x_min, x_max=x_max,
							 y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

X2, Y2, im2 = SinusoidalImage(wavelength=p,
					 amplitude=A,
					 phase=phase1).generate(x_min=x_min, x_max=x_max,
							 y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

fourier1 = FTP(im1, name='Reference Image', step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
			   window='hamming', padding=0.025, filter_width=1.)

fourier2 = FTP(im2, name='Other Image', step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
			   window='hamming', padding=0.025, filter_width=1.)

fourier1.compute()
print(f"---------- Signal: {'Reference Image'} ----------")
print(f"Fundamental wavelength for x: {1/fourier1.fund_x} m")
print(f"Fundamental wavelength for y: {1/fourier1.fund_y} m" + "\n")
fourier2.compute()
print(f"---------- Signal: {'Other Image'} ----------")
print(f"Fundamental wavelength for x: {1/fourier2.fund_x} m")
print(f"Fundamental wavelength for y: {1/fourier2.fund_y} m" + "\n")

phase = phase_diff(other_image=fourier2.inv, ref_image=fourier1.inv)
# remove padding
if fourier2._padding != (0,0):
	p1, p2 = fourier2._padding
	phase = phase[p1:-p1, p2:-p2]

# phase plotting
y = Y1[:,0]

phase1 = np.vectorize(phase1)
plt.plot(y, phase[:,0].T, label=r'Reconstructed $\Delta \phi$')
plt.plot(y, phase1(y), label=r'True $\Delta \phi$')
plt.xlabel(r'$y \: (m)$')
plt.ylabel(r'$\Delta \phi \: (rad)$')
plt.legend()
plt.title(f'$p = {p}$ m')
plt.show()
