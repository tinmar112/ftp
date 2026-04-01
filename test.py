import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from ftp import FTP
from sinusoidal import SinusoidalImage
from phase_diff import phase_diff

# parameters
lambda_x = 3e-2
p = 7e-3

def A(x: float, y: float) -> float:
	"""Amplitude of the sine wave"""
	return 1 #np.cos((2*np.pi/lambda_x)*x)

# translation invariance along x: phase independent of x
def phase0(y: float) -> float:
	"""Reference phase"""
	return 0

def phase1(y: float) -> float:
    """Phase of deformed surface"""
    return (np.pi)

# image bounds
x_min, x_max = 0., 2 * lambda_x
y_min, y_max = 0., 10 * p
Nx, Ny = 1000, 1000

# image generation

X1, Y1, im1 = SinusoidalImage(wawelength=p,
					 amplitude=A,
					 phase=phase0).generate(x_min=x_min, x_max=x_max,
							 y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

X2, Y2, im2 = SinusoidalImage(wawelength=p,
					 amplitude=A,
					 phase=phase1).generate(x_min=x_min, x_max=x_max,
							 y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

fourier1 = FTP(im1, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny)
fourier2 = FTP(im2, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny)

fourier1.compute()
fourier2.compute()

phase = phase_diff(image=fourier2.inv, ref_image=fourier1.inv)
print(phase)

# phase plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, Y1, phase)
ax.set_xlabel('X1')
ax.set_ylabel('Y1')
ax.set_zlabel('$\delta \phi$') #type: ignore
plt.show()
