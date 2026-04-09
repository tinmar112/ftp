import numpy as np

# parameters
lambda_x = 1e-2
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

def phase(y: float) -> float:
	return np.pi/2
phase = np.vectorize(phase)

def phase2(y: float) -> float:
	"""Phase of deformed surface"""
	if -p/2 <= y <= p/2:
		return -0.01 * 0.5 * (1 + np.cos((2*np.pi/p) * y))
	else:
		return 0.
phase2 = np.vectorize(phase2)

# image bounds
x_min, x_max = -2.5 * lambda_x, 2.5 * lambda_x
y_min, y_max = -5 * p, 5 * p
Nx, Ny = 1080, 1920
