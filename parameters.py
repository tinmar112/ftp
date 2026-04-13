import numpy as np

# parameters
lambda_x = 1e-2
p = 5e-3
L = 0.5
D = 0.25

# callables
def A(x: float, y: float) -> float:
	"""Amplitude of the sine wave"""
	return np.cos((2*np.pi/lambda_x)*x)

# translation invariance along x: phase independent of x
def phase0(y: float) -> float:
	"""Reference phase"""
	return 0
phase0 = np.vectorize(phase0)

def phase_const(y: float) -> float:
	return np.pi/2
phase_const = np.vectorize(phase_const)

def phase_square(y: float) -> float:
	return 0.5 * y ** 2
phase_square = np.vectorize(phase_square)

def phase_sine(y: float) -> float:
	return 0.1 * np.sin(300*y)
phase_sine = np.vectorize(phase_sine)

def phase_dimple(y: float) -> float:
	"""Phase of deformed surface"""
	if -p/2 <= y <= p/2:
		h = -0.001 * 0.5 * (1 + np.cos((2*np.pi/p) * y))
	else:
		h = 0.
	return ((2*np.pi/p * D) * h) / (h - L)
phase_dimple = np.vectorize(phase_dimple)

# image bounds
x_min, x_max = -2.5 * lambda_x, 2.5 * lambda_x
y_min, y_max = -5 * p, 5 * p
Nx, Ny = 1080, 1920
