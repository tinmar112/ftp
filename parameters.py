import numpy as np

# parameters
lambda_x = 1e-2
p = 1.4e-3 # 7 fringes per deformation (deformation ~ 10mm)
L = 0.5
D = 0.25

# callables
def A(x: float, y: float) -> float:
	"""Amplitude of the sine wave"""
	return 1.
A = np.vectorize(A)

# translation invariance along x: phase independent of x
def phase0(y: float) -> float:
	"""Reference phase"""
	return 0
phase0 = np.vectorize(phase0)

def phase_linear(y: float) -> float:
	lambda_y = 5e-1
	return (2*np.pi/lambda_y) * y
phase_linear = np.vectorize(phase_linear)

def phase_square(y: float) -> float:
	return 0.5 * y ** 2
phase_square = np.vectorize(phase_square)

def phase_sine(y: float) -> float:
	return 0.1 * np.sin(300*y)
phase_sine = np.vectorize(phase_sine)

def phase_dimple(y: float) -> float:
	"""Phase of deformed surface"""
	if -5e-3 <= y <= 5e-3:
		h = -0.001 * 0.5 * (1 + np.cos((2*np.pi/1e-2) * y))
	else:
		h = 0.
	return ((2*np.pi/p * D) * h) / (h - L)
phase_dimple = np.vectorize(phase_dimple)

# image bounds - as per projector resolution
x_min, x_max = -1.4e-2, 1.4e-2
y_min, y_max = -2.5e-2, 2.5e-2
Nx, Ny = 1080, 1920
