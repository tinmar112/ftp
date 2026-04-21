import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from ftp import FTP
from height import height
from sinusoidal import SinusoidalImage

from parameters import p, L, D, A, phase0, x_min, x_max, y_min, y_max, Nx, Ny
from parameters import phase_dimple as phase


def test(phase: Callable[[float], float], window_beta: float,
		 padding: float | None, filter_width: float) -> None:
    
	# image generation
	X0, Y0, im0 = SinusoidalImage(wavelength=p,
							   amplitude=A,
							   phase=phase0).generate(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)
	X, Y, im = SinusoidalImage(wavelength=p,
						amplitude=A,
						phase=phase).generate(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

	# FTP algorithm
	ftp = FTP(image=im, image_ref=im0, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
		   window_beta=window_beta, padding=padding, filter_width=filter_width)

	ftp.compute()

	delta_phi = ftp.phase_diff()

	# results plotting

	y = Y0[:,0]
	true_phase = phase(y)
	reconstructed = delta_phi[:,0]

	print(f'Mean L2-error in phase: {np.linalg.norm(reconstructed-true_phase, ord=2)/reconstructed.size}')

	i = np.argmax(np.abs(true_phase-reconstructed))
	print(f'Max absolute error in height: {np.abs(height(true_phase[i], p=p, L=L, D=D)-height(reconstructed[i], p=p, L=L, D=D))}') #type: ignore
	
	plt.plot(y, reconstructed, label=r'Reconstructed $\Delta \phi$')
	plt.plot(y, true_phase, label=r'True $\Delta \phi$')
	plt.xlabel(r'$y \: (m)$')
	plt.ylabel(r'$\Delta \phi \: (rad)$')
	plt.legend()
	plt.title(f'$p = {p}$ m')
	plt.show()

if __name__ == '__main__':
    test(phase=phase, window_beta=3, padding=0.1, filter_width=0.634)
    