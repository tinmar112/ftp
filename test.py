import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from ftp import FTP
from sinusoidal import SinusoidalImage

from parameters import p, A, phase0, x_min, x_max, y_min, y_max, Nx, Ny

def test(phase: Callable[[float], float], window: float, padding: float | None, filter_width: float) -> None:
    
	# image generation
	X0, Y0, im0 = SinusoidalImage(wavelength=p,
							   amplitude=A,
							   phase=phase0).generate(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)
	X, Y, im = SinusoidalImage(wavelength=p,
						amplitude=A,
						phase=phase).generate(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

	# FTP algorithm
	ftp = FTP(image=im, image_ref=im0, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
		   window=window, padding=padding, filter_width=filter_width)

	ftp.compute()

	delta_phi = ftp.phase_diff()

	# results plotting
	y = Y0[:,0]
	true_phase, reconstructed = phase(y), delta_phi[:,0]
	print(f'Mean L2-error: {np.linalg.norm(reconstructed-true_phase, ord=2)/reconstructed.size}')
	
	plt.plot(y, reconstructed, label=r'Reconstructed $\Delta \phi$')
	plt.plot(y, true_phase, label=r'True $\Delta \phi$')
	plt.xlabel(r'$y \: (m)$')
	plt.ylabel(r'$\Delta \phi \: (rad)$')
	plt.legend()
	plt.title(f'$p = {p}$ m')
	plt.show()
