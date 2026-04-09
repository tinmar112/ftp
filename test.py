import matplotlib.pyplot as plt

from ftp import FTP
from sinusoidal import SinusoidalImage

from parameters import p, A, phase0, phase, x_min, x_max, y_min, y_max, Nx, Ny

# image generation
X0, Y0, im0 = SinusoidalImage(wavelength=p,
						 amplitude=A,
						 phase=phase0).generate(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)
X, Y, im = SinusoidalImage(wavelength=p,
					 amplitude=A,
					 phase=phase).generate(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

# FTP algorithm
ftp = FTP(image=im, image_ref=im0, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
		  window='hamming', padding=0.1, filter_width=0.9444)

ftp.compute()

delta_phi = ftp.phase_diff()

# results plotting
y = Y0[:,0]

plt.plot(y, delta_phi[:,0].T, label=r'Reconstructed $\Delta \phi$')
plt.plot(y, phase(y), label=r'True $\Delta \phi$')
plt.xlabel(r'$y \: (m)$')
plt.ylabel(r'$\Delta \phi \: (rad)$')
plt.legend()
plt.title(f'$p = {p}$ m')
plt.show()
