import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from ftp import FTP
from height import height

if __name__ == '__main__':
    
    jpg0 = Image.open('./photos/30deg/more-fringes-ref.jpg').convert('L')  # 'L' mode converts to grayscale
    jpg = Image.open('./photos/30deg/more-fringes.jpg').convert('L')
    
    im0, im = np.array(jpg0), np.array(jpg)

    step = 3.65e-5 # m/pixel
    ftp = FTP(image=im, image_ref=im0, step_x=step, step_y=step,
              window_beta=3, padding=0.1, filter_width=0.634)
    
    ftp.compute(plot_spectrum=False)
    delta_phi = ftp.phase_diff()

    profile = height(delta_phi=delta_phi, p = ftp.p, L = 0.65, D = 0.067)

    extent = (0, profile.shape[1] * step, 0, profile.shape[0] * step)
    plt.imshow(profile, cmap='jet', extent=extent)
    plt.title('Height profile')
    plt.xlabel(r'$x \: (m)$')
    plt.ylabel(r'$y \: (m)$')
    plt.colorbar(label='m')
    plt.show()
