import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from ftp import FTP
from height import height

if __name__ == '__main__':
    
    jpg0 = Image.open('ref.jpg').convert('L')  # 'L' mode converts to grayscale
    jpg = Image.open('triangle.jpg').convert('L')
    
    im0, im = np.array(jpg0), np.array(jpg)

    #test(phase=phase, window_beta=3, padding=0.1, filter_width=0.634)
    ftp = FTP(image=im, image_ref=im0, step_x=0.054/1080, step_y=0.096/1920,
		   window_beta=3, padding=0.1, filter_width=0.634)
    
    ftp.compute(plotting=True)
    delta_phi = ftp.phase_diff()
    profile = height(delta_phi=delta_phi, p = 2.6e-3, L = 0.4, D = 0.15)

    plt.imshow(profile, cmap='jet')
    plt.colorbar()
    plt.show()
