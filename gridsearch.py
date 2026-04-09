import numpy as np

from ftp import FTP
from sinusoidal import SinusoidalImage

from parameters import p, A, phase0, phase, x_min, x_max, y_min, y_max, Nx, Ny

def grid_search(params: dict) -> tuple[tuple,float]:
    
    X0, Y0, im0 = SinusoidalImage(wavelength=p,
                                  amplitude=A,
                                  phase=phase0).generate(x_min=x_min, x_max=x_max,
                                                         y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

    X, Y, im = SinusoidalImage(wavelength=p, 
                               amplitude=A,
                               phase=phase).generate(x_min=x_min, x_max=x_max,
                                                      y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

    grid = [(window, padding, filter_width) for window in params['window'] for padding in params['padding'] for filter_width in params['filter_width']]
    res, min = (None, None, None), np.inf
    
    for (window, padding, filter_width) in grid:
        
        ftp = FTP(image=im, image_ref=im0, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
		  window=window, padding=padding, filter_width=filter_width)
        ftp.compute()
        delta_phi = ftp.phase_diff()

        score = np.linalg.norm(delta_phi[:,0]-phase(Y[:,0]), ord=2)
        
        if score < min:
            min = score # type: ignore
            res = (window, padding, filter_width)

    return (res, min) # type: ignore
