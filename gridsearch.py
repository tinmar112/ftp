import pandas as pd # type: ignore
import numpy as np
from tqdm import tqdm # type: ignore
from typing import Callable

from ftp import FTP
from sinusoidal import SinusoidalImage

from parameters import p, A, phase0, x_min, x_max, y_min, y_max, Nx, Ny


def grid_search(phase: Callable[[float], float], params: dict) -> tuple:
    
    X0, Y0, im0 = SinusoidalImage(wavelength=p,
                                  amplitude=A,
                                  phase=phase0).generate(x_min=x_min, x_max=x_max,
                                                         y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

    X, Y, im = SinusoidalImage(wavelength=p, 
                               amplitude=A,
                               phase=phase).generate(x_min=x_min, x_max=x_max,
                                                      y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

    grid = [(window_beta, padding, filter_width) for window_beta in params['window_beta'] for padding in params['padding'] for filter_width in params['filter_width']]
    res, min = (None, None, None), np.inf
    
    for (window_beta, padding, filter_width) in tqdm(grid):
        
        ftp = FTP(image=im, image_ref=im0, step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
		  window_beta=window_beta, padding=padding, filter_width=filter_width)
        
        ftp.compute(verbose=False)
        delta_phi = ftp.phase_diff()

        score = np.linalg.norm(delta_phi[:,0]-phase(Y[:,0]), ord=2)
        
        if score < min:
            min = score # type: ignore
            res = (window_beta, padding, filter_width)

    return (res, min) # type: ignore
