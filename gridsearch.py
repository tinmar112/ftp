import numpy as np

from ftp import FTP
from sinusoidal import SinusoidalImage
from phase_diff import phase_diff

def grid_search(params: dict) -> tuple[tuple,float]:
    
    # constants
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

    def phase1(y: float) -> float:
        """Phase of deformed surface"""
        #h = 0.01 * np.sin((2*np.pi/p) * y) # surface profile
        #return (((2*np.pi*D)/p) * h)/(h - L)
        if -p/2 <= y <= p/2:
            return -0.01 * 0.5 * (1 + np.cos((2*np.pi/p) * y))
        else:
            return 0.
    phase1 = np.vectorize(phase1)

    # image bounds
    x_min, x_max = -2.5 * lambda_x, 2.5 * lambda_x
    y_min, y_max = -5 * p, 5 * p
    Nx, Ny = 1080, 1920

    X_ref, Y_ref, im_ref = SinusoidalImage(wavelength=p,
                                           amplitude=A,
                                           phase=phase0).generate(x_min=x_min, x_max=x_max,
                                                                  y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

    X, Y, im = SinusoidalImage(wavelength=p, amplitude=A,
                               phase=phase1).generate(x_min=x_min, x_max=x_max,
                                                      y_min=y_min, y_max=y_max, Nx=Nx, Ny=Ny)

    grid = [(window, padding, filter_width) for window in params['window'] for padding in params['padding'] for filter_width in params['filter_width']]
    res, min = (None, None, None), np.inf
    
    for (window, padding, filter_width) in grid:
        
        fourier_ref = FTP(im_ref, name='Reference Image', step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
                          window=window, padding=padding, filter_width=filter_width)
        fourier = FTP(im, name='Other Image', step_x=(x_max-x_min)/Nx, step_y=(y_max-y_min)/Ny,
                      window=window, padding=padding, filter_width=filter_width)

        fourier_ref.compute()
        fourier.compute()

        phase = phase_diff(other_image=fourier.inv, ref_image=fourier_ref.inv)
        # remove padding
        if fourier._padding != (0,0):
            p1, p2 = fourier._padding
            phase = phase[p1:-p1, p2:-p2]

        score = np.linalg.norm(phase[:,0]-phase1(Y_ref[:,0]),ord=2)
        
        if score < min:
            min = score # type: ignore
            res = (window, padding, filter_width)

    return (res, min) # type: ignore