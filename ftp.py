import numpy as np


class FTP:

    def __init__(self, image: np.ndarray, image_ref: np.ndarray, step_x: float, step_y: float,
                 window_beta: float, padding: float | None, filter_width: float) -> None:
        
        self._image = image
        self._image_ref = image_ref
        self._step_x = step_x
        self._step_y = step_y
        self._window_beta = window_beta
        self._padding = padding
        self._filter_width = filter_width
    
    def compute(self):
        pass

    @property
    def p(self) -> float:
        return 0.
