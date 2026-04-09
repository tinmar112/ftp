import numpy as np

from gridsearch import grid_search
from parameters import phase_square

params={'window': ['hann', 'hamming', 'blackman'],
        'padding': [0.1],
        'filter_width': np.linspace(0.1, 2., num=10)}

print(grid_search(phase=phase_square, params=params))
