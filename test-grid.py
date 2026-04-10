import numpy as np

from gridsearch import grid_search
from parameters import phase_square

params={'window': ['hann', 'hamming', 'blackman'],
        'padding': [0.1],
        'filter_width': np.linspace(0.4, 0.75, num=10)}

(res, min) = grid_search(phase=phase_square, params=params)

print(f'Best parameters: {res}')
print(f'Minimum L2-error: {min}')
