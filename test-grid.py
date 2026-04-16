import numpy as np

from gridsearch import grid_search
from parameters import phase_sine

params={'window_beta': [5, 6, 8.6, 14],
        'padding': [None, 0.1, 0.25],
        'filter_width': np.linspace(0.1, 1., num=10)}

(res, min) = grid_search(phase=phase_sine, params=params)

print(f'Best parameters: {res}')
print(f'Minimum L2-error: {min}')
