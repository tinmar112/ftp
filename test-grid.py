import numpy as np

from gridsearch import grid_search
from parameters import phase_linear, phase_square, phase_dimple, phase_sine

phase_list = [phase_linear, phase_square, phase_dimple, phase_sine]

params={'window_beta': [5, 6, 8.6, 14],
        'padding': [None, 0.1, 0.25],
        'filter_width': np.linspace(0.1, 1., num=10)}

results = grid_search(phase_list=phase_list, params=params) #type: ignore

print(f'Best overall parameter: {results}')
