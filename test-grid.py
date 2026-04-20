import numpy as np

from gridsearch import grid_search
from parameters import phase_dimple as phase


params={'window_beta': np.linspace(3, 5, num=10),
        'padding': [0.1],
        'filter_width': np.linspace(0.5, 0.7, num=50)}

results = grid_search(phase=phase, params=params)

print(f'Best overall parameter: {results}')
