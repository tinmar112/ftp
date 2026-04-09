import numpy as np

from gridsearch import grid_search

params={'window': ['hann', 'hamming', 'blackman'],
        'padding': [0.1],
        'filter_width': np.linspace(0.1, 2., num=10)}

print(grid_search(params=params))