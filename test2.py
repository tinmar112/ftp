from gridsearch import grid_search

params={'window': ['hann', 'hamming'],
        'padding': [0.025, 0.05, 0.1],
        'filter_width': [0.1, 0.5, 1, 2, 4, 10]}

print(grid_search(params=params))