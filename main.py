from test import test
from parameters import phase_dimple as phase

if __name__ == '__main__':
    test(phase=phase, window_beta=5, padding=0.1, filter_width=0.1)
