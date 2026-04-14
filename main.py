from test import test
from parameters import phase_dimple as phase

if __name__ == '__main__':
    test(phase=phase, window='hamming', padding=0.5, filter_width=0.2)
