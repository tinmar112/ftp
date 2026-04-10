from test import test
from parameters import phase_dimple

if __name__ == '__main__':
    test(phase=phase_dimple, window='hamming', padding=0.1, filter_width=0.67)
