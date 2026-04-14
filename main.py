from test import test
from parameters import phase_const as phase

if __name__ == '__main__':
    test(phase=phase, window='hamming', padding=0.2, filter_width=0.5)
