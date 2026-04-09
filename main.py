from test import test
from parameters import phase_square

if __name__ == '__main__':
    test(phase=phase_square, window='hamming', padding=0.1, filter_width=0.73)