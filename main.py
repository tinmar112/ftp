import numpy as np
from surface_reader import SurfaceReader


if __name__ == '__main__':

    path_ref = './photos/60deg/60deg-100f-ref.jpg'
    path = './photos/60deg/60deg-100f.jpg'
    step = 3.8e-5  # m/pixel
    L = 45 * 1e-2        # m
    D = 67 * 1e-3        # m

    # creating the expected profile
    end_x, start_y, width = 1508, 260, 1442
    base, angle = 5.45e-2, np.pi/3

    expected_profile = np.zeros(shape=(3400,2400))
    half_triangle = np.linspace(0, (base/2)*np.tan(angle),num=width//2) # 30-degree triangle
    triangle = np.append(half_triangle, half_triangle[::-1])
    expected_profile[start_y: start_y+width, :end_x] = triangle[:, np.newaxis]

    # read profile + compare to reality
    surface_reader = SurfaceReader(path=path, path_ref=path_ref, step=step)

    surface_reader.read(alg='1D', filter_width=0.4, L=L, D=D, average=True)
    error = surface_reader.error(expected_profile=expected_profile, show='2D')
