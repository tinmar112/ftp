import numpy as np
from surface_reader import SurfaceReader


if __name__ == '__main__':

    path_ref = './photos/30deg/less-fringes-ref.jpg'
    path = './photos/30deg/less-fringes.jpg'
    step = 3.65e-5  # m/pixel
    L = 0.65        # m
    D = 0.0675      # m

    # creating the expected profile
    end_x, start_y, width = 1491, 1340, 1508
    base, angle = 5.5e-2, np.pi/6

    expected_profile = np.zeros(shape=(4096,2048))
    half_triangle = np.linspace(0, (base/2)*np.tan(angle),num=width//2) # 30-degree triangle
    triangle = np.append(half_triangle, half_triangle[::-1])
    expected_profile[start_y: start_y+width, :end_x] = triangle[:, np.newaxis]

    # read profile + compare to reality
    surface_reader = SurfaceReader(path=path, path_ref=path_ref, step=step)
    surface_reader.read(filter_width=0.36, L=L, D=D, average=False)
    error = surface_reader.error(expected_profile=expected_profile, show='2D')