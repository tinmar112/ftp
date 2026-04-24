import numpy as np
from surface_reader import SurfaceReader

if __name__ == '__main__':

    path_ref = './photos/30deg/more-fringes-ref.jpg'
    path = './photos/30deg/more-fringes.jpg'
    step = 3.65e-5  # m/pixel
    L = 0.65        # m
    D = 0.0675      # m

    surface_reader = SurfaceReader(path=path, path_ref=path_ref, step=step)

    # creating the expected profile
    base, angle = 2.25e-2, np.pi/6
    expected_profile = np.zeros(shape=surface_reader._image_ref.shape)
    half_triangle = np.linspace(0, base*np.tan(angle),num=754) # 30-degree triangle
    triangle = np.append(half_triangle, half_triangle[::-1])
    expected_profile[1242:1242+754*2, :1491] = triangle[:, np.newaxis]
    
    surface_reader.read(L=L, D=D)

    surface_reader.compare_with(expected_profile=expected_profile)
