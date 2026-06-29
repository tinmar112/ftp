import matplotlib.pyplot as plt
from surface_reader import SurfaceReader


if __name__ == '__main__':

    path_ref = './photos/phantom/DSC_8106.jpg'
    path = './photos/phantom/DSC_8107.jpg'
    step = 4.7e-5  # m/pixel
    L = 97 * 1e-2    # m
    D = 20 * 1e-2  # m


    # read profile + compare to reality
    surface_reader = SurfaceReader(path=path, path_ref=path_ref, step=step)
    widths = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    algs = ['1D', '2D']

    fig, axs = plt.subplots(nrows=len(algs), ncols=len(widths),
                            figsize=(4 * len(widths), 4 * len(algs)),
                            squeeze=False)

    for i, alg in enumerate(algs):
        for j, width in enumerate(widths):
            surface_reader.read(alg=alg, filter_width=width, #type: ignore
                                L=L, D=D, average=False,
                                show=False)

            ax = axs[i][j]
            profile = surface_reader.profile
            extent = (0, profile.shape[1] * step / 1e-3,
                      0, profile.shape[0] * step / 1e-3)

            im = ax.imshow(profile / 1e-3, cmap='jet', extent=extent)
            ax.set_title(f'{alg}, width={width}')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')

    plt.tight_layout()
    plt.show()

    import numpy as np
    surface_reader.read(alg='2D', filter_width=0.6, L=L, D=D, average=False, show=False)
    surface_reader.error(expected_profile=np.zeros(shape=surface_reader.profile.shape),show='2D')
