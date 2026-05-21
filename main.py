import matplotlib.pyplot as plt
from surface_reader import SurfaceReader


if __name__ == '__main__':

    path_ref = './photos/shell/shell-sc-ref.jpg'
    path = './photos/shell/shell-sc.jpg'
    step = 3.539e-5  # m/pixel
    L = 65 * 1e-2    # m
    D = 10 * 1e-2  # m


    # read profile + compare to reality
    surface_reader = SurfaceReader(path=path, path_ref=path_ref, step=step)
    widths = [0.06, 0.1, 0.2, 0.3, 0.5, 0.8]
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
