import matplotlib.pyplot as plt
import numpy as np

from surface_reader import SurfaceReader

if __name__ == '__main__':

    path_ref = './photos/nikon/dimple4/dimple4-40v-undef.jpg'
    path = './photos/nikon/dimple4/dimple4-40v-shell.jpg'
    step = 4.40e-5  # m/pixel
    L = 90 * 1e-2  # m
    D = 21 * 1e-2  # m


    # read profile + compare to reality
    surface_reader = SurfaceReader(path=path, path_ref=path_ref, step=step, shadow_r=0.22)
    widths = [0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]
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

    surface_reader.read(alg='2D', filter_width=0.40, L=L, D=D, average=False, show=False)
    
    center = (1310, 1356)
    radius = 569
    yy, xx = np.indices(surface_reader.profile.shape)
    rr = np.sqrt((xx - center[1])**2 + (yy - center[0])**2)
    expected_profile = np.zeros(surface_reader.profile.shape, dtype=float)
    mask = rr <= radius
    expected_profile[mask] = np.sqrt(radius**2 - rr[mask]**2)
    expected_profile[mask] = expected_profile[mask] * step
    expected_profile[mask] = expected_profile[mask] + 4e-3 # add base height if applicable

    # Uncomment when tracking with respect to an undeformed shell
    surface_reader.profile = expected_profile + surface_reader.profile

    surface_reader.error(expected_profile=expected_profile, show='2D')
