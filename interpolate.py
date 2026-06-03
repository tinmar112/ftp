import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def interpolate(im: np.ndarray, im_ref, r: float = 0.23) -> np.ndarray:
    
    q1, q3 = np.percentile(im, [25, 75])
    mask = (im <= q1 - r*(q3-q1))
    
    im_interpolated = np.where(mask, im_ref, im)
    return im_interpolated


if __name__ == '__main__':

    path = './photos/shell/rimless-c.jpg'
    path_ref = './photos/shell/rimless-c-ref.jpg'

    pil = Image.open(path).convert('L')
    im = np.array(pil)

    pil_ref = Image.open(path_ref).convert('L')
    im_ref = np.array(pil_ref)
    plt.imshow(interpolate(im, im_ref, r=0.22), cmap='gray')
    plt.show()
