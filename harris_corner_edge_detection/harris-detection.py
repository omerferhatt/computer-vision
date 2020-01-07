import numpy as np
from scipy import signal as sig
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray


def show_image(img_corn, img_edge):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax[0].set_title("corners found")
    ax[0].imshow(img_corn)
    ax[1].set_title("edges found")
    ax[1].imshow(img_edge)
    plt.show()


def create_x_y_kernels(multiplier=1.0):
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_kernel_y = np.array([[1,   2,   1],
                               [0,   0,   0],
                               [-1, -2,  -1]])
    sobel_kernel_x = sobel_kernel_x * multiplier
    sobel_kernel_y = sobel_kernel_y * multiplier
    return sobel_kernel_x, sobel_kernel_y


def convolve_with_kernel(image, kernel, mode='same'):
    return sig.convolve2d(image, kernel, mode=mode)


if __name__ == '__main__':
    img = imread('school.jpeg')
    gray_image = rgb2gray(img)

    kernel_x, kernel_y = create_x_y_kernels(multiplier=1.5)

    I_x = convolve_with_kernel(gray_image, kernel_x)
    I_y = convolve_with_kernel(gray_image, kernel_y)

    I_xx = gaussian_filter(I_x ** 2, sigma=1)
    I_xy = gaussian_filter(I_x * I_y, sigma=1)
    I_yy = gaussian_filter(I_y ** 2, sigma=1)

    k = 0.03

    A = np.array([[I_xx, I_xy],
                  [I_xy, I_yy]])

    detA = I_xx * I_yy - I_xy ** 2
    traceA = I_xx + I_yy

    harris_detect = detA - (k * (traceA ** 2))

    img_copy_for_corners = np.copy(img)
    img_copy_for_edges = np.copy(img)

    for rowindex, response in enumerate(harris_detect):
        for colindex, r in enumerate(response):
            if r > 0.001:
                # this is a corner
                img_copy_for_corners[rowindex, colindex] = [255, 0, 0]
            elif r < -0.00001:
                # this is an edge
                img_copy_for_edges[rowindex, colindex] = [0, 255, 0]

    show_image(img_copy_for_corners, img_copy_for_edges)
