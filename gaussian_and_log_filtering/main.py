from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt
import cv2


def gaussian_2d_kernel(ker_shape, sigm):
    """
    Creates gaussian kernel

    :param ker_shape: Used in defining kernel's shape
    :param sigm: sigma parameter for equation
    :return: Gaussian kernel with shape of `ker_shape`
    """
    ker = np.zeros(ker_shape)
    ind = np.arange(-np.floor(ker_shape[0] / 2), np.floor(ker_shape[1] / 2) + 1, 1)
    XX, YY = np.meshgrid(ind, ind)

    for row in range(ker_shape[0]):
        for col in range(ker_shape[1]):
            ker[row, col] = (1 / 2 * np.pi * sigm ** 2) * np.exp(
                (-1 * (XX[0, col] ** 2 + YY[row, 0] ** 2) / (2 * sigm ** 2)))

    return ker


def laplacian_of_gaussian_2d_kernel(ker_shape, sigm):
    """
    Creates laplacian of gaussian kernel

    :param ker_shape: Used in defining kernel's shape
    :param sigm: sigma parameter for equation
    :return: LoG kernel with shape of `ker_shape`
    """
    ker = np.zeros(ker_shape)
    ind = np.arange(-np.floor(ker_shape[0] / 2), np.floor(ker_shape[1] / 2) + 1, 1)
    XX, YY = np.meshgrid(ind, ind)

    for row in range(ker_shape[0]):
        for col in range(ker_shape[1]):
            ker[row, col] = (-1 / (np.pi * sigm ** 4)) * (1 - (XX[0, col] ** 2 + YY[row, 0] ** 2) / (2 * sigm ** 2)) * \
                               np.exp((-1 * (XX[0, col] ** 2 + YY[row, 0] ** 2) / (2 * sigm ** 2)))

    return -ker


def normalize(arr):
    """
    Normalize image between 0-255

    :param arr: Going to be normalized array
    :return: Normalized array
    """
    arr = arr / arr.max(initial=0) * 255
    arr = np.uint8(arr)
    return arr


def apply_filter(arr, ker):
    """
    Makes convolution operation

    :param arr: Going to be convolved
    :param ker: Convolution operation kernel
    :return: Convolved image
    """
    return sig.convolve2d(arr, ker, mode='same')


def show_image(*argv):
    """
    Plot images in order

    :param argv: Needs to be passed grayscale images
    """
    for arg in argv:
        plt.imshow(arg, cmap='gray')
        plt.show()


if __name__ == '__main__':
    try:
        img = cv2.imread('../test_images/test.png', cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("File don't exist!")

        else:
            sigma = 1
            kernel_shape = (7, 7)

            gaussian_kernel = gaussian_2d_kernel(kernel_shape, sigma)
            log_kernel = laplacian_of_gaussian_2d_kernel(kernel_shape, sigma)

            gaussian_result = normalize(apply_filter(img, gaussian_kernel))
            log_result = normalize(apply_filter(img, log_kernel))
            log_after_gaussian_result = normalize(apply_filter(apply_filter(img, gaussian_kernel), log_kernel))

            show_image(img, gaussian_result, log_result, log_after_gaussian_result)

    except ValueError as ve:
        print(ve)
