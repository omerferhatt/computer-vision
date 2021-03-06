import numpy as np
from scipy import signal as sig
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import cv2


def show_image(img_corn, img_edge):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax[0].set_title("corners found")
    ax[0].imshow(img_corn)
    ax[1].set_title("edges found")
    ax[1].imshow(img_edge)
    plt.show()


def create_x_y_kernels():
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_kernel_y = np.array([[1,   2,   1],
                               [0,   0,   0],
                               [-1, -2,  -1]])

    return sobel_kernel_x, sobel_kernel_y


def convolve_with_kernel(image, kernel, mode='same'):
    return sig.convolve2d(image, kernel, mode=mode)


if __name__ == '__main__':
    kernel_x, kernel_y = create_x_y_kernels()
    kernel_x, kernel_y = kernel_x * 2, kernel_y * 2
    k = 0.05

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            I_x = convolve_with_kernel(gray_image, kernel_x)
            I_y = convolve_with_kernel(gray_image, kernel_y)

            I_xx = gaussian_filter(I_x ** 2, sigma=1)
            I_xy = gaussian_filter(I_x * I_y, sigma=1)
            I_yy = gaussian_filter(I_y ** 2, sigma=1)

            A = np.array([[I_xx, I_xy],
                          [I_xy, I_yy]])

            detA = I_xx * I_yy - I_xy ** 2
            traceA = I_xx + I_yy

            harris_detect = detA - (k * (traceA ** 2))

            img_copy_for_corners = np.copy(img)
            img_copy_for_edges = np.copy(img)

            for rowindex, response in enumerate(harris_detect):
                for colindex, r in enumerate(response):
                    if r > 0:
                        # this is a corner
                        img_copy_for_corners[rowindex, colindex] = [255, 0, 0]
                    elif r < 0:
                        # this is an edge
                        img_copy_for_edges[rowindex, colindex] = [0, 255, 0]

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            cv2.imshow('corner', img_copy_for_corners)
            cv2.imshow('edge', img_copy_for_edges)
            # show_image(img_copy_for_corners, img_copy_for_edges)

    cv2.destroyAllWindows()
    cap.release()
