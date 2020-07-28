# This code is adapted from https://cvnote.ddlee.cn/2019/09/12/psnr-ssim-python
import numpy as np
import cv2
import math

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


###################### Numpy Implementation #############################
def ssim_np(reference_img, test_img):

    img1 = reference_img
    img2 = test_img

    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")

    elif img1.ndim == 2:
        return compare(img1, img2)

    elif img1.ndim == 3:
        # RGB
        if img1.shape[2] == 3:
            ssim = []
            for i in range(3):
                ssim.append(compare(img1, img2))
            return np.array(ssim).mean()
        # grayscale
        elif img1.shape[2] == 1:
            return compare(np.squeeze(img1), np.squeeze(img2))


def compare(img1, img2, C1=6.5025, C2=58.5225):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    gk = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(kernel, kernel.transpose())

    u1 = cv2.filter2D(img1, -1, win)[5:-5, 5:-5]
    u2 = cv2.filter2D(img2, -1, win)[5:-5, 5:-5]

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, win)[5:-5, 5:-5] - (u1 ** 2)
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, win)[5:-5, 5:-5] - (u2 ** 2)
    sigma12 = cv2.filter2D(img1 * img2, -1, win)[5:-5, 5:-5] - (u1 * u2)

    ssim_map = ((2 * (u1 * u2) + C1) * (2 * sigma12 + C2)) / (
        ((u1 ** 2) + (u2 ** 2) + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


#############################################################################3


def ssim_sk(reference_img, test_img):
    return ssim(reference_img, test_img), mean_squared_error(reference_img, test_img)
