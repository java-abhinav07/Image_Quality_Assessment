import cv2
import numpy as np
from math import log


def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_brightness = 255.0
    psnr = 10 * log((max_brightness ** 2) / mse)

    return psnr, mse
