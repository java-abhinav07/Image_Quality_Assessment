# Import libraries
import cv2
import numpy as np 
from math import log

def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    max_brightness = 255.0
    psnr = 10 * log((max_brightness**2) / mse)

    return psnr


    
