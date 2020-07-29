import numpy as np
import cv2
import skimage.measure
from PIL import Image


############# Numpy Implementation ###############
def entropy(img):
    hist = np.histogramdd(np.ravel(img), bins=256)[0] / img.size
    hist = list(filter(lambda p: p > 0, np.ravel(hist)))

    entropy = -np.sum(np.multiply(hist, np.log2(hist)))
    return entropy


def entropy_np(img, dehazed_img):
    return entropy(img) / entropy(dehazed_img)


##################################################


def entropy_sk(img, dehazed_img):
    entropy_gt = skimage.measure.shannon_entropy(img)
    entropy_dehazed = skimage.measure.shannon_entropy(dehazed_img)
    return entropy_dehazed / entropy_gt
