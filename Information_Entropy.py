import numpy as np
import cv2
import skimage.measure
from PIL import Image


############# Numpy Implementation ###############
def entropy_np(img):
    hist = np.histogramdd(np.ravel(img), bins=256)[0] / img.size
    hist = list(filter(lambda p: p > 0, np.ravel(hist)))

    entropy = -np.sum(np.multiply(hist, np.log2(hist)))
    return entropy


##################################################


def entropy_sk(img):
    entropy = skimage.measure.shannon_entropy(img)
    return entropy
