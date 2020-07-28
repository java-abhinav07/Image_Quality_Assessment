import imquality.brisque as brisque
from PIL import Image

########## Using image-quality library ################
def brisque_imquality(img):
    return brisque.score(img)


#######################################################

# the following code is highly inspired by https://github.com/ocampor/notebooks/blob/master/notebooks/image/quality/brisque.ipynb

# import collections
# from itertools import chain
# import urllib.request as request
# import pickle

# import numpy as np

# import scipy.signal as signal
# import scipy.special as special
# import scipy.optimize as optimize

# import matplotlib.pyplot as plt

# import skimage.io
# import skimage.transform

# import cv2

# from libsvm import svmutil
