import imquality.brisque as brisque
from PIL import Image
from timeout import timeout
########## Using image-quality library ################
def get_flag(*args):
    return args

@timeout(get_flag())
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
