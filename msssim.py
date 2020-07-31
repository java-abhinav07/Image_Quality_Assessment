#!/usr/bin/python
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.enable_eager_execution()


def compute_msssim(img1, img2, max_val=255):
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)
    out = tf.image.ssim_multiscale(img1, img2, max_val)
    out = out.numpy()
    return out[0]
