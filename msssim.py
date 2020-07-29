#!/usr/bin/python
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
import tensorflow as tf


def compute_msssim(
    img1,
    img2,
    max_val=255
):
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)
    return tf.image.ssim_multiscale(img1, img2, max_val).numpy()[0]
