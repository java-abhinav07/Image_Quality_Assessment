import sewar.full_ref
import numpy as np
from timeout import timeout
# import cv2

@timeout()
def compute_vif(gt, enhanced):
    return sewar.full_ref.vifp(gt, enhanced)

@timeout()
def compute_uqi(gt, enhanced):  # universal image quality index
    return sewar.full_ref.uqi(gt, enhanced)

@timeout()
def compute_rase(gt, enhanced):  # relative average spectral error
    return sewar.full_ref.rase(gt, enhanced)
