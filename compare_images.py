# Runs comparitive test using the following metrics:
# 1. SSEQ
# 2. BLINDS
# 3. NIQE
# 4. SSIM
# 5. PSNR
# 6. VIF
# 7. VER
# 8. FSIM
# 9. IE
# 10. MS-SSIM
# 11. BRISQUE

import argparse
import os
import numpy as np
import cv2
from PIL import Image
import oct2py
import json
from oct2py import octave as oc

from ssim import ssim_sk
from psnr import psnr
from brisque import brisque_imquality
from Information_Entropy import entropy_sk


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root", type=str, default="", help="path to root directory"
)  # contains subdirectories with GT+Test images for different datasets
args = parser.parse_args()

for dir in os.listdir(args.root):
    print(dir)
    print("----------------------------------------------------------")
    for dataset in os.listdir(os.path.join(args.root, dir)):
        print(dir, " -----> ", dataset)

        metrics = {
            "PSNR": 0,
            "SSIM": 0,
            "FSIM": 0,
            "BLINDS": 0,
            "BRISQUE": 0,
            "MS-SSIM": 0,
            "IE": 0,
            "e1": 0,
            "ns1": 0,
            "NIQE": 0,
            "SSEQ": 0,
            "VIF": 0,
            "MSE": 0,
        }
        images_list = os.listdir(os.path.join(args.root, dir, dataset, "Results"))
        num_images = len(images_list)
        for dehazed in images_list:
            gt_path = os.path.join(
                args.root,
                dir,
                dataset,
                "GT",
                dehazed.split("_")[0] + dehazed.split(".")[-1],
            )
            if gt_path.isdir():
                gt_image = Image.open(gt_path)
            else:
                gt_image = None

            dehazed = os.path.join(args.root, dir, dataset, "Results", dehazed)
            dehazed_image = Image.open(dehazed)

            ssim, mse = ssim_sk(gt_path, dehazed_image)

            metrics["SSIM"] += ssim
            metrics["PSNR"] += psnr(gt_image, dehazed_image)[0]
            metrics["BRISQUE"] += brisque_imquality(dehazed_image)
            metrics["MSE"] += mse
            metrics["IE"] += entropy_sk(dehazed_image)

            oc.addpath("./Bliinds2_code")
            bliinds_features = oc.bliinds2_feature_extraction(dehazed_image)
            metrics["BLINDS"] += oc.bliinds_prediction(bliinds_features)

            oc.addpath("./Multi_Scale_SSIM")
            metrics["MS-SSIM"] += oc.ssim_index_new(dehazed_image, gt_image)[0]

            oc.addpath("./Visual_Image_Fidelity")
            metrics["VIF"] += oc.vifvec(gt_image, dehazed_image)

            metrics["FSIM"] += oc.FeatureSIM(gt_image, dehazed_image)

            oc.addpath("./Visible_Edges_Ratio")
            e1, ns1 = oc.visible_edge_ratio(gt_path, dehazed)
            metrics["ns1"] += ns1
            metrics["e1"] += e1

            oc.addpath("./niqe_release")
            metrics["NIQE"] += oc.computequality(dehazed_image, 96, 96, 0, 0)

            oc.addpath("./SSEQ")
            metrics["SSEQ"] += oc.SSEQ(dehazed_image)

        for keys, values in metrics.items():
            metrics[keys] = values / num_images
            print(keys, metrics[keys])

        with open("{}_metrics".format(dir), "a") as f:
            f.write(dataset)
            f.write(json.dumps(metrics))
            f.write()

        print("finished writing {} to file.".format(dir))
        print("-----------------------------------------------------------------------")
