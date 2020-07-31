import argparse
import os
import numpy as np

# import cv2
from PIL import Image
import oct2py
import json
from oct2py import octave

from ssim import ssim_sk
from psnr import psnr
from brisque import brisque_imquality
from Information_Entropy import entropy_sk
from msssim import compute_msssim
from vif import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    type=str,
    default="/home/anil/Desktop/Dehazing_Abhinav/Evaluation/",
    help="path to root directory",
)  # contains subdirectories with GT+Test images for different datasets
args = parser.parse_args()

for dir in os.listdir(os.path.join(args.root, "Results")):
    print(dir)
    print("-------------------")
    for _, dataset in enumerate(os.listdir(os.path.join(args.root, "Results", dir))):
        print(str(_) + ". " + str(dataset))

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
            "UQI": 0,
            "RASE": 0,
            "CEIQ": 0,
            "ESSIM": 0,
            "FADE": 0,
            "GMSD": 0,
        }
        images_list = os.listdir(os.path.join(args.root, "Results", dir, dataset))
        extension = os.listdir(os.path.join(args.root, "GT", dataset))[0].split(".")[-1]
        num_images = len(images_list)
        for enhanced in images_list:
            gt_path = os.path.join(
                args.root, "GT", dataset, enhanced.split("_")[0] + "." + extension,
            )
            # print(gt_path)
            if os.path.isfile(gt_path):
                gt_image = Image.open(gt_path).convert("RGB")

            else:
                gt_image = None

            enhanced = os.path.join(args.root, "Results", dir, dataset, enhanced)
            enhanced_image = Image.open(enhanced).convert("RGB")

            if os.path.isfile(gt_path):
                h, w = enhanced_image.size
                hg, wg = gt_image.size
                gx = gt_image
                # print(w, h)
                # print(wg, hg)
                if wg > w:
                    gt_image = gt_image.resize((hg, w), Image.ANTIALIAS)

                if hg > h:
                    gt_image = gt_image.resize((h, wg), Image.ANTIALIAS)

                if w > wg:
                    enhanced_image = enhanced_image.resize((h, wg), Image.ANTIALIAS)

                if h > hg:
                    enhanced_image = enhanced_image.resize((hg, w), Image.ANTIALIAS)

                gt_image.show()
                enhanced_image.show()
                enhanced_image = np.asanyarray(enhanced_image)
                gt_original = np.asanyarray(gx)
                gt_image = np.asanyarray(gt_image)

            # ssim needs same sized images

            oc = oct2py.Oct2Py(temp_dir="./tmp/")

            # FR_IQA
            if os.path.isfile(gt_path):
                ssim, mse = ssim_sk(gt_image, enhanced_image)
                metrics["SSIM"] += ssim
                print("ssim:", metrics["SSIM"])

                psnr = psnr(gt_image, enhanced_image)
                metrics["PSNR"] += psnr
                print("psnr:", metrics["PSNR"])

                metrics["MSE"] += mse
                print("mse: ", metrics["MSE"])

                metrics["IE"] += entropy_sk(gt_image, enhanced_image)
                print("ie: ", metrics["IE"])

                metrics["VIF"] += compute_vif(gt_image, enhanced_image)
                print("vif: ", metrics["VIF"])

                metrics["MS-SSIM"] += compute_msssim(
                    np.array([gt_image]), np.array([enhanced_image]), max_val=255
                )
                print("msssim: ", metrics["MS-SSIM"])

                metrics["UQI"] += compute_uqi(gt_image, enhanced_image)
                print("uqi: ", metrics["UQI"])

                oc.addpath("./FSIM")
                metrics["FSIM"] += oc.FeatureSIM(gt_image, enhanced_image)
                print("fsim: ", metrics["FSIM"])

                oc.addpath("./Visible_Edges_Ratio")
                e1, ns1 = oc.EvaluationDescriptorCalculation(gt_path, enhanced)
                metrics["ns1"] += ns1
                metrics["e1"] += e1
                print("ver")

                oc.addpath("./ESSIM")
                metrics["ESSIM"] += oc.ESSIM(gt_image, enhanced_image)

                oc.addpath("./GMSD")
                metrics["GMSD"] += oc.GMSD(gt_image, enhanced_image)[0]

            # NR-IQA
            metrics["BRISQUE"] += brisque_imquality(enhanced_image)

            oc.addpath("./Bliinds2_code")
            bliinds_features = oc.bliinds2_feature_extraction(enhanced_image)
            metrics["BLINDS"] += oc.bliinds_prediction(bliinds_features)

            oc.addpath("./niqe_release")
            metrics["NIQE"] += oc.computequality(enhanced_image, 96, 96, 0, 0)

            oc.addpath("./SSEQ")
            metrics["SSEQ"] += oc.SSEQ(enhanced_image)

            oc.addpath("./CEIQ")
            metrics["CEIQ"] += oc.CEIQ(enhanced_image)

            oc.addpath("./FADE")
            metrics["FADE"] += oc.FADE(enhanced)

        for keys, values in metrics.items():
            metrics[keys] = values / num_images
            print(keys, metrics[keys])

        with open("{}_metrics.txt".format(dir), "a") as f:
            f.write(dataset)
            f.write(json.dumps(metrics))
            f.write("\n")

        print("finished writing {} to file.".format(dir))
        print("-----------------")
