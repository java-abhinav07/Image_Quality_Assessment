import argparse
import os
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut


# import cv2
from PIL import Image, ImageOps
import oct2py
import json
from oct2py import octave
from tqdm import tqdm

from ssim import ssim_sk
from psnr import psnr
from brisque import *
from Information_Entropy import entropy_sk
from msssim import compute_msssim
from vif import *
from time import time

import csv


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

        # images_list = os.listdir(os.path.join(args.root, "Results", dir, dataset))
        try:
            extension = os.listdir(os.path.join(args.root, "GT", dataset))[0].split(
                "."
            )[-1]
        except:
            extension = "None"
        # num_images = len(images_list)
        oc1 = oct2py.Oct2Py(temp_dir="./tmp1/")
        oc1.addpath("./FSIM/")
        oc2 = oct2py.Oct2Py(temp_dir="./tmp2/")
        oc2.addpath("./ESSIM/")
        oc3 = oct2py.Oct2Py(temp_dir="./tmp3/")
        oc3.addpath("./GMSD/")
        oc4 = oct2py.Oct2Py(temp_dir="./tmp4/")
        oc4.addpath("./CEIQ/")
        oc5 = oct2py.Oct2Py(temp_dir="./tmp5/")
        oc5.addpath("./FADE/")

        for enhanced in tqdm(
            os.listdir(os.path.join(args.root, "Results", dir, dataset))
        ):

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
                w, h = enhanced_image.size
                wg, hg = gt_image.size
                desired_ratio = wg / hg
                gx = gt_image
                # print(wg, hg)
                if wg > 640:
                    wpercent = 640 / gt_image.size[0]
                    hsize = int((float(gt_image.size[1]) * float(wpercent)))
                    gt_image = gt_image.resize((640, hsize), Image.ANTIALIAS)
                    enhanced_image = enhanced_image.resize(
                        gt_image.size, Image.ANTIALIAS
                    )
                else:
                    enhanced_image = enhanced_image.resize(
                        gt_image.size, Image.ANTIALIAS
                    )

                assert enhanced_image.size == gt_image.size
                # print(enhanced_image.size)

                # gt_image.show()
                # enhanced_image.show()

                gt_gray = ImageOps.grayscale(gt_image)
                gt_image = np.asanyarray(gt_image)
                gt_gray = np.asanyarray(gt_gray)

                # print(gt_gray.size, enhanced_gray.size)
            # ssim needs same sized images
            enhanced_gray = ImageOps.grayscale(enhanced_image)
            enhanced_image = np.asanyarray(enhanced_image)
            enhanced_gray = np.asanyarray(enhanced_gray)

            # FR_IQA
            if os.path.isfile(gt_path):

                ssim, mse = ssim_sk(gt_image, enhanced_image)

                psnr_val = psnr(gt_image, enhanced_image)

                try:
                    ie = entropy_sk(gt_image, enhanced_image)
                except:
                    ie = "null"

                try:
                    vif = compute_vif(gt_image, enhanced_image)
                except:
                    vif = "null"

                ms_ssim = compute_msssim(
                    np.array([gt_image]), np.array([enhanced_image]), max_val=255
                )

                try:
                    uqi = compute_uqi(gt_image, enhanced_image)
                except:
                    uqi = "null"

                try:
                    fsim = func_timeout(
                        10, oc1.FeatureSIM, args=(gt_image, enhanced_image)
                    )
                except FunctionTimedOut:
                    fsim = "null"
                except Exception as e:
                    fsim = "null"

                # oc.addpath("./Visible_Edges_Ratio")
                # e1, ns1 = oc.EvaluationDescriptorCalculation(gt_path, enhanced)
                # metrics["ns1"] += ns1
                # metrics["e1"] += e1
                # print("ver")
                try:
                    essim = func_timeout(7, oc2.ESSIM, args=(gt_image, enhanced_image))
                except FunctionTimedOut:
                    essim = "null"
                except Exception as e:
                    essim = "null"

                try:
                    gmsd = func_timeout(7, oc3.GMSD, args=(gt_gray, enhanced_gray))
                except FunctionTimedOut:
                    gmsd = "null"
                except Exception as e:
                    gmsd = "null"

            # NR-IQA
            t = 7
            #
            # get_flag(t)

            bris += brisque_imquality(enhanced_image)

            # oc.addpath("./Bliinds2_code")
            # bliinds_features = oc.bliinds2_feature_extraction(enhanced_image)
            # metrics["BLINDS"] += oc.bliinds_prediction(bliinds_features)
            # print("bliinds2: ", metrics["BLINDS"])

            # oc.addpath("./niqe_release")
            # metrics["NIQE"] += oc.computequality(enhanced_image, 96, 96, 0, 0)
            # print("niqe: ", metrics["NIQE"])

            if not os.path.isfile(gt_path):
                try:
                    ceiq = func_timeout(t, oc4.CEIQ, args=(enhanced_image,))
                except FunctionTimedOut:
                    ceiq = "null"
                except Exception as e:
                    ceiq = "null"

                # print("ceiq: ", metrics["CEIQ"])

                # t = time()
                try:
                    fade = func_timeout(t, oc5.FADE, args=(enhanced,))
                except FunctionTimedOut:
                    fade = "null"
                except Exception as e:
                    fade = "null"

            # oc.addpath("./SSEQ")
            # try:
            #     sseq = func_timeout(t, oc.SSEQ, args=(enhanced_image, ))
            #     metrics["SSEQ"] += sseq
            # except FunctionTimedOut:
            #     pass
            # except Exception as e:
            #     raise(e)

            # print(time()-t)
            # print("fade: ", metrics["FADE"])

            # one csv for each technique
            fields = [
                "Name",
                "SSIM",
                "PSNR",
                "MSE",
                "MS-SSIM",
                "UQI",
                "FSIM",
                "GMSD",
                "ESSIM",
                "VIF",
                "IE",
                "BRISQUE",
                "FADE",
                "CEIQ",
            ]

            rows = [
                enhanced + "_{}".format(dataset),
                ssim,
                psnr_val,
                mse,
                ms_ssim,
                uqi,
                fsim,
                gmsd,
                essim,
                vif,
                ie,
                bris,
                fade,
                ceiq,
            ]

            with open("{}_metrics.txt".format(dir), "a") as f:
                csvwriter = csv.writer(f)
                if _ == 0:
                    csvwriter.writerow(fields)
                csvwriter.writerows(rows)

    print("finished writing {} to file.".format(dir))
    print("-----------------")
