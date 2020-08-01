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
        try:
            extension = os.listdir(os.path.join(args.root, "GT", dataset))[0].split(".")[-1]
        except:
            extension = "None"
        num_images = len(images_list)
        for enhanced in tqdm(images_list):

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

                h, w = enhanced_image.size
                hg, wg = gt_image.size
                if hg > h:
                    gt_image = gt_image.resize((h, wg), Image.ANTIALIAS)
                h, w = enhanced_image.size
                hg, wg = gt_image.size
                if w > wg:
                    enhanced_image = enhanced_image.resize((h, wg), Image.ANTIALIAS)
                h, w = enhanced_image.size
                hg, wg = gt_image.size
                if h > hg:
                    enhanced_image = enhanced_image.resize((hg, w), Image.ANTIALIAS)

                h, w = enhanced_image.size
                hg, wg = gt_image.size

                assert(h==hg)
                assert(wg==w)

                if h*w > 250000:
                    gt_image = gt_image.resize((480, 640), Image.ANTIALIAS)
                    enhanced_image = enhanced_image.resize((480, 640), Image.ANTIALIAS)

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
            oc = oct2py.Oct2Py(temp_dir="./tmp/")

            # FR_IQA
            if os.path.isfile(gt_path):

                ssim, mse = ssim_sk(gt_image, enhanced_image)
                metrics["SSIM"] += ssim

                psnr_val = psnr(gt_image, enhanced_image)
                metrics["PSNR"] += psnr_val

                metrics["MSE"] += mse
                try:
                    metrics["IE"] += entropy_sk(gt_image, enhanced_image)
                except:
                    pass

                try:
                    metrics["VIF"] += compute_vif(gt_image, enhanced_image)
                except:
                    pass

                metrics["MS-SSIM"] += compute_msssim(
                    np.array([gt_image]), np.array([enhanced_image]), max_val=255
                )

                try:
                    metrics["UQI"] += compute_uqi(gt_image, enhanced_image)
                except:
                    pass

                try:
                    oc.addpath("./FSIM")
                    fsim = func_timeout(7, oc.FeatureSIM, args=(gt_image, enhanced_image))
                    metrics["FSIM"] += fsim
                except FunctionTimedOut:
                    pass
                except Exception as e:
                    raise(e)


                # oc.addpath("./Visible_Edges_Ratio")
                # e1, ns1 = oc.EvaluationDescriptorCalculation(gt_path, enhanced)
                # metrics["ns1"] += ns1
                # metrics["e1"] += e1
                # print("ver")

                try:
                    oc.addpath("./ESSIM")
                    essim = func_timeout(7, oc.ESSIM, args=(gt_image, enhanced_image))
                    metrics["ESSIM"] += essim
                except FunctionTimedOut:
                    pass
                except Exception as e:
                    raise(e)

                try:
                    oc.addpath("./GMSD")
                    gmsd = func_timeout(7, oc.GMSD, args=(gt_gray, enhanced_gray))
                    metrics["GMSD"] += gmsd
                except FunctionTimedOut:
                    pass
                except Exception as e:
                    raise(e)



            # NR-IQA
            if not os.path.isfile(gt_path):
                t = 15
            else:
                t = 1

            get_flag(t)

            try:
                metrics["BRISQUE"] += brisque_imquality(enhanced_image)
            except:
                pass
            # print("brisque: ", metrics["BRISQUE"])

            # oc.addpath("./Bliinds2_code")
            # bliinds_features = oc.bliinds2_feature_extraction(enhanced_image)
            # metrics["BLINDS"] += oc.bliinds_prediction(bliinds_features)
            # print("bliinds2: ", metrics["BLINDS"])

            # oc.addpath("./niqe_release")
            # metrics["NIQE"] += oc.computequality(enhanced_image, 96, 96, 0, 0)
            # print("niqe: ", metrics["NIQE"])
            try:
                oc.addpath("./SSEQ")
                sseq = func_timeout(t, oc.SSEQ, args=(enhanced_image, ))
                metrics["SSEQ"] += sseq
            except FunctionTimedOut:
                pass
            except Exception as e:
                raise(e)

            # print("sseq: ", metrics["SSEQ"])
            oc.addpath("./CEIQ")
            metrics["CEIQ"] += oc.CEIQ(enhanced_image)

            # print("ceiq: ", metrics["CEIQ"])

            # t = time()
            try:
                oc.addpath("./FADE")
                density = func_timeout(t, oc.FADE, args=(enhanced,))
                metrics["FADE"] += density
            except FunctionTimedOut:
                pass
            except Exception as e:
                raise(e)

            # print(time()-t)
            # print("fade: ", metrics["FADE"])

        for keys, values in metrics.items():
            metrics[keys] = values / num_images
            print(keys, metrics[keys])

        with open("{}_metrics.txt".format(dir), "a") as f:
            f.write(dataset)
            f.write(json.dumps(metrics))
            f.write("\n")

    print("finished writing {} to file.".format(dir))
    print("-----------------")
