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
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

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
            "PSNR": [0, 0],
            "SSIM": [0, 0],
            "FSIM": [0, 0],
            "BLINDS": [0, 0],
            "BRISQUE": [0, 0],
            "MS-SSIM": [0, 0],
            "IE": [0, 0],
            "e1": [0, 0],
            "ns1": [0, 0],
            "NIQE": [0, 0],
            "SSEQ": [0, 0],
            "VIF": [0, 0],
            "MSE": [0, 0],
            "UQI": [0, 0],
            "RASE": [0, 0],
            "CEIQ": [0, 0],
            "ESSIM": [0, 0],
            "FADE": [0, 0],
            "GMSD": [0, 0],
        }
        # images_list = os.listdir(os.path.join(args.root, "Results", dir, dataset))
        try:
            extension = os.listdir(os.path.join(args.root, "GT", dataset))[0].split(".")[-1]
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

        for enhanced in tqdm(os.listdir(os.path.join(args.root, "Results", dir, dataset))):

            gt_path = os.path.join(
                args.root, "GT", dataset, enhanced.split("_")[0] + "." + extension,
            )
            # print(gt_path)
            if os.path.isfile(gt_path):
                gt_image = Image.open(gt_path).convert("RGB")

            else:
                gt_image = None

            try:
                enhanced = os.path.join(args.root, "Results", dir, dataset, enhanced)
                enhanced_image = Image.open(enhanced).convert("RGB")
            except:
                # print("encountered truncated image, skipping")
                continue

            try:
                if os.path.isfile(gt_path):
                    w, h = enhanced_image.size
                    wg, hg = gt_image.size
                    desired_ratio = wg/hg
                    gx = gt_image
                    # print(wg, hg)
                    # print(w, h)
                    if wg > 640:
                        wpercent = (640/gt_image.size[0])
                        hsize = int((float(gt_image.size[1])*float(wpercent)))
                        gt_image = gt_image.resize((640, hsize), Image.ANTIALIAS)
                        enhanced_image = enhanced_image.resize(gt_image.size, Image.ANTIALIAS)
                    elif enhanced_image.size!=gt_image.size:
                        if wg < w:
                            enhanced_image = enhanced_image.resize(gt_image.size, Image.ANTIALIAS)
                        else:
                            gt_image = gt_image.resize(enhanced_image.size, Image.ANTIALIAS)

                    assert(enhanced_image.size==gt_image.size)
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
                    metrics["SSIM"][0] += ssim
                    metrics["SSIM"][1] += 1

                    psnr_val = psnr(gt_image, enhanced_image)
                    metrics["PSNR"][0] += psnr_val
                    metrics["PSNR"][1] += 1

                    metrics["MSE"][0] += mse
                    metrics["MSE"][1] += 1
                    try:
                        metrics["IE"][0] += entropy_sk(gt_image, enhanced_image)
                        metrics["IE"][1] += 1
                    except:
                        pass

                    try:
                        metrics["VIF"][0] += compute_vif(gt_image, enhanced_image)
                        metrics["VIF"][1] += 1
                    except:
                        pass

                    metrics["MS-SSIM"][0] += compute_msssim(
                        np.array([gt_image]), np.array([enhanced_image]), max_val=255
                    )
                    metrics["MS-SSIM"][1] += 1

                    try:
                        metrics["UQI"][0] += compute_uqi(gt_image, enhanced_image)
                        metrics["UQI"][1] += 1
                    except:
                        pass


                    try:
                        fsim = func_timeout(10, oc1.FeatureSIM, args=(gt_image, enhanced_image))
                        metrics["FSIM"][0] += fsim
                        metrics["FSIM"][1] += 1
                    except FunctionTimedOut:
                        pass
                    except Exception as e:
                        pass


                    # oc.addpath("./Visible_Edges_Ratio")
                    # e1, ns1 = oc.EvaluationDescriptorCalculation(gt_path, enhanced)
                    # metrics["ns1"] += ns1
                    # metrics["e1"] += e1
                    # print("ver")
                    try:
                        essim = func_timeout(7, oc2.ESSIM, args=(gt_image, enhanced_image))
                        metrics["ESSIM"][0] += essim
                        metrics["ESSIM"][1] += 1
                    except FunctionTimedOut:
                        pass
                    except Exception as e:
                        pass

                    try:
                        gmsd = func_timeout(7, oc3.GMSD, args=(gt_gray, enhanced_gray))
                        metrics["GMSD"][0] += gmsd
                        metrics["GMSD"][1] += 1
                    except FunctionTimedOut:
                        pass
                    except Exception as e:
                        pass



                # NR-IQA
                t = 7
                #
                # get_flag(t)

                metrics["BRISQUE"][0] += brisque_imquality(enhanced_image)
                metrics["BRISQUE"][1] += 1

                # try:
                #     metrics["BRISQUE"][0] += brisque_imquality(enhanced_image)
                #     metrics["BRISQUE"][1] += 1
                # except:
                #     pass
                # print("brisque: ", metrics["BRISQUE"])

                # oc.addpath("./Bliinds2_code")
                # bliinds_features = oc.bliinds2_feature_extraction(enhanced_image)
                # metrics["BLINDS"] += oc.bliinds_prediction(bliinds_features)
                # print("bliinds2: ", metrics["BLINDS"])

                # oc.addpath("./niqe_release")
                # metrics["NIQE"] += oc.computequality(enhanced_image, 96, 96, 0, 0)
                # print("niqe: ", metrics["NIQE"])


                # print("sseq: ", metrics["SSEQ"])

                # oc.addpath("./CEIQ")
                # metrics["CEIQ"] += oc4.CEIQ(enhanced_image)
                if not os.path.isfile(gt_path):
                    try:
                        ceiq = func_timeout(t, oc4.CEIQ, args=(enhanced_image, ))
                        metrics["CEIQ"][0] += ceiq
                        metrics["CEIQ"][1] += 1
                    except FunctionTimedOut:
                        pass
                    except Exception as e:
                        pass

                    # print("ceiq: ", metrics["CEIQ"])

                    # t = time()
                    try:
                        density = func_timeout(t, oc5.FADE, args=(enhanced,))
                        metrics["FADE"][0] += density
                        metrics["FADE"][1] += 1
                    except FunctionTimedOut:
                        pass
                    except Exception as e:
                        pass

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
            except:
                continue

        for keys, values in metrics.items():
            if values[1] > 0:
                metrics[keys][0] = values[0] / values[1]
                print(keys, metrics[keys][0])

        with open("{}_metrics.txt".format(dir), "a") as f:
            f.write(dataset)
            f.write(json.dumps(metrics))
            f.write("\n")

    print("finished writing {} to file.".format(dir))
    print("-----------------")
