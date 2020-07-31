from PIL import Image
import numpy as np
import os


root = "/home/anil/Desktop/Dehazing_Abhinav/Results/AOD/Results/"

l = [
    root + "D-Hazy/Middlebury",
    root + "D-Hazy/NYU",
    root + "frida2",
    root + "HazeRD",
    root + "RESIDE_SOTS/outdoor",
    root + "RESIDE_SOTS/indoor",
    root + "RESIDE_HSTS/real_world",
    root + "RESIDE_HSTS/synthetic",
    root + "I-Haze",
    root + "O-Haze",
]


for dir in l:
    for im in os.listdir(dir):
        print(im)
        image = Image.open(os.path.join(dir, im))
        w, h = image.size

        cropped = image.crop((w // 2, 0, w, h))
        # cropped.show()
        # break
        cropped.save(os.path.join(dir, im), quality=100)
