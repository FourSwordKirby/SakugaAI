import cv2
import numpy as np
import sys
import os
from PIL import Image

def EdgeDetect(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    canny = cv2.Canny(img, 50, 240)

    # cv2.imshow('Canny', canny)
    return Image.fromarray(canny)

# for arg in vars(args):
#     print('[%s] =' % arg, getattr(args, arg))

for filename in os.listdir(sys.argv[1]):
    image = EdgeDetect(filename)
    image.save(sys.argv[2] + "/" + filename)

# EdgeDetect('test2.png')
# EdgeDetect('test3.png')
# EdgeDetect('test4.png')
# EdgeDetect('test5.png')