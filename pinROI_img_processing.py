import torch
import cv2
import numpy as np
import time
import copy
import argparse
import matplotlib.pyplot as plt
# from loguru import logger
import glob
import emoji
from colorama import Fore, Style 
from ultralytics import YOLO
from pinROI import MODEL

def parseArgs():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--dataPath', type=str, default='',
                        help='Provide the path of data.txt file')
    parser.add_argument('--savePath', type=str, default='',
                        help='Provide a path to save the resultant image')
    args = parser.parse_args()
    return args

def brightnessCheck(img, type='img'):
    if type=='img':
        imgHSV  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(imgHSV)
        dim1, dim2 = v.shape
        return (np.sum(v))/(dim1*dim2)
    elif type=='v_channel':
        dim1, dim2 = img.shape
        return (np.sum(img))/(dim1*dim2)

def preProcess(img, thres=90):
    imgHSV  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imgHSV)
    dim1, dim2 = v.shape
    brightness = brightnessCheck(v, type='v_channel')

    diff = brightness - thres
    gamma = 1+(diff/25)
    gamma = gamma if gamma<1.7 else 1.7

    if brightness > thres:
        if brightness<100:
            v_processed = np.clip(255*(v / 255) ** gamma).astype(np.uint8)
        elif brightness>100:
            print(f"\t\t{(Fore.RED)}{(Style.BRIGHT)}{(emoji.emojize(':warning:'))} {' WARNING'} Pin is too muxh exposed, please adjust {Style.RESET_ALL}")
        
            v_scale     = np.clip(v*0.8, 0, 255)
            v_processed = np.clip(255*(v_scale / 255) ** gamma).astype(np.uint8)
    else:
        print('Brightness is Good')
        v_processed = v    

    img_processedHSV = cv2.merge([h,s,v_processed])
    img_processedRGB = cv2.cvtColor(img_processedHSV, cv2.COLOR_HSV2RGB)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # cv2.imshow('', h)
    # cv2.waitKey()
    # cv2.imshow('', img_processed)
    # cv2.waitKey()
    # cv2.destroyAllWindows()        

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Display images
    axes[0].imshow(imgRGB)
    axes[0].set_title("original image RGB")
    axes[0].axis("off")  # Hide axis

    axes[1].imshow(img_processedHSV)
    axes[1].set_title("processed image HSV")
    axes[1].axis("off")  # Hide axis

    axes[2].imshow(img_processedRGB)
    axes[2].set_title("processed image RGB")
    axes[2].axis("off")  # Hide axis

    axes[3].imshow(h)
    axes[3].set_title("Hue channel")
    axes[3].axis("off")  # Hide axis

    plt.tight_layout()  # Adjust spacing
    plt.show()

    print()


def process(img):

    # img_preProcessed = preProcess(img)
    preProcess(img)

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('', img_gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    print()

def getNameAndRoi(self, realName, meanROI):
    x1, y1, x2, y2 = meanROI

    roi_params = {
        "Top-pin-2nd_auto_0":  (-830, 15, 55, 920, 1890, 1910, -15, 550, 650, 8, 1275, 1290), # done
        "Top-pin-2nd_auto_1":  (-830, 30, 90, 860, 1890, 1915, -20, 550, 650, 8, 1275, 1290), # done
        "Top-pin_auto_0":  (-810, 0, 25, 870, 1790, 1870, -15, 420, 450, 35, 1260, 1275), # done
        "Top-pin_auto_1":  (-830, 0, 15, 890, 1810, 1860, -40, 430, 460, 45, 1260, 1275),  # done
        "Front-pin-2nd_auto_0": (-830, 25, 90, 910, 1900, 1915, -320, -3, 0, 150, 880, 910), # done
        "Front-pin-2nd_auto_1": (-830, 40, 90, 910, 1901, 1919, -320, -1, 3, 150, 880, 910), # done
        "Front-pin_auto_0": (-830, 1, 25, 910, 1880, 1910, -350, 40, 95, 150, 930, 990),   # done
        "Front-pin_auto_1": (-830, 1, 25, 910, 1870, 1900, -330, 40, 95, 150, 930, 990),   # done
    }

    for key, (dx1, min_x1, max_x1, dx2, min_x2, max_x2, dy1, min_y1, max_y1, dy2, min_y2, max_y2) in roi_params.items():
        if key==realName:
            _x1 = min(max(x1 + dx1, min_x1), max_x1)
            _x2 = min(max(x2 + dx2, min_x2), max_x2)
            _y1 = min(max(y1 + dy1, min_y1), max_y1)
            _y2 = min(max(y2 + dy2, min_y2), max_y2)
            roi = [_x1, _y1, _x2, _y2]
            break
    else:
        return (None, None, None, None)  # If no match is found

    mapping = {
        "Front-pin-2nd_auto_0": ("Front12", roi, -1, 21),
        "Front-pin-2nd_auto_1": ("Front22", roi, -1, 19),
        "Front-pin_auto_0": ("Front21", roi, -1, 19),
        "Front-pin_auto_1": ("Front11", roi, -1, 21),
        "Top-pin-2nd_auto_0": ("Top12", roi, -3, 21),
        "Top-pin-2nd_auto_1": ("Top22", roi, -3, 19),
        "Top-pin_auto_0": ("Top21", roi, -3, 19),
        "Top-pin_auto_1": ("Top11", roi, -3, 21),
    }

    return mapping.get(realName, (None, None, None, None))    

def main():
    args = parseArgs()

    print("Called with Args: ", args)

    dataPath = args.dataPath
    with open(dataPath, 'r') as f:
        imgPaths = f.readlines()

    for imgPath in imgPaths:
        imgPath = imgPath.strip()
        img = cv2.imread(imgPath)

        surfaceName = imgPath.split('Input-')[1].split('__Cam')[0]
        info={'Input':surfaceName}

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Run inference
        fig, axes = plt.subplots(1, 3, figsize=(15, 15))
        axes[0].imshow(img)
        axes[0].set_title("original image RGB")
        axes[0].axis("off")  # Hide axis

        axes[1].imshow(img_gray)
        axes[1].set_title("Gray image")
        axes[1].axis("off")  # Hide axis

        axes[2].imshow(binary_img, cmap=None)
        axes[2].set_title("Binarized image")
        axes[2].axis("off")  # Hide axis
        plt.tight_layout()
        plt.show()

        print()


if __name__=='__main__':
    main()