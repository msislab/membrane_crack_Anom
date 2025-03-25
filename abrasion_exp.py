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
    parser.add_argument('--roiModel', type=str, default=None,
                        help='Provide model.pt path')
    parser.add_argument('--roiConf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose a gpu device')
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

def main():
    args = parseArgs()

    print("Called with Args: ", args)

    roiModel = MODEL(model_path=args.roiModel, confidence_threshold=args.roiConf)
    dataPath = args.dataPath
    with open(dataPath, 'r') as f:
        imgPaths = f.readlines()

    for imgPath in imgPaths:
        imgPath = imgPath.strip()
        img = cv2.imread(imgPath)

        surfaceName = imgPath.split('Input-')[1].split('__Cam')[0]
        info={'Input':surfaceName}

        # Run inference
        Pin_img, pinROI_box, Pinmask, burImg, burROI_box, Burmask, pinPreds = roiModel.process(img, info=info)

        
        sorted_indices = np.argsort(pinPreds[:, 6])
        pinPreds = pinPreds[sorted_indices]
        print(len(pinPreds))

        for i, pin in enumerate(pinPreds):
            x4,y4,x3,y3,x2,y2,x1,y1 = map(int, pin)
            pinPatch = Pin_img[y1-2:y3+2, x1-10:x2+10]

            pinBrightness = brightnessCheck(pinPatch)

            print(f'Brightness level of Pin {i} is:', round(pinBrightness, 4))

            process(pinPatch)

            # cv2.imshow(f'{pinBrightness}', pinPatch)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            print()


if __name__=='__main__':
    main()