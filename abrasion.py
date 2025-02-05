import cv2, os, argparse, tqdm, glob
import numpy as np
from natsort import natsorted
from ultralytics import YOLO
import torch
from shapely.geometry import Polygon
import time
import colorama
from colorama import Style, Fore, Back
colorama.init(autoreset=True)
from roi import ROI_model
import copy
from patchify import patchify
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def drawBox(img, boxes, obb=False, boxType='xywh', color=(0,255,0), thickness=1):
    if obb:
        # print('visualizing obb preds')
        for box in boxes:
            _box = np.array([[int(box[0]), int(box[1])],
                             [int(box[2]), int(box[3])],
                             [int(box[4]), int(box[5])],
                             [int(box[6]), int(box[7])]])
            _box = _box.reshape(-1,1,2)
            img  = cv2.polylines(img, [_box], isClosed=True, color=color, thickness=thickness)
            # text = f"{box[0]};{box[-1]}"
            # (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
            # label_bg_top_left = (int(box[1]), int(box[2]) - h - 5)
            # label_bg_bottom_right = (int(box[1]) + w, int(box[2]))
            # cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
            # cv2.putText(img, text, (box[1]), (int(box[2]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)
    else:
        if boxType=='xywh':
            for box in boxes:
                x1 = int(box[1] - box[3]/2)
                x2 = int(box[1] + box[3]/2)
                y1 = int(box[2] - box[4]/2)
                y2 = int(box[2] + box[4]/2)

                img  = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                # text = f'{box[0]}:'
                # (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                # label_bg_top_left = (x1, y1 - h - 5)
                # label_bg_bottom_right = (x1 + w, y1)
                # cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                # cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)

        elif boxType=='xyxy':   # TODO: implement the box rendering for xyxy box coordinates
            for box in boxes:
                x1 = int(box[0])
                x2 = int(box[2])
                y1 = int(box[1])
                y2 = int(box[3])

                img  = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                # text = f'{box[0]}:'
                # (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                # label_bg_top_left = (x1, y1 - h - 5)
                # label_bg_bottom_right = (x1 + w, y1)
                # cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                # cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)      
    return img 

def processOBB(yoloOBB_rslt=[], patchPosz=None):
    preds = []
    for i,result in enumerate(yoloOBB_rslt):
        x1, y1, x2, y2 = patchPosz[i]
        w, h = x2-x1, y2-y1
        boxes  = result.obb.xyxyxyxyn.cpu().numpy().reshape(-1,8)
        boxes[:,::2]  *= w
        boxes[:,1::2] *= h
        clsz   = result.obb.cls.cpu().numpy()
        conf   = result.obb.conf.cpu().numpy()
        # _preds = np.hstack([clsz[:,None], boxes, conf[:,None]])
        _preds = np.hstack([boxes, conf[:,None]])
        preds.append(_preds)
        # img    = result.plot()
        # cv2.imshow('', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    # preds = np.delete(preds, 0, axis=0)
    return preds

def patchToimPred(preds=[], patchCoords=[]):
    imPreds = np.zeros((1,9))
    for i, _preds in enumerate(preds):
        x1,y1, x2,y2 = patchCoords[i]
        _preds[:,:-1:2]  += x1
        _preds[:,1:-1:2] += y1
        imPreds = np.vstack((imPreds, _preds))
    return imPreds[imPreds[:,-1]>0]

def getPin(img, pinOBB, kernel):
    pinTop    = [int(pinOBB[6])-5, int(pinOBB[7])]
    pinBottom = [int(pinOBB[2])+5, int(pinOBB[3])]
    pinImg    = img[pinTop[1]:pinBottom[1], pinTop[0]:pinBottom[0]]
    h,_,_     = pinImg.shape
    adjustedH = int(h*0.95)
    pinMask   = cv2.cvtColor(pinImg, cv2.COLOR_BGR2GRAY)

    pinMask[pinMask>0] = 255
    pinMask = cv2.morphologyEx(pinMask, cv2.MORPH_CLOSE, kernel)
    pinMask[adjustedH:,:] = 0
    pinImg = cv2.bitwise_and(pinImg,pinImg,mask=pinMask)
    # pinImg = pinImg[pinImg>0]
    return pinImg

def getBrightnessLevel(img):
    sum1 = np.sum(img[:,:,0])
    sum2 = np.sum(img[:,:,1])
    sum3 = np.sum(img[:,:,2])
    cum_sum = sum1 + sum2 + sum3
    # Formula to calculate the brightness of an image:
    # 2 * (cumulative sum of all pixel intensities(in every channels) / total pixels in image x 3x 255)
    brightnessLevel = cum_sum/(img.shape[0]*img.shape[1]*3*255)
    return brightnessLevel

def show2(img1, img2):
    h, w = img1.shape
    img2 = cv2.resize(img2, (w,h))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    dispImg = np.hstack((img1, img2))
    cv2.imshow('', dispImg)
    cv2.waitKey()
    cv2.destroyAllWindows()

def checkPin(pinImg, kernel):
    AnomalyFlag = False
    _pinImg = cv2.resize(pinImg, (212, 612))
    _pinImgGray = cv2.cvtColor(_pinImg, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('', _pinImg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    _pinImgGray[_pinImgGray>220] = 255
    _pinImgGray[_pinImgGray<255] = 0
    _pinImgEro = cv2.morphologyEx(_pinImgGray, cv2.MORPH_ERODE, kernel) 
    show2(_pinImgEro, _pinImg)
    # hist, bin_edges = np.histogram(_pinImgGray, bins=32, range=(0, 256))

    # # Plot the histogram
    # plt.bar(bin_edges[:-1], hist, width=256/32, color='gray', edgecolor='black')
    # plt.yticks(np.linspace(0, 10000, num=10))
    # plt.title(f"Grayscale Histogram with {32} Bins")
    # plt.xlabel("Pixel Intensity")
    # plt.ylabel("Frequency")

    # # Show the plot
    # plt.grid(True)
    # plt.show(block=True)
    # cv2.imshow('', _pinImg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return AnomalyFlag

def parseArgs():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--dataPath', type=str, default='',
                        help='Provide img directory path')
    # parser.add_argument('--detModel', type=str, default=None,
    #                     help='Provide model.pt path')
    parser.add_argument('--roiModel', type=str, default=None,
                        help='Provide model.pt path')
    # parser.add_argument('--detConf', type=float, default=0.5,
    #                     help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--roiConf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose a gpu device')
    # parser.add_argument('--imgSize', type=int, default=640,
    #                     help='Choose box type for annotation')
    args = parser.parse_args()
    return args

def main():
    args   = parseArgs()
    imgs   = glob.glob(f'{args.dataPath}/*.png')
    imgs   = natsorted(imgs)
    kernel =  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    roiModel = ROI_model(args.roiModel)

    # model.to(device)
    saveDir = "/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Abrasion-Scratch_pindata/pinImgs"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    idx = 0

    for imgPath in tqdm.tqdm(imgs):
        args.path = imgPath

        if 'Front-pin_auto_1' in imgPath:       # front surface 1, focus on first row
            surface = 'Front11'
        elif 'Front-pin-2nd_auto_0' in imgPath:     # front surface 1, focus on second row
            surface = 'Front12'    
        elif 'Front-pin_auto_0' in imgPath:     # front surface 2, focus on first row
            surface = 'Front21'
        elif 'Front-pin-2nd_auto_1' in imgPath:     # front surface 2, focus on second row
            surface = 'Front22'    
        elif 'Top-pin_auto_1' in imgPath:       # top surface 1, focus on first row
            surface = 'Top11'
        elif 'Top-pin-2nd_auto_0' in imgPath:       # top surface 1, focus on second row
            surface = 'Top12'    
        elif 'Top-pin_auto_0' in imgPath:       # top surface 2, focus on first row     
            surface = 'Top21'
        elif 'Top-pin-2nd_auto_1' in imgPath:       # top surface 2, focus on second row
            surface = 'Top22'

        img = cv2.resize(cv2.imread(imgPath), (1920,1280))
        # _img = copy.deepcopy(img)
        fullPin_img, Pin_img, Pin_img2, pinBox, pin1, pin2, fullMask = roiModel.pinROI(img=img, surface=surface)

        # cv2.imshow('', Pin_img)
        # cv2.waitKey()
        # # cv2.imshow('', _img)
        # # cv2.waitKey()
        # cv2.destroyAllWindows()
        # _img = drawBox(_img, np.array([ROIbox]), boxType='xyxy')
        sorted_indices = np.argsort(pin1[:, 6])
        pinPreds = pin1[sorted_indices]
        decision = []
        for i,pin in enumerate(pinPreds):
            saveName = f"{saveDir}/{idx}.png"
            idx+=1
            pinImg = getPin(fullPin_img, pin, kernel)
            pinImg = cv2.resize(pinImg, (640,640))
            cv2.imwrite(saveName, pinImg)
            # l = getBrightnessLevel(pinImg)
            # if l>0.53 and pinImg.shape[1]<60:
            #     brightnessDiff = round(((l-0.52)/0.52), 3)
            #     print()
            # elif l>0.59 and pinImg.shape[1]>60:
            #     pass
            # pinImg = cv2.resize(pinImg, (200, pinImg.shape[0]), cv2.INTER_CUBIC)
            # fullPin_img = drawBox(fullPin_img, np.array([pin]), obb=True)
            # cv2.imshow(f'{round(l,2), pinImg.shape[1]}', pinImg)
            # cv2.imshow('', pinImg)
            # cv2.waitKey()
            # cv2.destroyAllWindows
            # cv2.imshow('', _img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # anom = checkPin(pinImg, kernel)
            # decision.append(anom)
        print()
        

if __name__=="__main__":
    main()      