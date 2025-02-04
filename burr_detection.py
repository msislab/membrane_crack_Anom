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

def parseArgs():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--dataPath', type=str, default='',
                        help='Provide img directory path')
    parser.add_argument('--detModel', type=str, default=None,
                        help='Provide model.pt path')
    parser.add_argument('--roiModel', type=str, default=None,
                        help='Provide model.pt path')
    parser.add_argument('--detConf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--roiConf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose a gpu device')
    parser.add_argument('--imgSize', type=int, default=640,
                        help='Choose box type for annotation')
    args = parser.parse_args()
    return args

def main():
    args  = parseArgs()
    imgs  = glob.glob(f'{args.dataPath}/*.png')
    imgs  = natsorted(imgs)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    detModel = YOLO(args.detModel)

    roiModel = ROI_model(args.roiModel)

    # model.to(device)

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

        fullImg, _, burImg, _, ROIbox, pinPreds = roiModel.burrROI(img=img, surface=surface)
        # fullPin_img, Pin_img, Pin_img2, pinBox, pin1, pin2, fullMask = roiModel.pinROI(img=img, surface=surface)
        # _img = drawBox(_img, np.array([ROIbox]), boxType='xyxy')
        sorted_indices = np.argsort(pinPreds[:, 6])
        pinPreds = pinPreds[sorted_indices]
        cv2.imshow('', burImg)
        cv2.waitKey()
        # cv2.imshow('', _img)
        # cv2.waitKey()
        cv2.destroyAllWindows()

        patchedImg = patchify(burImg, surface=surface, pinPreds=pinPreds)

        patches, oriImg_shape, patchPosz = patchedImg.get('patches'), patchedImg.get('img_shape'), patchedImg.get('patch_positions')

        modelPreds = detModel.predict(patches, conf=args.detConf, device=device, verbose=False)
        preds   = processOBB(yoloOBB_rslt=modelPreds, patchPosz=patchPosz)
        for i, pred in enumerate(preds):
            x1,y1,x2,y2 = patchPosz[i]
            patchImg = patches[i]
            patchImg = cv2.resize(patchImg, (x2-x1, y2-y1))
            annotImg = drawBox(patchImg, pred, obb=True)
            cv2.imshow('', annotImg)
            cv2.waitKey()
            cv2.destroyAllWindows()
            # cv2.imwrite(f"patch{i}.jpg", annotImg)
        imPreds = patchToimPred(preds, patchPosz)
        img_ = drawBox(burImg, imPreds, obb=True)
        cv2.imshow('', img_)
        cv2.waitKey()
        cv2.destroyAllWindows()
        imPreds[:,:-1:2]  += ROIbox[0]
        imPreds[:,1:-1:2] += ROIbox[1]

        img_ = drawBox(img, imPreds, obb=True)
        cv2.imshow('', img_)
        cv2.waitKey()
        cv2.destroyAllWindows() 
        s = time.time()
        # preds = predYOLO_obb(img=_img, args=args, model=detModel, device=device)
        # e = time.time()
        # total_time+=e-s
        # annotate image with preds
        # annotImg = drawBox(img=_img, preds=preds, obb=True)

if __name__=="__main__":
    main()        