import cv2
import numpy as np
import argparse
import copy
from ultralytics import YOLO
import glob
import os
from roi import ROI_model

def args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--dataPath', type=str, default='',
                        help='Provide the path of data.txt file')
    parser.add_argument('--roiModel', type=str, default=None,
                        help='Provide model.pt path')
    parser.add_argument('--roiConf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose a gpu device')
    parser.add_argument('--imgSize', type=int, default=640,
                        help='Choose image size for model inference')
    parser.add_argument('--savePath', type=str, default='',
                        help='Provide a path to save the resultant image')
    args = parser.parse_args()
    return args

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
                text = f'{box[0]}:'
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                label_bg_top_left = (x1, y1 - h - 5)
                label_bg_bottom_right = (x1 + w, y1)
                cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)

        elif boxType=='xyxy':   # TODO: implement the box rendering for xyxy box coordinates
            for box in boxes:
                x1 = int(box[1])
                x2 = int(box[3])
                y1 = int(box[2])
                y2 = int(box[4])

                img  = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                text = f'{box[0]}:'
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                label_bg_top_left = (x1, y1 - h - 5)
                label_bg_bottom_right = (x1 + w, y1)
                cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)      
    return img

def makeLabelList(preds):
    labelList = []
    for pred in preds:
        line = str(0)  + ' {} {} {} {} {} {} {} {}\n'.format(pred[0], pred[1],
                                                     pred[2], pred[3],
                                                     pred[4], pred[5],
                                                     pred[6], pred[7])
        labelList.append(line)
    return labelList    

def getNameAndRoi(realName):
    mapping = {
    "Front-pin-2nd_auto_0": ("Front12", None, -1),
    "Front-pin-2nd_auto_1": ("Front22", None, -1),
    "Front-pin_auto_0": ("Front21", [115, 95, 1850, 875], -1),
    "Front-pin_auto_1": ("Front11", [75, 95, 1835, 875], -1),
    "Top-pin-2nd_auto_0": ("Top12", None, -3),
    "Top-pin-2nd_auto_1": ("Top22", None, -3),
    "Top-pin_auto_0": ("Top21", [125, 500, 1900, 1230], -3),
    "Top-pin_auto_1": ("Top11", [125, 500, 1900, 1230], -3),}
    return mapping.get(realName, (None, None, None))

def main():
    _args = args()

    savePath = os.path.join(_args.savePath, 'ROI_dataset')
    if not os.path.exists(savePath):
        os.makedirs(savePath, exist_ok=True)

    roiModel = ROI_model(_args.roiModel)

    with open(_args.dataPath, 'r') as f:
        pathList = f.readlines()

    idx = 0
    for imgPath in pathList:
        imgPath = imgPath.strip()
        img = cv2.resize(cv2.imread(imgPath), (1920,1280))

        preds = roiModel.predYOLO_obb(img, _args.roiConf)        
        preds = preds[:,1:9]

        name    = imgPath.split('/')[-1].split('Input-')[-1].split('__Cam')[0]
        _name, roi, adj = getNameAndRoi(name)

        pin1, pin2 = roiModel.processPins(img, preds, _name)

        if pin1 is not None and pin2 is not None:
            _preds = np.vstack((pin1,pin2))
        else:
            _preds = pin1

        _preds[:,::2]  /= 1920
        _preds[:,1::2] /= 1280

        labels = makeLabelList(_preds)

        imgName = f'anom_{idx}.png'
        saveName = os.path.join(savePath, imgName)
        cv2.imwrite(saveName, img)
        txtName = f'anom_{idx}.txt'
        saveName = os.path.join(savePath, txtName)
        with open(saveName, 'w') as f:
            f.writelines(labels)
        idx += 1    

        # img = drawBox(img, preds, obb=True)
        # cv2.imshow('', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()