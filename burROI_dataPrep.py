import cv2
import numpy as np
import tqdm
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

# Optimized function with reduced redundant checks and improved efficiency
def getNameAndRoi(realName, meanROI):
    x1, y1, x2, y2 = meanROI

    roi_params = {
        "Top-pin-2nd_auto_0":  (-800, 35, 80, 900, 1850, 1900, -10, 660, 700, 5, 1280, 1290), # done
        "Top-pin-2nd_auto_1":  (-800, 80, 150, 840, 1875, 1910, -15, 660, 695, 5, 1280, 1290), # done
        "Top-pin_auto_0":  (-775, 5, 50, 850, 1760, 1820, -10, 470, 490, 20, 1140, 1180), # done
        "Top-pin_auto_1":  (-800, 3, 20, 870, 1760, 1810, -30, 450, 480, 30, 1150, 1180),  # done
        "Front-pin-2nd_auto_0": (-800, 70, 135, 870, 1880, 1915, -300, -3, 1, 120, 750, 810), # done
        "Front-pin-2nd_auto_1": (-800, 90, 185, 900, 1901, 1919, -300, 1, 20, 120, 760, 810), # done
        "Front-pin_auto_0": (-800, 20, 85, 900, 1815, 1890, -330, 90, 120, 130, 910, 950),   # done
        "Front-pin_auto_1": (-800, 10, 70, 900, 1800, 1830, -310, 90, 140, 100, 880, 910),
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

def getPin_dim_inPixels(pinPreds):
    cumsum = 0
    idx = 0
    for pred in pinPreds:
        pinDim = pred[4] - pred[6]
        cumsum += pinDim
        idx += 1
    return cumsum/idx

def getPatch(img, pinPoses):
    patch = img[pinPoses[1]:pinPoses[3],pinPoses[0]:pinPoses[2]]
    return patch

def getCrop_offsets(pins=[], roiBox=[]):
    pinPoses = []
    x1, y1, w, h = int(roiBox[0]), int(roiBox[1]), int(roiBox[2]), int(roiBox[3])
    roiH = h-y1
    for pin in pins:
        x2 = int(((pin[0]+pin[2])/2)-5)
        y2 = int(y1 + (roiH/2))
        pinPoses.append([x1, y1, x2,y2])
        pinPoses.append([x1, y2, x2,h])
        x1 = x2
    # pinPoses.append([x1, int(y1/2), w, h])
    y2 = int(y1 + (roiH/2))
    pinPoses.append([x1, y1, w, y2])
    pinPoses.append([x1, y2, w, h])
    return pinPoses    

def patchify(img=[], mask=[], pinPreds=None, roi=[], save=False, saveMask=False, savePath='', idx=None):
    if len(pinPreds)==21:
        patches, patch_positions = [], []
        pinPoses = getCrop_offsets([pinPreds[3], pinPreds[7], pinPreds[11], pinPreds[15], pinPreds[18]], roi)
        _patch   = None
        for pos in pinPoses:
            patch = getPatch(img,pos)
            if saveMask:
                _patch = getPatch(mask, pos)
            # cv2.rectangle(img, (pos[0], pos[1]), (pos[2],pos[3]), (0,0,255))
            # cv2.imshow('', img)
            # cv2.waitKey()
            if save:
                _idx = f'{idx}'.zfill(4)
                idx += 1
                name = os.path.join(f'{savePath}/imgs', f'{_idx}.png')
                if not os.path.exists(f'{savePath}/imgs'):
                    os.makedirs(f'{savePath}/imgs')
                cv2.imwrite(name, cv2.resize(patch, (640,640)))
                if _patch is not None:
                    name = os.path.join(f'{savePath}/masks', f'{_idx}.png')
                    if not os.path.exists(f'{savePath}/masks'):
                        os.makedirs(f'{savePath}/masks')
                    cv2.imwrite(name, cv2.resize(_patch, (640,640)))

            else:
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pos)
        # cv2.destroyAllWindows()        
    elif (len(pinPreds))==19:
        patches, patch_positions = [], []
        pinPoses = getCrop_offsets([pinPreds[2], pinPreds[6], pinPreds[10], pinPreds[14], pinPreds[17]], roi)
        _patch   = None
        for pos in pinPoses:
            patch = getPatch(img,pos)
            if saveMask:
                _patch = getPatch(mask, pos)
            # cv2.rectangle(img, (pos[0], pos[1]), (pos[2],pos[3]), (0,0,255))
            # cv2.imshow('', img)
            # cv2.waitKey()
            if save:
                _idx = f'{idx}'.zfill(4)
                idx += 1
                name = os.path.join(f'{savePath}/imgs', f'{_idx}.png')
                if not os.path.exists(f'{savePath}/imgs'):
                    os.makedirs(f'{savePath}/imgs')
                cv2.imwrite(name, cv2.resize(patch, (640,640)))
                if _patch is not None:
                    name = os.path.join(f'{savePath}/masks', f'{_idx}.png')
                    if not os.path.exists(f'{savePath}/masks'):
                        os.makedirs(f'{savePath}/masks')
                    cv2.imwrite(name, cv2.resize(_patch, (640,640)))

            else:
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pos)        
    else:
        NotImplementedError    
    # cv2.destroyAllWindows()
    return {"patches": patches, "patch_positions": np.array(patch_positions)}, idx

def getBoxes(roi):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    h,w,_  = roi.shape

    patch = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    patch = cv2.threshold(patch, 35, 255, cv2.THRESH_BINARY)[1]
    patch = cv2.erode(patch, kernel, iterations=1)

    # post processing for patch top
    y = int(0.1 * h)
    x_coords = np.where(patch[y, :] > 0)[0]

    try:
        left_x = np.min(x_coords)+1
        right_x = np.max(x_coords)-1
    except:
        left_x  = 3
        right_x = w-3    

    patch[0:y, left_x:right_x] = 255

    # post processing for patch Bottom
    y = int(0.55 * h)
    x_coords = np.where(patch[y, :] > 0)[0]

    try:
        left_x = np.min(x_coords)+1
        right_x = np.max(x_coords)-1
    except:
        left_x  = 3
        right_x = w-3

    patch[y:h-1, left_x:right_x] = 255
    
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = np.zeros((8,))
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Need at least 5 points to fit a rotated rectangle
            continue
        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)

        box = box[np.argsort(box[:,1])]
        if box[2,0]<box[3,0]: box[[2,3]] = box[[3,2]]
        box = box[::-1].flatten()
        if box[5]>=8:
            box[5]=1
        if box[1]<=h-10:
            box[1] = h-1                
        # box = box.astype(np.int32)  # Convert to integer coordinates
        boxes = np.vstack((boxes, box))
    if boxes.ndim>1: return boxes[1:,:]
    else: return np.array([])    

def recoverPinBoxes(img, preds=[], numpins=21):
    preds  = preds[np.argsort(preds[:, 6])]
    topLxs = preds[:,6]
    separation = np.diff(topLxs)
    idx = np.where(separation>115)[0]
    if numpins ==19:
        idx = idx[idx<15]
    _boxes = np.zeros(8,)
    for _idx in idx:
        pred1 = preds[_idx]
        pred2 = preds[_idx+1]

        patchTop    = int(pred1[4]+2), int(pred1[5])
        patchBottom = int(pred2[0]-2), int(pred2[1])
        patch = img[patchTop[1]:patchBottom[1], patchTop[0]:patchBottom[0]]
        boxes = getBoxes(patch)
        if boxes.shape[0]>0:
            boxes[:,::2] += patchTop[0]
            boxes[:,1::2] += patchTop[1]
            _boxes = np.vstack((_boxes,boxes))
    if _boxes.ndim>1: return _boxes[1:,:]
    else: np.array([])    

def preProcess_img(img, gamma=0.5):
    processed_img = np.zeros_like(img, dtype=np.uint8)
    r = img[:,:,2]
    g = img[:,:,1]
    b = img[:,:,0]
    _r = np.array(255*(r / 255) ** gamma, dtype = 'uint8')
    _g = np.array(255*(g / 255) ** gamma, dtype = 'uint8')
    _b = np.array(255*(b / 255) ** gamma, dtype = 'uint8')

    processed_img[:,:,2] = _r
    processed_img[:,:,1] = _g
    processed_img[:,:,0] = _b

    return processed_img

def main():
    _args = args()
    w,h   = 1920,1280

    savePath = os.path.join(_args.savePath, 'Burr_dataset_demo_3')
    if not os.path.exists(savePath):
        os.makedirs(savePath, exist_ok=True)

    roiModel = ROI_model(_args.roiModel)

    with open(_args.dataPath, 'r') as f:
        pathList = f.readlines()    

    idx = 0
    for imgPath in tqdm.tqdm(pathList):
        imgPath = imgPath.strip()
        print(imgPath)
        # imgPath = '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Segregated_Burr_Anomalies/Segregated_bur _anomalies1_1_27/Input-Front-pin_auto_0__Cam-Front__Camera-FNO-807__ProductID-42.png'
        img = cv2.resize(cv2.imread(imgPath), (w,h))
        name    = imgPath.split('/')[-1].split('Input-')[-1].split('__Cam')[0]
        
        preds = roiModel.predYOLO_obb(img, _args.roiConf)        
        preds = preds[:,1:9]
        preds[:,::2]  = np.clip(preds[:,::2], 0, w-1)
        preds[:,1::2] = np.clip(preds[:,1::2], 0, h-1)

        _roi = np.mean(preds[:, [6, 7, 2, 3]], axis=0).astype(int)
        print(name)
        _name, roi, adj, numPins = getNameAndRoi(name, _roi)

        if roi is not None:
           preds   = roiModel.filterPreds(preds, roi)   

        pin1, _ = roiModel.processPins(img=img, preds=preds, surface=_name, adjustment=adj, num_pins=numPins)   
        
        if pin1.shape[0]<numPins:
            recBoxes = recoverPinBoxes(img, pin1, numPins)
            if recBoxes.ndim>0: pin1 = np.vstack((pin1,recBoxes))
        
        # # /home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Segregated_Burr_Anomalies/Segregated_burr_anomalie2_01_27/Input-Top-pin_auto_1__Cam-Top__Camera-FNO-456__ProductID-27.png

        _img = preProcess_img(img)
        
        sorted_indices = np.argsort(pin1[:, 6])
        pin1           = pin1[sorted_indices]

        roiImg, roiBox, mask = roiModel.burrROI(_img, pin1)

        # roiImg = roiModel.drawBox(roiImg, pin1, obb=True)
        # cv2.rectangle(img, (roi[0], roi[1]), (roi[2],roi[3]), color=(0,0,255))
        # cv2.imshow('', roiImg)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print()
        # roiImg = cv2.multiply(roiImg, np.array([2,2,2], dtype=np.float32))
        # roiImg = np.clip(roiImg, 0, 255).astype(np.uint8)


        # _, idx = patchify(img=roiImg, mask=mask,
        #                       pinPreds=pin1, roi= roiBox,
        #                       save=True, saveMask=True, savePath=savePath,
        #                       idx=idx)
        
        _, idx = patchify(img=roiImg,
                              pinPreds=pin1, roi= roiBox,
                              save=True, savePath=savePath,
                              idx=idx)
        # patches, _, patchPosz = patchedImg.get('patches'), patchedImg.get('img_shape'), patchedImg.get('patch_positions')
        # cv2.imshow('', img)
        # cv2.waitKey()
        # cv2.imshow('', roiImg)
        # cv2.waitKey()
        # cv2.imshow('', mask)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

# missed pin detection cases at greater than 0.4 model confidence
# '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Segregated_Burr_Anomalies/Segregated_burr_anomalie2_01_27/Input-Top-pin-2nd_auto_0__Cam-Top__Camera-FNO-374__ProductID-22.png' 
# imgPath = '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Segregated_Burr_Anomalies/Segregated_burr_anomalie2_01_27/Input-Top-pin-2nd_auto_0__Cam-Top__Camera-FNO-33__ProductID-2.png' 
# imgPath = '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Segregated_Burr_Anomalies/Segregated_burr_anomalie2_01_27/Input-Top-pin_auto_0__Cam-Top__Camera-FNO-734__ProductID-42.png'