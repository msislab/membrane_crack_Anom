import cv2
import json
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
        "Top-pin-2nd_auto_0":  (-830, 15, 55, 920, 1890, 1910, -15, 625, 660, 8, 1280, 1290), # done
        "Top-pin-2nd_auto_1":  (-830, 30, 90, 860, 1890, 1915, -20, 625, 660, 8, 1280, 1290), # done
        "Top-pin_auto_0":  (-810, 0, 25, 870, 1790, 1870, -15, 420, 450, 35, 1250, 1275), # done
        "Top-pin_auto_1":  (-830, 0, 15, 890, 1810, 1860, -40, 430, 460, 45, 1250, 1275),  # done
        "Front-pin-2nd_auto_0": (-830, 25, 90, 910, 1900, 1915, -320, -3, 0, 150, 870, 900), # done
        "Front-pin-2nd_auto_1": (-830, 40, 90, 910, 1901, 1919, -320, -1, 5, 150, 870, 900), # done
        "Front-pin_auto_0": (-830, 1, 25, 910, 1880, 1910, -350, 50, 85, 150, 920, 980),   # done
        "Front-pin_auto_1": (-830, 1, 25, 910, 1870, 1900, -330, 50, 90, 150, 920, 980),   # done
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
    # roiH = h-y1
    for pin in pins:
        x2 = int((pin[2]+10))
        # y2 = int(y1 + (roiH/2))
        pinPoses.append([x1, y1, x2,h])
        # pinPoses.append([x1, y2, x2,h])
        x1 = x2
    # pinPoses.append([x1, int(y1/2), w, h])
    # y2 = int(y1 + (roiH/2))
    pinPoses.append([x2, y1, w, h])
    # pinPoses.append([x1, y2, w, h])
    return pinPoses    

def patchify(img=[], mask=[], pinPreds=None, roi=[], getMask=False):
    mask = (mask*255).astype(np.uint8)
    if len(pinPreds)>=20:
        patches, patch_positions, masks = [], [], []
        pinPoses = getCrop_offsets([pinPreds[3], pinPreds[7], pinPreds[11], pinPreds[15], pinPreds[18]], roi)
        # _patch   = None
        for pos in pinPoses:
            patch = getPatch(img,pos)
            patches.append(cv2.resize(patch, (640,640)))
            patch_positions.append(pos)
            # brightness = brightness_level(patch)
            # print('brightness level is: ', brightness)
            # if brightness>thres:
            #     patch = preProcess_img(patch, gamma=gamma)
            if getMask:
                _patch = getPatch(mask, pos)
                masks.append(cv2.resize(_patch, (640,640)))
            # cv2.rectangle(img, (pos[0], pos[1]), (pos[2],pos[3]), (0,0,255))
            # cv2.imshow('', img)
            # cv2.waitKey()
            # if save:
            #     _idx = f'{idx}'.zfill(4)
            #     idx += 1
            #     name = os.path.join(f'{savePath}/imgs', f'{_idx}.png')
            #     if not os.path.exists(f'{savePath}/imgs'):
            #         os.makedirs(f'{savePath}/imgs')
            #     cv2.imwrite(name, cv2.resize(patch, (640,640)))
            #     if _patch is not None:
            #         name = os.path.join(f'{savePath}/masks', f'{_idx}.png')
            #         if not os.path.exists(f'{savePath}/masks'):
            #             os.makedirs(f'{savePath}/masks')
            #         cv2.imwrite(name, cv2.resize(_patch, (640,640)))

            # else:
            
        # cv2.destroyAllWindows()        
    elif (len(pinPreds))<=19:
        patches, patch_positions, masks = [], [], []
        pinPoses = getCrop_offsets([pinPreds[2], pinPreds[6], pinPreds[10], pinPreds[14], pinPreds[17]], roi)
        # _patch   = None
        for pos in pinPoses:
            patch = getPatch(img,pos)
            patches.append(cv2.resize(patch, (640,640)))
            patch_positions.append(pos)
            # brightness = brightness_level(patch)
            # print('brightness level is: ', brightness)
            # if brightness>thres:
            #     patch = preProcess_img(patch, gamma=gamma)
            if getMask:
                _patch = getPatch(mask, pos)
                masks.append(cv2.resize(_patch, (640,640)))
            # cv2.rectangle(img, (pos[0], pos[1]), (pos[2],pos[3]), (0,0,255))
            # cv2.imshow('', img)
            # cv2.waitKey()
            # if save:
            #     _idx = f'{idx}'.zfill(4)
            #     idx += 1
            #     name = os.path.join(f'{savePath}/imgs', f'{_idx}.png')
            #     if not os.path.exists(f'{savePath}/imgs'):
            #         os.makedirs(f'{savePath}/imgs')
            #     cv2.imwrite(name, cv2.resize(patch, (640,640)))
            #     if _patch is not None:
            #         name = os.path.join(f'{savePath}/masks', f'{_idx}.png')
            #         if not os.path.exists(f'{savePath}/masks'):
            #             os.makedirs(f'{savePath}/masks')
            #         cv2.imwrite(name, cv2.resize(_patch, (640,640)))

            # else:        
    else:
        NotImplementedError    
    # cv2.destroyAllWindows()
    return {"patches": patches, "patch_positions": np.array(patch_positions), "masks":np.array(masks)}

def getBoxes(roi):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    h,w,_  = roi.shape

    patch = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    patch = cv2.inRange(patch, 36, 255)
    patch = cv2.erode(patch, kernel, iterations=1)
    patch[0:150,0:w] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=4)
    
    # post process to remove very small connected components and extending bigger components from top and bottom if necessary
    if num_labels>0:
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] < 2500:
                patch[labels == i] = 0
            else:
                x, y, _w, _h, area = stats[i]
                x1 = int(x+8)
                x2 = int(x+_w-1)
                y1 = int(y+2)
                y2 = int(y+_h-1)
                if y1>50:
                    patch[8:y1+5,x1:x2]=255
                    # y1 = 5
                if y2<((h/2)+10):
                    patch[y2:h-1, x1:x2]
                    # y2 = h-1
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if len(contours)>0:
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            rect = cv2.minAreaRect(contour)

            box = cv2.boxPoints(rect)

            box = box[np.argsort(box[:,1])]
            if box[0,0]>box[1,0]: box[[0,1]] = box[[1,0]]
            if box[2,0]<box[3,0]: box[[2,3]] = box[[3,2]]
            box = box[::-1].flatten()
            # if box[5]>=8:
            #     box[5]=1
            # if box[1]<=h-10:
            #     box[1] = h-1                
            boxes.append(box)
    return np.vstack(boxes) if boxes else np.empty((0, 4))      

def recoverPinBoxes(img, preds=[], numpins=21, roi=[]):
    _boxes = []
    sort_idx = np.argsort(preds[:, 6])
    preds = preds[sort_idx]
    # check if the pins are missing at the start or end and recover
    startPred = preds[0].astype(int)
    endPred   = preds[-1].astype(int)
    if startPred[6]-roi[0] > 100:
        patch = img[startPred[7]:startPred[1], roi[0]+20:startPred[0]-5]
        boxes = getBoxes(patch)
        if boxes.shape[0]>0:
            boxes[:,::2] += roi[0]+20    
            # boxes[:,1::2] += roi[1]+20
            boxes[:,[1, 3, 5,7]] = startPred[1],startPred[3],startPred[5],startPred[7]
            _boxes.append(boxes)
    if roi[2]-endPred[4] > 120:
        patch = img[endPred[5]:endPred[1], endPred[4]+5:roi[2]-20]
        boxes = getBoxes(patch)
        if boxes.shape[0]>0:
            boxes[:,::2] += endPred[4]+5    
            # boxes[:,1::2] += patchTop[1]
            boxes[:,[1, 3, 5,7]] = endPred[1], endPred[3], endPred[5], endPred[7]
            _boxes.append(boxes)        
    topLxs = preds[:,6]
    separation = np.diff(topLxs)
    idx = np.where(separation>115)[0]
    # if numpins ==19:
    #     idx = idx[idx<14]
    for _idx in idx:
        pred1, pred2 = preds[_idx], preds[_idx + 1]
        patchTop, patchBottom = (int(pred1[4] + 4), int(pred1[5])), (int(pred2[0] - 4), int(pred2[1]))
        patch = img[patchTop[1]:patchBottom[1], patchTop[0]:patchBottom[0]]
        boxes = getBoxes(patch)
        if boxes.shape[0]>0:
            boxes[:,::2] += patchTop[0]    
            boxes[:,1::2] += patchTop[1]
            boxes[:,[1, 3, 5,7]] = startPred[1],startPred[3],startPred[5],startPred[7]
            _boxes.append(boxes)
    return np.concatenate(_boxes, axis=0) if _boxes else np.empty((0, 8))    

def preProcess_img(img, gamma=0, thres = [0.7, 0.5]):
    processed_img = np.zeros_like(img, dtype=np.uint8)
    thres = thres[gamma]
    r = thres*img[:,:,2]
    g = thres*img[:,:,1]
    b = thres*img[:,:,0]
    # _r = np.array(250*(r / 255) ** gamma, dtype = 'uint8')
    # _g = np.array(250*(g / 255) ** gamma, dtype = 'uint8')
    # _b = np.array(250*(b / 255) ** gamma, dtype = 'uint8')

    processed_img[:,:,2] = r
    processed_img[:,:,1] = g
    processed_img[:,:,0] = b

    return processed_img

def preProcess_patch(patch_img, scale_factor=[0.9,0.9,0.9, 2.1], alpha=1.5, beta=-1):
    # patch = cv2.convertScaleAbs(patch_img, alpha=alpha, beta=beta)
    # lab = cv2.cvtColor(patch_img, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)

    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9,9))
    # l = clahe.apply(l)

    # # # Merge channels and convert back to BGR
    # lab = cv2.merge((l, a, b))
    # patch_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    b, g, r = patch_img[:,:,0], patch_img[:,:,1], patch_img[:,:,2]

    b = np.array(255*(b / 255) ** scale_factor[-1])
    g = np.array(255*(g / 255) ** scale_factor[-1])
    r = np.array(255*(r / 255) ** scale_factor[-1])
    
    # b = cv2.equalizeHist(b)
    # g = cv2.equalizeHist(g)
    # r = cv2.equalizeHist(r)
    # b[b>230] = 230
    # g[g>230] = 220
    # r[r>230] = 220
    b = np.clip(scale_factor[0]*b, 0, 255).astype(np.uint8)
    g = np.clip(scale_factor[1]*g, 0, 255).astype(np.uint8)
    r = np.clip(scale_factor[2]*r, 0, 255).astype(np.uint8)

    patch_img[:,:,0] = b
    patch_img[:,:,1] = g
    patch_img[:,:,2] = r

    # patch = cv2.convertScaleAbs(patch_img, alpha=alpha, beta=beta)

    return patch_img

def brightness_level(img):
    '''
    This function computes the brightness level of an image based on the intensities present in the image   \n
    Args:
        img: np.ndarray
    Return
        brightness: brightness level int he image, float between 0 and 2  
    '''
    sum1 = np.sum(img[:,:,0])
    sum2 = np.sum(img[:,:,1])
    sum3 = np.sum(img[:,:,2])
    cum_sum = sum1 + sum2 + sum3
    # Formula to calculate the brightness of an image:
    # 2 * (cumulative sum of all pixel intensities(in every channels) / total pixels in image x 3x 255)
    brightness = 2*(cum_sum/(img.shape[0]*img.shape[1]*3*255))         
    return brightness

def getPin(img, pinOBB, kernel):
    pinTop    = [int(pinOBB[6])-2, int(pinOBB[7])]
    pinBottom = [int(pinOBB[2])+2, int(pinOBB[3])]
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

def main():
    _args = args()
    w,h   = 1920,1280
    
    with open('brightness_config.json', "r", encoding="utf-8") as f:
        config = json.load(f)

    brightness_adjustment_thres = {
        "Front": [0.8,0.8,0.95, 1.1],
        "Top": [0.75,0.75,0.9, 1.6]
    }
    
    img_savePath = os.path.join(_args.savePath, 'images')
    mask_savePath = os.path.join(_args.savePath, 'masks')
    if not os.path.exists(img_savePath):
        os.makedirs(img_savePath, exist_ok=True)
    if not os.path.exists(mask_savePath):
        os.makedirs(mask_savePath, exist_ok=True)    

    roiModel = ROI_model(_args.roiModel)

    with open(_args.dataPath, 'r') as f:
        pathList = f.readlines()    

    name1  = _args.dataPath.split('/')[-1].split('.txt')[0]
    
    idx = 0
    brightness = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0}
    numImgs = 0
    for imgPath in tqdm.tqdm(pathList):
        imgPath = imgPath.strip()
        print(imgPath)
        numImgs += 1
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # imgPath = '/home/zafar/old_pc/data_sets/robot-project-datasets/normal-pin-data/Normal_connectors/Top-pin_auto_1/Input-Top-pin_auto_1__Cam-Top__Camera-FNO-171__ProductID-11__FA.png'
        img = cv2.resize(cv2.imread(imgPath), (w,h))
        # cv2.imwrite('img.jpg', img)
        name    = imgPath.split('/')[-1].split('Input-')[-1].split('__Cam')[0]
        
        preds = roiModel.predYOLO_obb(img, _args.roiConf)        
        preds = preds[:,1:9]
        preds[:,::2]  = np.clip(preds[:,::2], 0, w-1)
        preds[:,1::2] = np.clip(preds[:,1::2], 0, h-1)

        # manual tests for missing preds (pins)
        # sorted_indices = np.argsort(preds[:, 6])
        # preds           = preds[sorted_indices]
        # preds = np.delete(preds, [0, 1, 18], axis=0)

        _roi = np.mean(preds[:, [6, 7, 2, 3]], axis=0).astype(int)
        print(name)
        _name, roi, adj, numPins = getNameAndRoi(name, _roi)

        if roi is not None:
           preds   = roiModel.filterPreds(preds, roi)   

        pin1, _ = roiModel.processPins(img=img, preds=preds, surface=_name, adjustment=adj, num_pins=numPins)   
        
        # manual tests for missing pin recovery
        # sorted_indices = np.argsort(pin1[:, 6])
        # pin1           = pin1[sorted_indices]
        # pin1 = np.delete(pin1, [0,1,18], axis=0)
        if pin1.shape[0]<numPins:
            recBoxes = recoverPinBoxes(img, pin1, numPins, roi)
            if recBoxes.any(): pin1 = np.concatenate((pin1,recBoxes), axis=0)
        
        # # /home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Segregated_Burr_Anomalies/Segregated_burr_anomalie2_01_27/Input-Top-pin_auto_1__Cam-Top__Camera-FNO-456__ProductID-27.png
        
        sorted_indices = np.argsort(pin1[:, 6])
        pin1           = pin1[sorted_indices]

        roiImg, roiBox, mask = roiModel.pinROI(img, _name, pin1)

        # for i,pin in enumerate(pin1):
        #     pinImg = getPin(roiImg, pin, kernel)
        #     # cv2.imwrite(f'pin_{i}.jpg',pinImg)
        #     # print(surface[0], self.brightness_level(pinImg))
        #     pinImg = cv2.resize(pinImg, (200,640))
        #     _idx = f'{idx}'.zfill(4)
        #     idx += 1
        #     name = f'{savePath}/{_idx}.png'
        #     cv2.imwrite(name, pinImg)

        # roiImg = roiModel.drawBox(roiImg, pin1, obb=True)
        # cv2.rectangle(img, (roi[0], roi[1]), (roi[2],roi[3]), color=(0,0,255))
        # cv2.imshow('', roiImg)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print()
        # roiImg = cv2.multiply(roiImg, np.array([2,2,2], dtype=np.float32))
        # roiImg = np.clip(roiImg, 0, 255).astype(np.uint8)


        # patchedImg, idx = patchify(img=roiImg,
        #                 pinPreds=pin1, roi= roiBox,
        #                 idx=idx)

        patchedImg = patchify(img=roiImg, mask=mask,
                        pinPreds=pin1, roi= roiBox, getMask=True)
        
        # _, idx = patchify(img=roiImg, mask=mask,
        #                       pinPreds=pin1, roi= roiBox,
        #                       save=True, saveMask=True, savePath=savePath,
        #                       idx=idx)
        
        # _, idx = patchify(img=roiImg,
        #                       pinPreds=pin1, roi= roiBox,
        #                       save=True, savePath=savePath,
        #                       thres=brightness_thres, gamma=gamma, idx=idx)
        patches, patchPosz, patchMasks = patchedImg.get('patches'), patchedImg.get('patch_positions'), patchedImg.get('masks')
        for i, patch in enumerate(patches):
            patchMask = patchMasks[i]
            # bLevel = brightness_level(patch)
            # if bLevel < config[f'{name}'][f'{i}']:
            if 'Front' in name:
                scale = brightness_adjustment_thres['Front']
            elif 'Top' in name:
                scale = brightness_adjustment_thres['Top']    
            patch = preProcess_patch(patch, scale_factor=scale)
            _idx = f'{idx}'.zfill(4)
            if 'Front' in name:
                surface = 'Front'
            elif 'Top' in name:
                surface = 'Top'    
            imgName = os.path.join(img_savePath, f'lineA_{surface}_{_idx}.png')
            maskName = os.path.join(mask_savePath, f'lineA_{surface}_{_idx}.png')
            idx += 1
            cv2.imwrite(imgName, patch)
            cv2.imwrite(maskName, patchMask)
        print()        
            # brightness[f"{i}"] +=bLevel
        # print(brightness)
    # avg = {key:val/numImgs for (key , val) in brightness.items()}
    
    # with open(f'{name1}.json', 'w') as fp:
    #     json.dump(avg, fp)
    # print()        

        
        
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