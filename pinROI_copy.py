import torch
import cv2
import numpy as np
# from concurrent.futures import ThreadPoolExecutor
import time
# import copy
# from loguru import logger
# from src.utils.colors import *
# import glob
import emoji
from colorama import Fore, Style 
from ultralytics import YOLO
import json
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's available
import matplotlib.pyplot as plt
plt.ioff()



class MODEL:
    """
    m07_PinROI model class. This class is implimented to get the desired foreground ROIs (Pin-ROI, Bur-ROI) of Pin surfaces.
    """
    def __init__(self, model_path='', confidence_threshold=0.25,
                 brightnessConfig = '/home/gpuadmin/Desktop/WEIGHTS/08_Abrasion/brightness_config.json', 
                 height=3000, width=4096, warmup_runs=3, 
                 device_id=0):  #TODO 3: Update the model_path, Add the parameters you need which can be modified from GUI
        """
        Initialize the model.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Confidence threshold for detections.
            warmup_runs (int): Number of dummy inference runs to warm up the model.
        """
        num_gpus = torch.cuda.device_count()
        self.conf_thres = confidence_threshold
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and num_gpus > device_id else "cuda:0")
        
        self.h = height
        self.w = width

        self.kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        with open(brightnessConfig, "r") as file:
                self.BThres = json.load(file)

        self.model = YOLO(model_path)
 
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(warmup_runs):
            _ = self.model.predict(source=dummy_image, conf=self.conf_thres, device=self.device, verbose=False)

    def processOBB(self, yoloOBB_rslt, conf_thres):
        # s = time.time()
        preds_list = []
        for result in yoloOBB_rslt:
            boxes  = result.obb.xyxyxyxy.cpu().numpy().reshape(-1,8)
            clsz   = result.obb.cls.cpu().numpy()
            conf   = result.obb.conf.cpu().numpy()
            _preds = np.hstack([clsz[:,None], boxes, conf[:,None]])
            preds_list.append(_preds)
        if preds_list:
            preds = np.vstack(preds_list)
            return preds[preds[:, -1] > conf_thres]
        else:
            return np.empty((0, 10))
    
    def checkROI(self, roi):
        if roi is None or roi.size == 0 or min(roi.shape[:2]) == 0:
            return False 
        roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi   = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)[1]  # More efficient
        roi   = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        white = np.sum(roi == 255)
        total = roi.size  # To
        black = total - white
        ratio = white/black if black!=0 else 1   
        return ratio > 0.9
    
    def pinPost_process(self, img, pinPreds, numpins=21, pinHeight=900, pinWidth=80):
        if pinPreds.shape[0] <= numpins:
            return pinPreds
        pinPreds = pinPreds[np.argsort(pinPreds[:, 6])]
        # filter small preds (preds with less height and width may be predicted as extra unnecessary pred)
        hs = pinPreds[:,3] - pinPreds[:,7]
        ws = pinPreds[:,4] - pinPreds[:,6]
        valid_mask = (hs > pinHeight) & (ws > pinWidth)
        pinPreds   = pinPreds[valid_mask] 
        # merge overlapping preds
        if pinPreds.shape[0]>numpins:
            topL_xs    = pinPreds[:,6]
            separation = np.diff(topL_xs)
            merge_idx  = np.where(separation < 9)[0]
            pinPreds[merge_idx] = (pinPreds[merge_idx] + pinPreds[merge_idx + 1]) / 2
            pinPreds = np.delete(pinPreds, merge_idx + 1, axis=0)         
        # filter unnecessary preds in between pins
        if pinPreds.shape[0] > numpins:
            topLxs = pinPreds[1:, 6]
            topRxs = pinPreds[:-1, 4]
            separation = topLxs - topRxs
            idx = np.where(separation < 60)[0]
            if idx.size > 0:
                rois_to_check = [
                            (pinPreds[i], img[int(pinPreds[i][7]):int(pinPreds[i][3]), int(pinPreds[i][6]):int(pinPreds[i][2])])
                            for i in idx
                                        ] + [
                            (pinPreds[i + 1], img[int(pinPreds[i + 1][7]):int(pinPreds[i + 1][3]),
                            int(pinPreds[i + 1][6]):int(pinPreds[i + 1][2])])
                            for i in idx
                            ]
                roi_states = np.array([self.checkROI(roi) for _, roi in rois_to_check])

                nonPinIdxs = np.ones(pinPreds.shape[0], dtype=bool)
                idx_pairs = np.concatenate([idx, idx + 1])
                nonPinIdxs[idx_pairs] = roi_states

                pinPreds = pinPreds[nonPinIdxs]               
        return pinPreds
    
    def processPins(self, img, preds, surface, adjustment=0, num_pins=21):
        '''Post-processes the predicted yolo ROI for pins and separate the upper and lower pins in Front view'''
        pin1 = None

        def pinSeparation(_preds, numPins=num_pins):
            '''method to separate the upper and lower pins in front views'''
            lowerPin_thres = np.max(_preds[:,[5,7]])
            # pin1 seperation
            pin1  = preds[np.all(preds[:,[1,3,5,7]]<lowerPin_thres, axis=1)]
            pin1  = pin1[pin1[:,0].argsort()]
            if pin1.shape[0]>numPins:
                pin1 = self.pinPost_process(img, pin1, numPins)
            # cap the top y points (top left y, top right y) within a y threshold (190)
            topThres      = np.average(pin1[:,[5,7]])
            pin1[:,[5,7]] = np.maximum(pin1[:,[5,7]],topThres)
            pin1[:,[5,7]] = np.minimum(pin1[:,[5,7]],topThres+10)
            # cap the bottom y points (bottom left y, bottom right y) with in a threshold range (710-720)
            # choose a threshold based on the model prediction
            thres = np.max(pin1[:,[1,3]])-5
            pin1[:,[1,3]] = np.maximum(pin1[:,[1,3]],thres)
            pin1[:,[1,3]] = np.minimum(pin1[:,[1,3]],thres)
            return pin1    

        if surface=='Front11':
            pin1 = pinSeparation(preds)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment

        elif surface=='Front12':
            pin1 = pinSeparation(preds)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment

        elif surface=='Front21':
            pin1 = pinSeparation(preds)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment

        elif surface=='Front22':
            pin1 = pinSeparation(preds)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment
   
        elif surface=='Top11' or surface=='Top21':
            pin1 = preds

            if pin1.shape[0]!=num_pins:
                pin1 = self.pinPost_process(img, pin1, num_pins)

            topThres      = np.average(pin1[:,[5,7]])
            pin1[:,[5,7]] = np.maximum(pin1[:,[5,7]],topThres)
            pin1[:,[5,7]] = np.minimum(pin1[:,[5,7]],topThres)

            bottomThres   = np.average(pin1[:,[1,3]]) + 10
            pin1[:,[1,3]] = np.maximum(pin1[:,[1,3]],bottomThres)
            pin1[:,[1,3]] = np.minimum(pin1[:,[1,3]],bottomThres)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment
        elif surface=='Top12' or surface=='Top22':
            pin1 = preds

            if pin1.shape[0]!=num_pins:
                pin1 = self.pinPost_process(img, pin1, num_pins)

            topThres      = np.average(pin1[:,[5,7]])
            pin1[:,[5,7]] = np.maximum(pin1[:,[5,7]],topThres)
            pin1[:,[5,7]] = np.minimum(pin1[:,[5,7]],topThres)

            bottomThres   = np.average(pin1[:,[1,3]]) + 2
            pin1[:,[1,3]] = np.maximum(pin1[:,[1,3]],bottomThres)
            pin1[:,[1,3]] = np.minimum(pin1[:,[1,3]],bottomThres)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment
        return pin1    

    def generateMask(self, img, obbBoxes, itrs=1, callfrom=''):
        # s = time.time()
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        for box in obbBoxes:
            if callfrom=='segment_pin':
                x1, y1 = max(0, int(box[6])-3), max(0, int(box[7]))
                x2, y2 = min(width, int(box[2]+3)), min(height, int(box[3]))
            elif callfrom=='segment_burr':
                x1, y1 = max(0, int(box[6])-1), max(0, int(box[7]))
                x2, y2 = min(width, int(box[2])+1), min(height, int(box[3]))   
            
            patch = img[y1:y2,x1:x2]    
            patch = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)[1]
            # patch = cv2.morphologyEx(patch, cv2.MORPH_DILATE, self.kernel, iterations=1)

            h, w  = patch.shape

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=4)

            # post process to remove very small connected components and extending bigger components from top and bottom if necessary
            if num_labels>0:
                for i in range(1, num_labels):  # Skip background (label 0)
                    if stats[i, cv2.CC_STAT_AREA] < 2000:
                        continue
                    else:
                        x, y, _w, _h, area = stats[i]
                        _x1 = int(x+3)
                        _x2 = int(x+_w-7)
                        _y1 = int(y)
                        # _y2 = int(y+_h)
                        # if _y1>5:
                        #     _y1 = 2   
                        #     # y1 = 5
                        # if _y2<((h/2)+10):
                        _y2 = h-3
                        
                        patch[_y1:_y2, _x1:_x2] = 255    

            # # post processing for patch top
            # y = int(0.3 * h)
            # x_coords = np.where(patch[y, :] > 0)[0]

            # left_x = np.min(x_coords) + 1 if x_coords.size > 0 else 3
            # right_x = np.max(x_coords) - 1 if x_coords.size > 0 else w - 3    

            # patch[0:y, left_x:right_x] = 255

            # # post processing for patch Bottom
            y = int(0.7 * h)
            x_coords = np.where(patch[y, :] > 0)[0]

            left_x = np.min(x_coords) + 2 if x_coords.size > 0 else _x1 + 2
            right_x = np.max(x_coords) - 2 if x_coords.size > 0 else _x2 + 6
            patch[y:h-3, left_x:right_x] = 255

            # patch[y:height-1, left_x:right_x] = 255
            patch = cv2.morphologyEx(patch, cv2.MORPH_DILATE, self.kernel, iterations=itrs)
            if callfrom=='segment_pin':
                patch = cv2.morphologyEx(patch, cv2.MORPH_ERODE, self.kernel, iterations=itrs+2)
            elif callfrom=='segment_burr':
                patch = cv2.morphologyEx(patch, cv2.MORPH_ERODE, self.kernel, iterations=itrs)
            contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>0:
                component_mask = np.zeros_like(patch)
                cv2.drawContours(component_mask, contours, -1, 255, thickness=cv2.FILLED)
                patch[component_mask == 255] = 255
            # patch = cv2.erode(patch, self._kernel, iterations=1)
            
            mask[y1:y2, x1:x2] = patch>0

        # print('Generate mask method time: ', time.time()-s)
        return mask
    
    def segment(self, img, obbBoxes, roi='Pin'):

        if roi=='Pin':
            # s = time.time()
            roiBox = self.ROIbox(obbBoxes)
            roiImg  = np.zeros_like(img, dtype=np.uint8)
            x1, y1, x2, y2 = roiBox
            roiImg[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            mask = self.generateMask(img, obbBoxes, itrs=1, callfrom='segment_pin')
            roiImg = cv2.bitwise_and(roiImg, roiImg, mask=mask.astype(np.uint8))
            # print('Segment method time (Pins): ', time.time()-s)
        elif roi=='Bur':
            # s = time.time()
            roiBox = self.ROIbox(obbBoxes, padding=(60,0))
            # adjustment for bur ROI
            roiBox[1] += 10
            roiBox[3] -= 10

            roiImg = np.zeros_like(img, dtype=np.uint8)
            x1, y1, x2, y2 = roiBox
            roiImg[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            mask = self.generateMask(img, obbBoxes, itrs=1, callfrom='segment_burr')
            mask = ~mask
            roiImg = cv2.bitwise_and(roiImg, roiImg, mask=mask.astype(np.uint8))
            # roiImg[y1:3, x1:x2] = 0
            # roiImg[y2-5]
            # print('Segment method time (Burr): ', time.time()-s)      
        return roiImg, roiBox, mask    

    def ROIbox(self, obbPreds, width=4096, height=3000, padding=(100,20)):
        # s = time.time()

        # Extract x and y coordinates
        coords = obbPreds[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        x_coords, y_coords = coords[:, :4], coords[:, 4:]

        # Compute min and max with padding
        x_min, x_max = np.min(x_coords, axis=1) - padding[0], np.max(x_coords, axis=1) + padding[0]
        y_min, y_max = np.min(y_coords, axis=1), np.max(y_coords, axis=1)

        # Clip values within bounds
        x_min, x_max = np.clip([x_min, x_max], 0, width - 1).astype(int)
        y_min, y_max = np.clip([y_min, y_max], 0, height - 1).astype(int)

        # print(f'Method ROIbox process time: {time.time() - s:.6f} sec')

        return [min(x_min), min(y_min), max(x_max), max(y_max)]
    
    def filterPreds(self,predictions, roi):
        x_min, y_min, x_max, y_max = roi

        # Reshape predictions to (N, 4, 2) for easy indexing of points
        points = predictions.reshape(-1, 4, 2)

        # Check if all points lie within the ROI
        inside_x = (points[:, :, 0] >= x_min) & (points[:, :, 0] <= x_max)  # x-coordinates within ROI
        inside_y = (points[:, :, 1] >= y_min) & (points[:, :, 1] <= y_max)  # y-coordinates within ROI
        inside_roi = np.all(inside_x & inside_y, axis=1)  # Check all corners are inside

        # Filter predictions based on the ROI condition
        preds = predictions[inside_roi]

        return preds
    
    def preProcess_img(self, img, gamma=0.5):
        r, g, b = img[:,:,2], img[:,:,1], img[:,:,0]

        _r = np.array(255*(r / 255) ** gamma, dtype = 'uint8')
        _g = np.array(255*(g / 255) ** gamma, dtype = 'uint8')
        _b = np.array(255*(b / 255) ** gamma, dtype = 'uint8')
        return np.dstack((_b, _g, _r))

    def preProcess_patch(self, patch_img, surfaceName='', patchIdx=0):

        if 'Front' in surfaceName:
            bThres = self.BThres[surfaceName][f'{patchIdx}'] - 2
        elif 'Top' in surfaceName:
            bThres = self.BThres[surfaceName][f'{patchIdx}'] - 1.2      
        
        _patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
        # print(surfaceName)
        h, s, v = cv2.split(_patch_img)

        brightness = np.sum(v)/(v.shape[0]*v.shape[1])

        diff = brightness - bThres

        if diff>0:
            scale_factor = 1 + (diff/25)
            if scale_factor > 1.7:
                # TODO: give warning of too much higher brightness
                scale_factor = 1.7
                v = np.clip(v * 0.9, 0, 255).astype(np.uint8)
        
            v = np.array(255*(v / 255) ** scale_factor, dtype = 'uint8')
        
        h = np.mod(h * 1.15, 180).astype(np.uint8)
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        
        patch_img = cv2.merge([h,s,v])
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_HSV2BGR)

        patch_img[:,:,2] = np.clip(patch_img[:,:,2]*1.25, 0, 255).astype(np.uint8)

        # patch = cv2.convertScaleAbs(patch_img, alpha=alpha, beta=beta)

        return patch_img

    def pinROI(self, img, pin1):
        # s = time.time() 
        # Here only the pin region will be retained in the image
        # if 'Front' in surface:
        # fullPin_img, pinBox = self.segment(img, pin1, 'Pin')
            # pinBox   = [upperPin_box]
        # elif 'Top' in surface:
        #     fullPin_img, pinBox = self.segment(img, pin1, 'Pin')
            # pinBox   = [pinBox]
        # print(f'Method pinROI process time: {time.time()-s}')
        return self.segment(img, pin1, 'Pin')    
    
    def burrROI(self, img, pin):
        # s = time.time()
        # img = self.preProcess_img(img)
        # print(f'Method burrROI process time: {time.time()-s}')
        return self.segment(img, pin, 'Bur')

    def drawBox(self, img, boxes, obb=False, boxType='xywh', color=(0,255,0), thickness=1, text=''):
        # s = time.time()
        if obb:
            for box in boxes:
                if box.shape[0]==10:
                    _box = np.array([[int(box[1]), int(box[2])],
                                    [int(box[3]), int(box[4])],
                                    [int(box[5]), int(box[6])],
                                    [int(box[7]), int(box[8])]])
                elif box.shape[0]==8:
                    _box = np.array([[int(box[0]), int(box[1])],
                                    [int(box[2]), int(box[3])],
                                    [int(box[4]), int(box[5])],
                                    [int(box[6]), int(box[7])]])    
                _box = _box.reshape(-1,1,2)
                img  = cv2.polylines(img, [_box], isClosed=True, color=color, thickness=thickness)
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
                    x1 = int(box[0])
                    x2 = int(box[2])
                    y1 = int(box[1])
                    y2 = int(box[3])

                    img  = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    text = f'{text}:'
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                    label_bg_top_left = (x1, y1 - h - 5)
                    label_bg_bottom_right = (x1 + w, y1)
                    cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)
        # print(f'Method drawBox process time: {time.time()-s}')
        return img    
    
    def getNameAndRoi(self, realName, meanROI):
        x1, y1, x2, y2 = meanROI

        # roi_params = {
        #     "Top-pin-2nd_auto_0":  (-830, 15, 55, 920, 1890, 1910, -15, 550, 650, 8, 1275, 1290), # done
        #     "Top-pin-2nd_auto_1":  (-830, 30, 90, 860, 1890, 1915, -20, 550, 650, 8, 1275, 1290), # done
        #     "Top-pin_auto_0":  (-810, 0, 25, 870, 1790, 1870, -15, 420, 450, 35, 1260, 1275), # done
        #     "Top-pin_auto_1":  (-830, 0, 15, 890, 1810, 1860, -40, 430, 460, 45, 1260, 1275),  # done
        #     "Front-pin-2nd_auto_0": (-830, 25, 90, 910, 1900, 1915, -320, -3, 0, 150, 880, 910), # done
        #     "Front-pin-2nd_auto_1": (-830, 40, 90, 910, 1901, 1919, -320, -1, 3, 150, 880, 910), # done
        #     "Front-pin_auto_0": (-830, 1, 25, 910, 1880, 1910, -350, 40, 95, 150, 930, 990),   # done
        #     "Front-pin_auto_1": (-830, 1, 25, 910, 1870, 1900, -330, 40, 95, 150, 930, 990),   # done
        # }
        # for lineB---------------------------------------------------------------------------------
        # "Top-pin_auto_0":  (-1720, 100, 220, 1930, 3950, 4040, -120, 1240, 1310, 100, 2920, 2980)
        # "Top-pin_auto_1":  (-1720, 100, 200, 1950, 3930, 4010, -100, 1210, 1280, 100, 2890, 2950)
        #-------------------------------------------------------------------------------------------
        # for lineA--------------------------------------------------------------------------------------------
        # "Top-pin_auto_0":  (-1720, 100, 220, 1930, 3950, 4040, -120, 890, 990, 100, 2560, 2650)
        # "Top-pin_auto_1":  (-1720, 100, 200, 1950, 3930, 4010, -100, 890, 990, 100, 2570, 2620)
        #------------------------------------------------------------------------------------------------------
        roi_params = {
            "Top-pin-2nd_auto_0":  (-1750, 100, 180, 1970, 3950, 4030, -100, 1520, 1640, 20, 3002, 3005), # done
            "Top-pin-2nd_auto_1":  (-1750, 120, 220, 1970, 3990, 4060, -110, 1520, 1640, 20, 3002, 3005), # done
            "Top-pin_auto_0":  (-1720, 100, 220, 1930, 3950, 4040, -120, 1240, 1310, 100, 2920, 2980), # done
            "Top-pin_auto_1":  (-1720, 100, 200, 1950, 3930, 4010, -100, 1210, 1280, 100, 2890, 2950), # done
            "Front-pin-2nd_auto_0": (-1800, 50, 260, 1950, 3990, 4080, -750, -5, 0, 350, 1810, 1950), # done
            "Front-pin-2nd_auto_1": (-1800, 50, 260, 1950, 3990, 4080, -750, -5, 0, 350, 1810, 1950), # done
            "Front-pin_auto_0": (-1750, 50, 220, 1990, 3960, 4070, -780, 120, 250, 450, 2150, 2250), # done
            "Front-pin_auto_1": (-1800, 80, 200, 1920, 3960, 4070, -780, 100, 200, 400, 1980, 2080), # done
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
            "Top-pin-2nd_auto_0": ("Top12", roi, -2, 21),
            "Top-pin-2nd_auto_1": ("Top22", roi, -2, 19),
            "Top-pin_auto_0": ("Top21", roi, -2, 19),
            "Top-pin_auto_1": ("Top11", roi, -2, 21),
        }

        return mapping.get(realName, (None, None, None, None))    
    
    def getBoxes(self, roi):
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
                    x1 = int(x+4)
                    x2 = int(x+_w-4)
                    y1 = int(y+2)
                    y2 = int(y+_h-1)
                    if y1>50:
                        patch[5:y1+5,x1:x2]=255
                        # y1 = 5
                    if y2<((h/2)+10):
                        patch[y2:h-1, x1:x2]
                        # y2 = h-1
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        if len(contours)>0:
            for contour in contours:
                if cv2.contourArea(contour) < 2500:
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
    
    def recoverPinBoxes(self, img, preds=[], roi=[]):
        _boxes = []
        sort_idx = np.argsort(preds[:, 6])
        preds = preds[sort_idx]
        # check if the pins are missing at the start or end and recover
        startPred = preds[0].astype(int)
        endPred   = preds[-1].astype(int)
        if startPred[6]-roi[0] > 280:
            patch = img[startPred[7]:startPred[1], roi[0]+20:startPred[0]-5]
            boxes = self.getBoxes(patch)
            if boxes.shape[0]>0:
                boxes[:,::2] += roi[0]+20    
                # boxes[:,1::2] += roi[1]+20
                boxes[:,[1, 3, 5,7]] = startPred[1],startPred[3],startPred[5],startPred[7]
                _boxes.append(boxes)
        if roi[2]-endPred[4] > 350:
            patch = img[endPred[5]:endPred[1], endPred[4]+5:roi[2]-20]
            boxes = self.getBoxes(patch)
            if boxes.shape[0]>0:
                boxes[:,::2] += endPred[4]+5    
                # boxes[:,1::2] += patchTop[1]
                boxes[:,[1, 3, 5,7]] = endPred[1], endPred[3], endPred[5], endPred[7]
                _boxes.append(boxes)        
        topLxs = preds[:,6]
        separation = np.diff(topLxs)
        idx = np.where(separation>300)[0]
        # if numpins ==19:
        #     idx = idx[idx<14]
        for _idx in idx:
            pred1, pred2 = preds[_idx], preds[_idx + 1]
            patchTop, patchBottom = (int(pred1[4] + 4), int(pred1[5])), (int(pred2[0] - 4), int(pred2[1]))
            patch = img[patchTop[1]:patchBottom[1], patchTop[0]:patchBottom[0]]
            boxes = self.getBoxes(patch)
            if boxes.shape[0]>0:
                boxes[:,::2] += patchTop[0]    
                boxes[:,1::2] += patchTop[1]
                boxes[:,[1, 3, 5,7]] = startPred[1],startPred[3],startPred[5],startPred[7]
                _boxes.append(boxes)
        return np.concatenate(_boxes, axis=0) if _boxes else np.empty((0, 8))
    
    def getCrop_offsets_1(self, pins=[], roiBox=[]):
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
    
    def getCrop_offsets_2(self, pins=[], roiBox=[]):
        pinPoses = []
        x1, y1, w, h = int(roiBox[0]), int(roiBox[1]), int(roiBox[2]), int(roiBox[3])
        for pin in pins:
            x2 = int((pin[2]+10))
            pinPoses.append([x1, y1, x2,h])
            x1 = x2
        pinPoses.append([x2, y1, w, h])
        return pinPoses
    
    def getPatch(self, img, pinPoses):
        patch = img[pinPoses[1]:pinPoses[3],pinPoses[0]:pinPoses[2]]
        return patch
    
    def patchify_1(self, img=[], mask=[], pinPreds=None, roi=[]):
        patches, patch_positions, maskPatches = [], [], []
        if len(pinPreds)>=20:
            pinPoses = self.getCrop_offsets_1([pinPreds[3], pinPreds[7], pinPreds[11], pinPreds[15], pinPreds[18]], roi)
            for pos in pinPoses:
                patch = self.getPatch(img,pos)
                patches.append(cv2.resize(patch, (640,640)))
                maskPatch = self.getPatch(mask, pos)
                maskPatches.append(cv2.resize(maskPatch, (640,640)))
                patch_positions.append(pos)
            # cv2.destroyAllWindows()        
        elif (len(pinPreds))<=19:            
            pinPoses = self.getCrop_offsets_1([pinPreds[2], pinPreds[6], pinPreds[10], pinPreds[14], pinPreds[17]], roi)
            for pos in pinPoses:
                patch = self.getPatch(img,pos)
                patches.append(cv2.resize(patch, (640,640)))
                maskPatch = self.getPatch(mask, pos)
                maskPatches.append(cv2.resize(maskPatch, (640,640)))
                patch_positions.append(pos)           
        # cv2.destroyAllWindows()
        return {"patches": patches, "patch_positions": np.array(patch_positions), "patch_masks":maskPatches}

    def patchify_2(self, img=[], mask=[], pinPreds=None, roi=[], surface=''):
        patches, patch_positions, maskPatches = [], [], []
        if len(pinPreds)>=20:
            pinPoses = self.getCrop_offsets_2([pinPreds[3], pinPreds[7], pinPreds[11], pinPreds[15], pinPreds[18]], roi)
            for pos in pinPoses:
                patch = self.getPatch(img,pos)
                patch = self.preProcess_patch(patch, surfaceName=surface)
                patches.append(cv2.resize(patch, (640,640)))
                patchMask = self.getPatch(mask, pos)
                maskPatches.append(cv2.resize(patchMask, (640,640)))
                patch_positions.append(pos)   
        elif (len(pinPreds))<=19:
            # patches, patch_positions = [], []
            pinPoses = self.getCrop_offsets_2([pinPreds[2], pinPreds[6], pinPreds[10], pinPreds[14], pinPreds[17]], roi)
            for pos in pinPoses:
                patch = self.getPatch(img,pos)
                patch = self.preProcess_patch(patch, surfaceName=surface)
                patches.append(cv2.resize(patch, (640,640)))
                patchMask = self.getPatch(mask, pos)
                maskPatches.append(cv2.resize(patchMask, (640,640)))
                patch_positions.append(pos)
        return {"patches": patches, "patch_positions": np.array(patch_positions), "patch_masks":maskPatches}

    def process(self, image: np.ndarray, previous_detections = [], info={}): #TODO 7: You will receive image and the detection results from previous model (if this model is not the first one)

        # Check if the input image is valid
        if image is None:
            raise ValueError("Input image is empty or None.")
        
        # start_time = time.time()
        img = np.copy(image)

        # img = cv2.resize(np.copy(image), (self.w, self.h), interpolation=cv2.INTER_AREA)

        realName = info['Input']
        # print(realName)

        # _s = time.time()
        results = self.model.predict(img, imgsz=768, conf=self.conf_thres, device=self.device, verbose=False)
        # print('Model Inference Time: ', time.time()-_s)

        preds = self.processOBB(results, self.conf_thres)
        
        if preds.shape[0]>10:
            # print(preds.shape[0])
            preds = preds[:,1:9]
            preds = np.clip(preds, [0, 0, 0, 0, 0, 0, 0, 0], [self.w-1, self.h-1]*4)

            _roi = (np.add.reduce(preds[:, [6, 7, 2, 3]], axis=0) / preds.shape[0]).astype(int)
            name, roi, adj, numPins = self.getNameAndRoi(realName, _roi)
            
            # img = self.drawBox(img, [roi], boxType='xyxy', thickness=3)
            # img = self.drawBox(img, preds, obb=True, thickness=2, color=(0, 0, 255))
            # plt.close('all')
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show(block=True)
            # For ROI  verification, uncomment below lines to verify and adjust the roi box for filtering unwanted box outside ROI
            # _img = copy.deepcopy(img)
            # _img = self.drawBox(_img, [_roi], boxType='xyxy')
            # _img = self.drawBox(_img, [roi], boxType='xyxy')
            # cv2.imwrite('img.jpg', _img)

            if roi is not None:
                _preds = self.filterPreds(preds, roi)
                if _preds.shape[0]>0:
                    preds = _preds
                elif _preds.shape[0]==0:
                    print(f"\t\t{(Fore.RED)}{(Style.BRIGHT)}{(emoji.emojize(':warning:'))} {' WARNING'} All predictions removed with PinROI based filtering, Please Check input image and adjust PinROI {Style.RESET_ALL}")
                    preds = preds
                    
            pin1 = self.processPins(img=img, preds=preds, surface=name, adjustment=adj, num_pins=numPins)
            
            # # manual tests for missing pin recovery
            # # pin1 = np.delete(pin1, [10,12], axis=0)

            if pin1.shape[0]<numPins:
                recBoxes = self.recoverPinBoxes(img, pin1, roi)
                if recBoxes.any(): pin1 = np.concatenate((pin1,recBoxes), axis=0)    

            # img = self.drawBox(img, [roi], boxType='xyxy', thickness=3)
            # img = self.drawBox(img, pin1, obb=True, thickness=2, color=(0, 0, 255))
            # plt.close('all')
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show(block=True)
            
            # For pin prediction verification
            # uncomment below lines to verify all the pins are predicted after prediction and post-process steps
            # _img = copy.deepcopy(img)
            # _img = self.drawBox(_img, pin1, obb=True)
            # cv2.imwrite('img.jpg', _img)

            # _s = time.time()
            Pin_img, pinROI_box, Pinmask = self.pinROI(img=img, pin1=pin1)
            burImg, burROI_box, Burmask  = self.burrROI(img=img, pin=pin1)
            return Pin_img, pinROI_box, Pinmask, burImg, burROI_box, Burmask, pin1
            
        elif preds.shape[0]<=10:
            print(f"\t\t{(Fore.RED)}{(Style.BRIGHT)}{(emoji.emojize(':warning:'))} {' WARNING'} Nothing Detected, Please Check input image {Style.RESET_ALL}")
            return None, None, None, None, None, None, None
            # return None         

if __name__ == "__main__":
    import tqdm
    import argparse
    import os

    def arguments():
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
    
    args = arguments()
    # Load the model
    model = MODEL(model_path=args.roiModel, confidence_threshold=0.2, warmup_runs=3,
                  brightnessConfig='/home/zafar/old_pc/data_sets/robot-project-datasets/code-integration/brightness_config.json')

    dataPath = args.dataPath
    savePath = args.savePath

    with open(dataPath, 'r') as f:
        imgPaths = f.readlines()
    
    idx = 0
    # Load an image
    for imagePath in tqdm.tqdm(imgPaths):
        print(imagePath)
        # imagePath   = "/home/zafar/old_pc/data_sets/robot-project-datasets/code-integration/AIRobot/LinA_latest/Front-pin_auto_0/Input-Front-pin_auto_0__Cam-Front__Camera-FNO-33__ProductID-2__.png"
        imagePath   = imagePath.strip()
        surfaceName = imagePath.split('/')[-2]
        # try:
        #     surfaceName = imagePath.split(f'/LineB_')[1].split('.png')[0].rsplit('-', 1)[0]
        # except:
        #     surfaceName = imagePath.split(f'/LineA_')[1].split('.png')[0].rsplit('-', 1)[0]
        # surfaceName = imagePath.split('Input-')[1].split('__Cam')[0]
        # surfaceName = imagePath.split('/')[-1].split('.png')[0].rsplit('-', 1)[0]
        # surfaceName  = '_'.join(imagePath.split('/')[-1].split('.png')[0].split('_')[1:])
        # surfaceName  = '_'.join(imagePath.split('/')[-1].split('.webp')[0].split('_')[1:])
        # surfaceName  = imagePath.split('/')[-1].split('.webp')[0].split('-', 1)[1]
        info={'Input':surfaceName}
        image = cv2.imread(imagePath)
        plt.imshow(image)
        plt.show()
        plt.close('all')

        # Run inference
        # model.process(image, info=info)
        # print()    
        Pin_img, pinROI_box, Pinmask, burImg, burROI_box, Burmask, pinPreds = model.process(image, info=info)
        
        pin_img = model.drawBox(Pin_img, [pinROI_box], boxType='xyxy', thickness=3)
        pin_img = model.drawBox(pin_img, pinPreds, obb=True, thickness=2, color=(0, 0, 255))
        plt.imshow(Pin_img)
        plt.show()
        plt.close('all')

        bur_img = model.drawBox(burImg, [burROI_box], boxType='xyxy', thickness=3)
        plt.imshow(bur_img)
        plt.show()
        plt.close('all')

        
        # sorted_indices = np.argsort(pinPreds[:, 6])
        # pinPreds = pinPreds[sorted_indices]
        # print(len(pinPreds))

        # # Pin image Preparation------------------------------------------------------------------------------------
        # Pinmask = (Pinmask*255).astype(np.uint8)
        # patchedImg = model.patchify_2(Pin_img, mask=Pinmask, pinPreds=pinPreds, roi=pinROI_box, surface=surfaceName)
        # patches, patchPosz, patchmasks = patchedImg.get('patches'), patchedImg.get('patch_positions'), patchedImg.get('patch_masks')

        # # write patched images
        # for i, patch in enumerate(patches):
        #     patchMask = patchmasks[i]
        #     imgSavePath = os.path.join(savePath, 'images')
        #     img_name    = os.path.join(imgSavePath, f'pinImg_{idx}.png')
        #     maskSavePath = os.path.join(savePath, 'masks')
        #     mask_name    = os.path.join(maskSavePath, f'pinImg_{idx}.png')

        #     idx += 1

        #     cv2.imwrite(img_name, patch)
        #     cv2.imwrite(mask_name, patchMask)
        
        #-----------------------------------------------------------------------------------------------------------
        # Burr image preparation------------------------

        # adjust roi to igonre black coated region
        # if 'Front' in imagePath:
        #     burROI_box[1] += 10
        #     burROI_box[3] -= 150
        # elif 'Top' in imagePath:
        #     burROI_box[1] += 5
        #     burROI_box[3] -= 10

        # Burmask = (Burmask*255).astype(np.uint8)    

        # # preprocess brr roi image
        # _burImg = cv2.cvtColor(burImg, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(_burImg)
        # if 'Top' in imagePath:
        #     v_gamma = np.clip(255*(v/255)**0.5, 0, 255)
        # elif 'Front' in imagePath:
        #     v_gamma = np.clip(255*(v/255)**0.4, 0, 255)        
        # v_gamma_bright = np.clip(v_gamma*1.5, 0, 255).astype(np.uint8)
        # v_gamma_bright[v_gamma_bright<20] = 0
        # _burImg_filterd_th_gamma = cv2.merge([h, s, v_gamma_bright])
        # _burImg_filterd_th_gamma = cv2.cvtColor(_burImg_filterd_th_gamma, cv2.COLOR_HSV2BGR)

        # # cv2.imshow('',_burImg_filterd_th_gamma)
        # # cv2.waitKey()
        # # cv2.destroyAllWindows()
        
        # # patchify
        # patchedImg = model.patchify_1(_burImg_filterd_th_gamma, mask=Burmask, pinPreds=pinPreds, roi=burROI_box)
        # patches, patchPosz, patchmasks = patchedImg.get('patches'), patchedImg.get('patch_positions'), patchedImg.get('patch_masks')

        # # write patched images
        # for i, patch in enumerate(patches):
        #     patchMask = patchmasks[i]
        #     imgSavePath = os.path.join(savePath, 'images')
        #     img_name    = os.path.join(imgSavePath, f'burImg_{idx}.png')
        #     maskSavePath = os.path.join(savePath, 'masks')
        #     mask_name    = os.path.join(maskSavePath, f'burImg_{idx}.png')

        #     idx += 1

        #     cv2.imwrite(img_name, patch)
        #     cv2.imwrite(mask_name, patchMask)
    #-----------------------------------------------------------------------------------------------------------------------        